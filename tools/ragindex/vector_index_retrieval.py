import os
import re
import time
import json
import logging
import asyncio
from typing import Annotated, Optional, List, Dict, Any, Tuple
from urllib.parse import urlparse

import aiohttp
from azure.identity import (
    ManagedIdentityCredential,
    AzureCliCredential,
    ChainedTokenCredential,
)
from connectors import AzureOpenAIClient

from .types import (
    VectorIndexRetrievalResult,
    MultimodalVectorIndexRetrievalResult,
    DataPointsResult,
)

# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------


async def _get_azure_search_token() -> str:
    """
    Acquires an Azure Search access token using chained credentials.
    """
    try:
        credential = ChainedTokenCredential(
            ManagedIdentityCredential(), AzureCliCredential()
        )
        # Wrap the synchronous token acquisition in a thread.
        token_obj = await asyncio.to_thread(
            credential.get_token, "https://search.azure.com/.default"
        )
        return token_obj.token
    except Exception as e:
        logging.error("Error obtaining Azure Search token.", exc_info=True)
        raise Exception("Failed to obtain Azure Search token.") from e


async def _perform_search(
    url: str, headers: Dict[str, str], body: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Performs an asynchronous HTTP POST request to the given URL with the provided headers and body.
    Returns the parsed JSON response.

    Raises:
        Exception: When the request fails or returns an error status.
    """
    async with aiohttp.ClientSession() as session:
        try:
            async with session.post(url, headers=headers, json=body) as response:
                if response.status >= 400:
                    text = await response.text()
                    error_message = f"Error {response.status}: {text}"
                    logging.error(f"[_perform_search] {error_message}")
                    raise Exception(error_message)
                return await response.json()
        except Exception as e:
            logging.error("Error during asynchronous HTTP request.", exc_info=True)
            raise Exception("Failed to execute search query.") from e


# -----------------------------------------------------------------------------
# Main Functions
# -----------------------------------------------------------------------------


async def vector_index_retrieve(
    input: Annotated[str, "Optimized query string"],
    security_ids: str = "anonymous",
) -> Annotated[
    VectorIndexRetrievalResult,
    "A Pydantic model containing the search results as a string",
]:
    """
    Performs a keyword/semantic search against Azure Cognitive Search and maps freeform
    user queries to specific document fields or returns all fields for 'show all' requests.
    """
    # Load settings
    search_top_k = int(os.getenv("AZURE_SEARCH_TOP_K", 3))
    use_semantic = os.getenv("AZURE_SEARCH_USE_SEMANTIC", "false").lower() == "true"
    semantic_config = os.getenv(
        "AZURE_SEARCH_SEMANTIC_SEARCH_CONFIG", "my-semantic-config"
    )
    service = os.getenv("AZURE_SEARCH_SERVICE", "search0jdjja")
    index = os.getenv("AZURE_SEARCH_INDEX", "purchase-orders-from-blob")
    api_version = os.getenv("AZURE_SEARCH_API_VERSION", "2023-07-01-Preview")

    # Field label mapping from the search service indexes - Spanish to English
    field_labels = {
        "Id_de_Requisicion": "Requisition ID",
        "Numero_de_Caso": "Case Number",
        "Costo_Unitario_Estimado_de_Articulo": "Estimated Unit Cost",
        "Descripcion_de_Articulo": "Description",
        "Marca_de_Articulo": "Brand",
        "Modelo_de_Articulo": "Model",
        "Garantia_de_Articulo": "Warranty",
        "Unidad_de_Medida": "Unit",
        "Cantidad": "Quantity",
        "Fecha_Recibo_de_Requisicion": "Received Date",
        "Numero_de_Requisicion": "Requisition Number",
        "Titulo_de_Requisicion": "Requisition Title",
        "Categoria_de_Requisicion": "Category",
        "SubCategoria_de_Requisicion": "Subcategory",
        "Agencia": "Agency",
        "Nombre_de_Agencia_de_Entrega": "Delivery Agency",
        "Metodo_de_Adquisicion": "Acquisition Method",
        "Costo_Estimado_Total_de_Orden_de_Articulo": "Estimated Total Cost",
        "Numero_de_Contrato": "Contract Number",
        "Costo_Unitario_Final_de_Articulo": "Final Unit Cost",
        "Costo_Final_de_Orden_de_Articulo": "Final Order Cost",
        "Numero_de_Orden_de_Compra": "Purchase Order Number",
        "Nombre_de_Archivo_de_Orden_de_Compra": "Order File Name",
        "Url_de_Archivo_de_Orden_de_Compra": "Order File URL",
        "Nombre_de_Suplidor": "Supplier",
        "Telefono_de_Contacto_de_Suplidor": "Supplier Phone",
        "Email_de_Suplidor": "Supplier Email",
    }
    all_fields = list(field_labels.keys())

    # Automatic mapping of field keywords from labels and field names
    from typing import List, Tuple

    dynamic_field_map: List[Tuple[List[str], str]] = []
    for field, label in field_labels.items():
        # keywords are words in the English label and parts of the field name
        keywords = set(label.lower().split()) | set(field.lower().split("_"))
        dynamic_field_map.append((list(keywords), field))

    # Specific field keywords
    field_map = [
        (
            [
                "unit cost",
                "estimated unit cost",
                "costo unitario",
                "costo unitario estimado",
                "costo unitario estimado del artículo",
            ],
            "Costo_Unitario_Estimado_de_Articulo",
        ),
        (
            ["description", "descripcion", "descripción", "descripción del artículo"],
            "Descripcion_de_Articulo",
        ),
        (["supplier", "suplidor", "proveedor"], "Nombre_de_Suplidor"),
        (
            [
                "supplier phone",
                "phone",
                "telephone",
                "supplier telephone",
                "telefono",
                "numero de contacto del suplidor",
                "numero de teléfono del suplidor" "telefono de contacto",
            ],
            "Telefono_de_Contacto_de_Suplidor",
        ),
        (
            [
                "supplier email",
                "email",
                "correo",
                "correo electrónico",
                "correo electrónico del suplidor",
            ],
            "Email_de_Suplidor",
        ),
        (
            [
                "warranty",
                "garantia",
                "garantía" "tiempo de garantia",
                "tiempo de garantía",
            ],
            "Garantia_de_Articulo",
        ),
        (
            ["estimated cost", "costo estimado"],
            "Costo_Estimado_Total_de_Orden_de_Articulo",
        ),
        (["final cost", "costo final"], "Costo_Final_de_Orden_de_Articulo"),
        (["requisition id", "id de requisicion"], "Id_de_Requisicion"),
        (
            [
                "case number",
                "numero de caso",
                "caso numero",
                "número de caso",
                "caso número",
            ],
            "Numero_de_Caso",
        ),
        (
            ["requisition number", "numero de requisicion", "número de requisición"],
            "Numero_de_Requisicion",
        ),
        (
            [
                "requisition title",
                "title of the requisition",
                "titulo de requisicion",
                "título de requisión",
            ],
            "Titulo_de_Requisicion",
        ),
        (["category", "categoria"], "Categoria_de_Requisicion"),
        (
            ["subcategory", "subcategoria", "subcategoría"],
            "SubCategoria_de_Requisicion",
        ),
        (["agency", "agencia"], "Agencia"),
        (
            [
                "delivery agency",
                "name of the delivery agency",
                "agencia de entrega",
                "nombre de la agencia de entrega",
            ],
            "Nombre_de_Agencia_de_Entrega",
        ),
        (
            ["acquisition method", "metodo de adquisicion", "método de adquisición"],
            "Metodo_de_Adquisicion",
        ),
        (
            [
                "purchase order number",
                "number of the purchase order",
                "numero de orden de compra",
                "número de orden de compra",
            ],
            "Numero_de_Orden_de_Compra",
        ),
        (
            ["order file name", "nombre de archivo de orden de compra"],
            "Nombre_de_Archivo_de_Orden_de_Compra",
        ),
        (
            ["order file url", "url de archivo de orden de compra"],
            "Url_de_Archivo_de_Orden_de_Compra",
        ),
        (["brand", "marca", "marca del artículo"], "Marca_de_Articulo"),
        (["model", "modelo", "model del artículo"], "Modelo_de_Articulo"),
        (["unit", "unit measurement", "unidad de medida"], "Unidad_de_Medida"),
        (["quantity", "cantidad"], "Cantidad"),
        (
            [
                "received date",
                "fecha recibo",
                "fecha de recibo",
                "fecha de recibo de requisicion",
                "fecha de recibo de requisición",
            ],
            "Fecha_Recibo_de_Requisicion",
        ),
    ]

    SHOW_ALL_TRIGGERS = [
        "all information",
        "all info",
        "full",
        "everything",
        "todos",
        "toda",
    ]

    results: List[str] = []
    error: Optional[str] = None
    query = input.strip()
    q = query.lower()

    try:
        # Prepare headers and body
        token = await _get_azure_search_token()
        api_key = os.getenv("AZURE_SEARCH_API_KEY")
        headers = {
            "Content-Type": "application/json",
            **(
                {"api-key": api_key}
                if api_key
                else {"Authorization": f"Bearer {token}"}
            ),
        }

        body: Dict[str, Any] = {
            "search": query,
            "select": ",".join(all_fields),
            "top": search_top_k,
        }
        if use_semantic and semantic_config:
            body.update(
                {"queryType": "semantic", "semanticConfiguration": semantic_config}
            )

        case_match = re.search(
            r"(?:caso\s+(?:n[uú]mero\s+de\s+)?)?(?:caso)?\s*[#:]*\s*([0-9]{2}[A-Z]-[0-9]+)",
            q,
            re.IGNORECASE,
        )

        if case_match:
            case_number_for_filter = case_match.group(1).upper()
            body["search"] = "*"  # ignore full text search
            body["filter"] = f"Numero_de_Caso eq '{case_number_for_filter}'"
            body["top"] = 1
            print(f"📌 Filtering by Numero_de_Caso: {case_number_for_filter}")

        url = f"https://{service}.search.windows.net/indexes/{index}/docs/search?api-version={api_version}"
        resp = await _perform_search(url, headers, body)

        for doc in resp.get("value", []):
            # First, try to match using field_map only
            matched_fields = [
                field for kws, field in field_map if any(kw in q for kw in kws)
            ]

            # If nothing matched, fall back to dynamic_field_map
            if not matched_fields:
                matched_fields = [
                    field
                    for kws, field in dynamic_field_map
                    if any(kw in q for kw in kws)
                ]

            if matched_fields:
                lines = []
                for field in matched_fields:
                    val = doc.get(field)
                    if val:
                        label = field_labels.get(field, field)
                        lines.append(f"{label}: {val}")
                if lines:
                    results.append("\n".join(lines))
                else:
                    results.append("No matching field values found.")
                continue

            if any(
                kw in q
                for kw in SHOW_ALL_TRIGGERS
                + ["toda la información", "información del caso"]
            ):
                lines = []
                pdf_url = doc.get("Url_de_Archivo_de_Orden_de_Compra")
                pdf_name = (
                    doc.get("Nombre_de_Archivo_de_Orden_de_Compra")
                    or "Archivo de la Orden de Compra"
                )

                for f in all_fields:
                    if f in [
                        "Url_de_Archivo_de_Orden_de_Compra",
                        "Nombre_de_Archivo_de_Orden_de_Compra",
                    ]:
                        continue
                    val = doc.get(f)
                    if val:
                        lines.append(f"{field_labels[f]}: {val}")

                text_block = "\n".join(lines)

                if pdf_url:
                    from urllib.parse import urlparse, urlencode, parse_qsl

                    parsed = urlparse(pdf_url)
                    query = dict(parse_qsl(parsed.query))
                    query.update({"rsct": "application/pdf", "rscd": "inline"})
                    new_url = parsed._replace(query=urlencode(query)).geturl()

                    text_block += f"\n\n[INLINE_PDF:{pdf_name}|{new_url}]"
                else:
                    text_block += "\n\nArchivo de la Orden de Compra: No disponible."

                results.append(text_block)
            else:
                results.append("❗ Especifique claramente qué campo desea consultar.")

        # Fallback: show inline PDF if no matched fields but PDF exists
        if not results and resp.get("value"):
            doc = resp["value"][0]
            pdf_url = doc.get("Url_de_Archivo_de_Orden_de_Compra")
            pdf_name = (
                doc.get("Nombre_de_Archivo_de_Orden_de_Compra")
                or "Archivo de la Orden de Compra"
            )
            if pdf_url:
                from urllib.parse import urlparse, urlencode, parse_qsl

                parsed = urlparse(pdf_url)
                query = dict(parse_qsl(parsed.query))
                query.update({"rsct": "application/pdf", "rscd": "inline"})
                new_url = parsed._replace(query=urlencode(query)).geturl()
                pdf_filename = pdf_url.split("/")[-1].split("?")[0]
                results.append(
                    f"El archivo del caso {doc.get('Numero_de_Caso')} está disponible:\n\n[INLINE_PDF:{pdf_filename}|{new_url}]"
                )
            else:
                results.append("No se encontró un archivo asociado al caso.")

    except Exception as e:
        error = str(e)
        logging.error("vector_index_retrieve error", exc_info=True)

    return VectorIndexRetrievalResult(result="\n\n".join(results), error=error)


def extract_captions(str_captions):
    # Regular expression pattern to match image references followed by their descriptions
    pattern = r"\[.*?\]:\s(.*?)(?=\[.*?\]:|$)"

    # Find all matches
    matches = re.findall(pattern, str_captions, re.DOTALL)

    return [match.strip() for match in matches]


def replace_image_filenames_with_urls(content: str, related_images: list) -> str:
    """
    Replace image filenames or relative paths in the content string with their corresponding full URLs
    from the related_images list.
    """
    for image_url in related_images:
        # Parse the URL and remove the leading slash from the path
        logging.debug(f"[multimodal_vector_index_retrieve] image_url: {image_url}.")
        parsed_url = urlparse(image_url)
        image_path = parsed_url.path.lstrip(
            "/"
        )  # e.g., 'documents-images/myfolder/filename.png'
        logging.debug(f"[multimodal_vector_index_retrieve] image_path: {image_path}.")
        # Replace occurrences of the relative path in the content with the full URL
        content = content.replace(image_path, image_url)
        logging.debug(f"[multimodal_vector_index_retrieve] content: {content}.")
        # Also replace only the filename if it appears alone
        # filename = image_path.split('/')[-1]
        # content = content.replace(filename, image_url)

    return content


async def multimodal_vector_index_retrieve(
    input: Annotated[
        str,
        "An optimized query string based on the user's ask and conversation history, when available",
    ],
    security_ids: str = "anonymous",
) -> Annotated[
    MultimodalVectorIndexRetrievalResult,
    "A Pydantic model containing the search results with separate lists for texts and images",
]:
    """
    Variation of vector_index_retrieve that fetches text and related images from the search index.
    Returns the results wrapped in a Pydantic model with separate lists for texts and images.
    """
    aoai = AzureOpenAIClient()
    search_top_k = int(os.getenv("AZURE_SEARCH_TOP_K", 3))
    search_approach = os.getenv("AZURE_SEARCH_APPROACH", "hybrid")  # or 'hybrid'
    semantic_search_config = os.getenv(
        "AZURE_SEARCH_SEMANTIC_SEARCH_CONFIG", "my-semantic-config"
    )
    search_service = os.getenv("AZURE_SEARCH_SERVICE", "search0jdjja")
    search_index = os.getenv("AZURE_SEARCH_INDEX", "purchase-orders-from-blob")
    search_api_version = "2023-07-01-Preview"
    use_semantic = os.getenv("AZURE_SEARCH_USE_SEMANTIC", "false").lower() == "true"

    logging.info(f"[multimodal_vector_index_retrieve] User input: {input}")

    text_results: List[str] = []
    image_urls: List[List[str]] = []
    captions: List[str] = []
    error_message: Optional[str] = None

    # 1. Generate embeddings for the query.
    try:
        start_time = time.time()
        embeddings_query = await asyncio.to_thread(aoai.get_embeddings, input)
        embedding_time = round(time.time() - start_time, 2)
        logging.info(
            f"[multimodal_vector_index_retrieve] Query embeddings took {embedding_time} seconds"
        )
    except Exception as e:
        error_message = f"Error generating embeddings: {e}"
        logging.error(
            f"[multimodal_vector_index_retrieve] {error_message}", exc_info=True
        )
        return MultimodalVectorIndexRetrievalResult(
            texts=[], images=[], error=error_message
        )

    # 2. Acquire Azure Search token.
    try:
        azure_search_token = await _get_azure_search_token()
    except Exception as e:
        error_message = f"Error acquiring token for Azure Search: {e}"
        logging.error(
            f"[multimodal_vector_index_retrieve] {error_message}", exc_info=True
        )
        return MultimodalVectorIndexRetrievalResult(
            texts=[], images=[], error=error_message
        )

    # 3. Build the request body.
    body: Dict[str, Any] = {
        "select": "title, content, filepath, url, imageCaptions, relatedImages",
        "top": search_top_k,
        "vectorQueries": [
            {
                "kind": "vector",
                "vector": embeddings_query,
                "fields": "contentVector",
                "k": int(search_top_k),
            },
            {
                "kind": "vector",
                "vector": embeddings_query,
                "fields": "captionVector",
                "k": int(search_top_k),
            },
        ],
    }

    if use_semantic and search_approach != "vector":
        body["queryType"] = "semantic"
        body["semanticConfiguration"] = semantic_search_config

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {azure_search_token}",
    }

    search_url = (
        f"https://{search_service}.search.windows.net"
        f"/indexes/{search_index}/docs/search"
        f"?api-version={search_api_version}"
    )

    # 4. Query Azure Search.
    try:
        start_time = time.time()
        print("🔎 Final search_url2:", search_url)
        print("🔎 Request headers2:", headers)
        print("🔎 Request body2:", json.dumps(body, indent=2))
        response_json = await _perform_search(search_url, headers, body)
        response_time = round(time.time() - start_time, 2)
        logging.info(
            f"[multimodal_vector_index_retrieve] Finished querying Azure AI Search in {response_time} seconds"
        )

        for doc in response_json.get("value", []):

            content = doc.get("content", "")
            str_captions = doc.get("imageCaptions", "")
            captions.append(extract_captions(str_captions))
            url = doc.get("url", "")

            # Convert blob URL to relative path
            uri = re.sub(r"https://[^/]+\.blob\.core\.windows\.net", "", url)
            text_results.append(f"{uri}: {content.strip()}")

            # Replace image filenames with URLs
            content = replace_image_filenames_with_urls(
                content, doc.get("relatedImages", [])
            )

            # Extract image URLs from <figure> tags
            # doc_image_urls = re.findall(r'<figure>(https?://.*?)</figure>', content)
            # image_urls.append(doc_image_urls)
            image_urls.append(doc.get("relatedImages", []))

    except Exception as e:
        error_message = f"Exception in retrieval: {e}"
        logging.error(
            f"[multimodal_vector_index_retrieve] {error_message}", exc_info=True
        )

    return MultimodalVectorIndexRetrievalResult(
        texts=text_results, images=image_urls, captions=captions, error=error_message
    )


def get_data_points_from_chat_log(chat_log: list) -> DataPointsResult:
    """
    Parses a chat log to extract data points (e.g., filenames with extension) from tool call events.
    Returns a Pydantic model containing the list of extracted data points.
    """
    # Regex patterns.
    request_call_id_pattern = re.compile(r"id='([^']+)'")
    request_function_name_pattern = re.compile(r"name='([^']+)'")
    exec_call_id_pattern = re.compile(r"call_id='([^']+)'")
    exec_content_pattern = re.compile(r"content='(.+?)', call_id=", re.DOTALL)

    # Allowed file extensions.
    allowed_extensions = [
        "vtt",
        "xlsx",
        "xls",
        "pdf",
        "docx",
        "pptx",
        "png",
        "jpeg",
        "jpg",
        "bmp",
        "tiff",
    ]
    filename_pattern = re.compile(
        rf"([^\s:]+\.(?:{'|'.join(allowed_extensions)})\s*:\s*.*?)(?=[^\s:]+\.(?:{'|'.join(allowed_extensions)})\s*:|$)",
        re.IGNORECASE | re.DOTALL,
    )

    relevant_call_ids = set()
    data_points = []

    for msg in chat_log:
        if msg["message_type"] == "ToolCallRequestEvent":
            content = msg["content"][0]
            call_id_match = request_call_id_pattern.search(content)
            function_name_match = request_function_name_pattern.search(content)
            if call_id_match and function_name_match:
                if function_name_match.group(1) == "vector_index_retrieve_wrapper":
                    relevant_call_ids.add(call_id_match.group(1))
        elif msg["message_type"] == "ToolCallExecutionEvent":
            content = msg["content"][0]
            call_id_match = exec_call_id_pattern.search(content)
            if call_id_match and call_id_match.group(1) in relevant_call_ids:
                content_part_match = exec_content_pattern.search(content)
                if not content_part_match:
                    continue
                content_part = content_part_match.group(1)
                try:
                    parsed = json.loads(content_part)
                    texts = parsed.get("texts", [])
                except json.JSONDecodeError:
                    texts = [
                        re.split(
                            r'["\']images["\']\s*:\s*\[', content_part, 1, re.IGNORECASE
                        )[0]
                    ]
                for text in texts:
                    text = bytes(text, "utf-8").decode("unicode_escape")
                    for match in filename_pattern.findall(text):
                        extracted = match.strip(' ,\\"').lstrip("[").rstrip("],")
                        if extracted:
                            data_points.append(extracted)
    return DataPointsResult(data_points=data_points)
