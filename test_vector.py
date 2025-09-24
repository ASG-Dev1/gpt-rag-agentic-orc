import asyncio
import os
from dotenv import load_dotenv

from tools.ragindex.vector_index_retrieval import vector_index_retrieve

load_dotenv()  # Load your .env file


async def run_test():
    query = "Quien es el suplidor del numero de caso 24J-15661"

    result = await vector_index_retrieve(query)
    print("📄 Query:", query)
    print("📄 Result:", result.result)
    # result = await vector_index_retrieve(query2)
    # print("📄 Query2:", query2)
    # print("📄 Result2:", result.result)
    if result.error:
        print("❌ Error:", result.error)


if __name__ == "__main__":
    asyncio.run(run_test())


# 23J-12377
# 24J-15661
# 23J-10876
# 24J-03624
# 24J-05104
# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# THESE QUERIES (DO NOT) RESPOND IN TESTS
# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# query = "Quien es el suplidor del numero de caso 24J-15661"
# query = "Cual es el costo unitario del numero de caso 24J-15661"
# query = "Necesito la descripción del artículo en el numero de caso 24J-15661"
# query = "Con que agencia se realizó la requisicion en el numero de caso 24J-15661"
# query = "En que fecha se realizó el recibo de requisición para el caso 24J-15661"

# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# THESE QUERIES RESPOND, BUT BRING ALL INFORMATION OF THE CASE NUMBER IN THE RESULTS, PLUS BRING ABOUT 3 DIFFERENT CASE NUMBERS INFO INCLUDING THE CORRECT CASE NUMBER.
# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# query = "Cual es la marca del artículo en 24J-15661"
# query = "Cual es el modelo en 24J-15661"
# query = "Cuanto tiempo de garantía tiene el caso 24J-15661"
# query = "Cual fue el costo unitario final de 24J-15661"
# query = "Cual fue el costo final del caso 24J-15661"
# query = "Menciona el nombre del suplidor en 24J-15661"
# query = "Cual es el numero de reqisicion en 24J-15661"
# query = "Cual fue el costo estimado del caso 24J-15661"

# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# THESE QUERIES RESPOND, BUT BRING ALL INFORMATION OF ABOUT 3 DIFFERENT CASE NUMBERS (OTHER CASE NUMBERS)IN THE RESULTS, (NOT) INCLUDING THE CORRECT CASE NUMBER.
# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# query = "Cual es la unidad de medida para el número 24J-15661"
# query = "Cual es el número de requisicion para el caso 24J-15661"
# query = "Dime el titulo de requisicion para el caso numero 24J-15661"
# query = "Cual es el costo estimado total de la orden del articulo en el caso de 24J-15661"
# query = "Cual es el Número de Contrato del Número de Caso 24J-15661"
# query = "Cual es la orden de compra en caso numero 24J-15661"
# query = "Dame el nombre del archivo de la orden de compra en el caso 24J-15661"
# query = "Cual es el contacto o telefono del suplidor"
# query = "Cual es el correo electronico del suplidor en el caso 24J-15661"

# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# THESE QUERIES RESPOND, AND BRING ACCURATE INFORMATION OF THE CASE NUMBER IN THE RESULTS, PLUS BRING ABOUT 1-3 DIFFERENT CASE NUMBERS ACURATE INFO INCLUDING THE CORRECT CASE NUMBER.
# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# query = "Que cantidad existe en 24J-15661"
# query = "Cual es la subcategoria del caso 24J-15661"
# query = "Cual fue el metodo de adquisicion 24J-15661"
# query = "Dame toda la información del caso 24J-15661"

# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# THESE QUERIES RESPOND ACCURATE RESULTS IN TESTS ALTHOUGH ALSO MAY BRING ACCURATE RESULTS OF OTHER CASE NUMBERS
# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# query = "Para el caso 24J-15661, en que Agencia se entregó"
# query = "Cual es la categoria en 24J-15661"
# query = "Adjunta el url que posee el caso 24J-15661"

# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Questions like this do not bring the correct information in tests, but if asked in a different way, they do bring the correct information of the case number plus information of other case numbers.
# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# query = "Cual es la información  del caso 24J-15661"
# query = (
#     "Quien es el suplidor del numero de caso 23J-12377 y numero de caso 24J-15661"
# )
