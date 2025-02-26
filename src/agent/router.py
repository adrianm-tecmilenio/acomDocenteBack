import os
import io
import pdfplumber
import uuid
from datetime import datetime

from fastapi import APIRouter, UploadFile, File, HTTPException
from dotenv import load_dotenv
from pydantic import BaseModel
from pydantic_ai.messages import ModelRequest, ModelResponse, UserPromptPart, SystemPromptPart, TextPart

from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from openai import AsyncAzureOpenAI
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langgraph.graph import StateGraph
from pinecone import Pinecone
import snowflake.connector

from .prompts import PYDANTIC_SYSTEM_PROMPT



# Cargar credenciales desde .env
load_dotenv()

router = APIRouter()

# Snowflake connection configuration
SNOWFLAKE_CONFIG = {
    "user": os.getenv("SNOWFLAKE_USER"),
    "password": os.getenv("SNOWFLAKE_PASSWORD"),
    "account": os.getenv("SNOWFLAKE_ACCOUNT"),
    "warehouse": os.getenv("SNOWFLAKE_WAREHOUSE"),
    "database": os.getenv("SNOWFLAKE_DATABASE"),
    "schema": os.getenv("SNOWFLAKE_SCHEMA"),
}

class Item(BaseModel):
    message: str
    session_id: str 

class ChatState(BaseModel):
    query: str  
    context: str  
    response: str  
    session_id: str

def get_snowflake_connection():
    """Get a connection to Snowflake."""
    try:
        conn = snowflake.connector.connect(
            user=os.getenv("SNOWFLAKE_USER"),
            password=os.getenv("SNOWFLAKE_PASSWORD"),
            account=os.getenv("SNOWFLAKE_ACCOUNT"),
            warehouse=os.getenv("SNOWFLAKE_WAREHOUSE"),
            database=os.getenv("SNOWFLAKE_DATABASE"),
            schema=os.getenv("SNOWFLAKE_SCHEMA")
        )
        return conn
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error connecting to Snowflake: {e}")

def save_message_to_snowflake(session_id: str, role: str, message: str):
    """Save a message to Snowflake."""
    message_id = str(uuid.uuid4())
    timestamp = datetime.utcnow()

    query = """
    INSERT INTO ACOMP_DOCENTE (session_id, message_id, role, message, timestamp)
    VALUES (%s, %s, %s, %s, %s)
    """

    # Initialize variables
    conn = None
    cursor = None

    try:
        conn = get_snowflake_connection()

        cursor = conn.cursor()
        cursor.execute(query, (session_id, message_id, role, message, timestamp))
        conn.commit()
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail=f"Error saving message to Snowflake: {e}")
    finally:
        # Close the cursor and connection if they were successfully created
        if cursor is not None:
            cursor.close()
        if conn is not None:
            conn.close()

def get_chat_history_from_snowflake(session_id: str, limit: int = 10):
    """Retrieve chat history from Snowflake."""
    query = """
    SELECT role, message
    FROM ACOMP_DOCENTE
    WHERE session_id = %s
    ORDER BY timestamp DESC
    LIMIT %s
    """

    # Initialize variables
    conn = None
    cursor = None

    try:
        conn = get_snowflake_connection()
        cursor = conn.cursor()
        cursor.execute(query, (session_id, limit))
        rows = cursor.fetchall()
        return [{"role": row[0], "message": row[1]} for row in rows]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving chat history: {e}")
    finally:
        # Close the cursor and connection if they were successfully created
        if cursor is not None:
            cursor.close()
        if conn is not None:
            conn.close()

# @router.post('/agent')
# async def agent(body:Item):
#     try:
#         llm = AzureChatOpenAI(
#             deployment_name="gpt4oIA-AM",
#             model_name="gpt-4o",
#             azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
#         )

#         messages = [
#             SystemMessage(content="""
#                           You are an expert chef, 
#                           and you are teaching a cooking class.
#                           Give the user a cooking instruction.""")
#         ]

#         messages.append(
#             HumanMessage(
#                 content=body.message
#             )
#         )

#         res = llm.invoke(messages)
   
#         return {'message': res.content}
#     except Exception as e:
#         raise e
#         # return {'message': str(e)}

client = AsyncAzureOpenAI(
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_version='2024-05-01-preview',
            api_key=os.getenv("OPENAI_API_KEY"),
        )

embeddings = AzureOpenAIEmbeddings(
    dimensions = 3072, #Cambiar si cambiamos dimensiones para vector_store
    model=os.getenv("AZURE_EMBEDDINGS_DEPLOYMENT"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("OPENAI_API_KEY"),
    openai_api_version=os.getenv("OPENAI_API_VERSION")
)

# Initialize Pinecone using the new Pinecone class
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

index_name = "acom-docente"
if index_name not in pc.list_indexes().names():
    pc.create_index(name=index_name, dimension=3072, metric="cosine")
index = pc.Index(index_name) 

def upsert_in_batches(index, vectors, batch_size=100):
    """Upsert vectors in batches to avoid exceeding Pinecone's size limit."""
    for i in range(0, len(vectors), batch_size):
        batch = vectors[i:i + batch_size]
        index.upsert(vectors=batch)
        print(f"Upserted batch {i // batch_size + 1}")

async def load_pdfs_and_store_embeddings(pdf_files):
    """Carga PDFs, extrae texto y almacena embeddings en FAISS."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=300)

    for file in pdf_files:
        # Read the PDF file in memory
        pdf_bytes = file.file.read()

        # Wrap the bytes in a file-like object
        pdf_stream = io.BytesIO(pdf_bytes)

        # Use pdfplumber to extract text and tables
        with pdfplumber.open(pdf_stream) as pdf:
            text = ""
            for page in pdf.pages:
                # Extract text
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"

                # Extract tables (optional)
                tables = page.extract_tables()
                for table in tables:
                    # Convert the table to text (e.g., as a CSV-like string)
                    table_text = "\n".join(["\t".join(map(str, row)) for row in table])
                    text += table_text + "\n"

        if not text.strip():
            print("⚠️ Advertencia: No se extrajo texto del PDF.")
            continue

        try:
            with open("file.txt", "w", encoding="utf-8") as f:
                f.write(text + "\n\n---\n\n")  # Agregar separador entre documentos
            print("✅ Texto escrito en file.txt correctamente")
        except Exception as e:
            print(f"❌ Error al escribir en file.txt: {e}")


        chunks = text_splitter.split_text(text)
        
        # Generate embeddings for each chunk
        embeddings_list = embeddings.embed_documents(chunks)

        vectors = []
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings_list)):
            vector_id = f"{file.filename}_{i}"
            vectors.append((vector_id, embedding, {"text": chunk}))

        # Upsert vectors in batches
        upsert_in_batches(index, vectors, batch_size=100)

        print(f"✅ {len(vectors)} vectors upserted into Pinecone index.")

        # for i in range(len(chunks)):
        #     with open(f"chunk{i+1}.txt", "w", encoding="utf-8") as f:
        #         f.write(chunks[i] + "\n\n---\n\n") 
        #     print(f"✅ Chunk{i+1} en file.txt correctamente")

        # vector_store.add_texts(chunks)

        # print("Number of documents in vector store:", len(vector_store.docstore._dict))



@router.post("/upload-pdf")
async def upload_pdfs(files: list[UploadFile] = File(...)):
    """Sube PDFs y almacena sus embeddings."""
    await load_pdfs_and_store_embeddings(files)
    return {"message": "Documentos cargados correctamente."}

# --------------------------------------
# LangGraph - Flujo del agente
# --------------------------------------

# Definir estado inicial del grafo
class ChatState(BaseModel):
    query: str  # Pregunta del usuario
    context: str  # Contexto del vector store FAISS
    response: str  # Respuesta del agente
    session_id: str #ID de la sesión

# Función de recuperación de contexto
async def retrieve_context(state: ChatState):
    """Recupera contexto relevante usando búsqueda semántica en FAISS."""

    # Generate embedding for the query
    query_embedding = embeddings.embed_query(state.query)

    # Query Pinecone index
    results = index.query(vector=query_embedding, top_k=20, include_metadata=True)

    context = "\n".join([match.metadata["text"] for match in results.matches])

    with open("retrieved.txt", "w") as f:
        f.write(context)

    print("Context retrieved and written to retrieved.txt")

    state.context = context  # Update the context field
    return state

def format_chat_history_for_pydantic(chat_history):
    """Convert chat history from Snowflake into PydanticAI message format."""
    formatted_history = []
    for message in chat_history:
        if message["role"] == "user":
            formatted_history.append(
                ModelRequest(
                    parts=[
                        UserPromptPart(
                            content=message["message"],
                            timestamp=datetime.utcnow(),
                            part_kind="user-prompt",
                        )
                    ],
                    kind="request",
                )
            )
        elif message["role"] == "assistant":
            formatted_history.append(
                ModelResponse(
                    parts=[
                        TextPart(
                            content=message["message"],
                            part_kind="text",
                        )
                    ],
                    model_name="gpt-4o",  # Replace with your model name
                    timestamp=datetime.utcnow(),
                    kind="response",
                )
            )
    return formatted_history

# Función que ejecuta Pydantic AI con el contexto recuperado
async def call_pydantic_ai(state: ChatState):
    """Ejecuta Pydantic AI con el contexto obtenido."""
    system_prompt = f"Instrucciones:\n{PYDANTIC_SYSTEM_PROMPT}"
    prompt = f"Contexto:\n{state.context}\n\nPregunta:\n{state.query}"

    chat_history = get_chat_history_from_snowflake(state.session_id)

    formatted_history = format_chat_history_for_pydantic(chat_history)

    model = OpenAIModel('gpt-4o', openai_client=client)
    agent = Agent(model, system_prompt=system_prompt)

    response = await agent.run(prompt, message_history=formatted_history)


    state.response = response.data 

    # Save the assistant's response to Snowflake
    save_message_to_snowflake(state.session_id, "assistant", state.response)

    return state

# Construir grafo de LangGraph
graph = StateGraph(ChatState)

graph.add_node("retrieve_context", retrieve_context)
graph.add_node("call_pydantic_ai", call_pydantic_ai)

# Conectar los nodos en orden
graph.set_entry_point("retrieve_context")
graph.add_edge("retrieve_context", "call_pydantic_ai")
graph.set_finish_point("call_pydantic_ai")

# Compilar el grafo
executor = graph.compile()

@router.post("/delete-all")
async def delete_all_records():
    """Delete all records from the Pinecone index."""
    try:
        # Delete all vectors in the index
        index.delete(delete_all=True)
        return {"message": "All records deleted successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting records: {e}")

@router.post('/pydantic-agent')
async def pydantic_agent(body: Item):
    try:
        # Save the user's message to Snowflake
        save_message_to_snowflake(body.session_id, "user", body.message)

        initial_state = ChatState(query=body.message, context="", response="", session_id=body.session_id)

        # Stream the state updates
        final_state = None
        async for output in executor.astream(initial_state):
            final_state = output  # Capture the final state

        # Return the final response
        if final_state:
            return {"message": body.message, "response": final_state['call_pydantic_ai'].get('response')}
        else:
            return {"message": "Error" ,"response": "No response generated"}
        
    except Exception as e:
        raise e