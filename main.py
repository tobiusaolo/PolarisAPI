from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from firebase_admin import firestore, credentials
from pyngrok import ngrok
import os
import faiss
from typing import List, Optional, Dict
from sentence_transformers import SentenceTransformer
from utils.open_model import process_files_and_create_json
import uuid
from utils.document_processing import parse_document
import json
import firebase_admin
from google.cloud import storage
from google.oauth2 import service_account
from utils.conversations_management import create_new_conversation
from utils.computer_node import get_node_id
import signal
import sys
from pyngrok import conf


service_account_info = json.load(open("./chatbot.json"))
cred = credentials.Certificate(service_account_info)
firebase_admin.initialize_app(cred)
db = firestore.client()
gcs_credentials = service_account.Credentials.from_service_account_info(service_account_info)
storage_client = storage.Client(credentials=gcs_credentials)
bucket_name = 'tericv-8eed9.appspot.com'
bucket = storage_client.bucket(bucket_name)


app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Check if a GPU is available
USE_GPU = faiss.get_num_gpus() > 0

embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

BASE_DIR = "./faiss_indices"
os.makedirs(BASE_DIR, exist_ok=True)
def save_or_update_node_status(node_id: str, ngrok_url: str, status: str):
    """
    Save or update the node status in Firebase.
    """
    doc_ref = db.collection("compute_nodes").document(node_id)
    doc = doc_ref.get()

    if doc.exists:
        # Update the existing document
        doc_ref.update({
            "ngrok_url": ngrok_url,
            "status": status,
            "updated_at": firestore.SERVER_TIMESTAMP
        })
        print(f"Updated node {node_id} in Firebase.")
    else:
        # Create a new document
        doc_ref.set({
            "node_id": node_id,
            "ngrok_url": ngrok_url,
            "status": status,
            "created_at": firestore.SERVER_TIMESTAMP,
            "updated_at": firestore.SERVER_TIMESTAMP
        })
        print(f"Created new node {node_id} in Firebase.")


def set_node_offline(node_id: str):
    """
    Set the node status to offline in Firebase.
    """
    doc_ref = db.collection("compute_nodes").document(node_id)
    if doc_ref.get().exists:
        doc_ref.update({
            "status": "offline",
            "updated_at": firestore.SERVER_TIMESTAMP
        })
        print(f"Set node {node_id} status to offline in Firebase.")

conversation_contexts: Dict[str, List[Dict[str, str]]] = {}
def create_firestore_conversation(user_id: str, conversation_id: str):
    doc_ref = db.collection("Polaris").document(user_id).collection("conversations").document(conversation_id)
    doc_ref.set({
        "created_at": str(firestore.SERVER_TIMESTAMP),
        "updated_at": str(firestore.SERVER_TIMESTAMP),
        "messages": []
    })

def get_firestore_conversation_history(user_id: str, conversation_id: str):
    doc_ref = db.collection("Polaris").document(user_id).collection("conversations").document(conversation_id)
    doc = doc_ref.get()
    if doc.exists:
        return doc.to_dict()["messages"]
    return []

def update_firestore_conversation(user_id: str, conversation_id: str, message: dict):
    doc_ref = db.collection("Polaris").document(user_id).collection("conversations").document(conversation_id)
    doc_ref.update({
        "messages": firestore.ArrayUnion([message]),
        "updated_at": str(firestore.SERVER_TIMESTAMP)
    })

@app.post("/upload")
async def upload_files(user_id: str = Form(...), files: List[UploadFile] = File(...), title: str = Form(...), description: str = Form(...)):
    agent_id = str(uuid.uuid4())
    faiss_index_path = os.path.join(BASE_DIR, f"{user_id}_index.faiss")
    chunks_path = os.path.join(BASE_DIR, f"chunks_{user_id}.json")

    dimension = embedding_model.get_sentence_embedding_dimension()
    if os.path.exists(faiss_index_path):
        faiss_index = faiss.read_index(faiss_index_path)
    else:
        faiss_index = faiss.IndexFlatL2(dimension)

    document_chunks = []
    if os.path.exists(chunks_path):
        with open(chunks_path, "r") as f:
            document_chunks = json.load(f)

    processed_files = []
    failed_files = []

    for file in files:
        content = await file.read()
        try:
            text_chunks = parse_document(content, file.filename)
            embeddings = embedding_model.encode(text_chunks)
            faiss_index.add(embeddings)
            document_chunks.extend(text_chunks)
            processed_files.append(file.filename)
        except Exception as e:
            failed_files.append(file.filename)

    faiss.write_index(faiss_index, faiss_index_path)
    with open(chunks_path, "w") as f:
        json.dump(document_chunks, f)
    
    blob = bucket.blob(f"faiss_indices/{agent_id}_index.faiss")
    blob.upload_from_filename(faiss_index_path)
    index_url = blob.public_url

    db.collection("agent_store").document(agent_id).set({
        "user_id": user_id,
        "faiss_index_url": index_url,
        "chunks_path": chunks_path,
        "title": title,
        "description": description,
        "created_at": str(firestore.SERVER_TIMESTAMP)
    })
    total_files = len(files)
    processed_percentage = (len(processed_files) / total_files) * 100

    return {
        "status": "success",
        "processed_files": processed_files,
        "failed_files": failed_files,
        "processed_percentage": processed_percentage,
        "message": f"Documents uploaded and indexed for user {user_id}",
        "index_url": index_url
    }

@app.post("/agents/update_index")
async def update_agent_index(agent_id: str =Form(...), files: List[UploadFile] = File(...)):
    agent_data = db.collection("agent_store").document(agent_id).get()
    if not agent_data.exists:
        raise HTTPException(status_code=404, detail="Agent data not found")

    agent_info = agent_data.to_dict()
    user_id = agent_info["user_id"]
    chunks_path = agent_info["chunks_path"]
    
    faiss_index_path = os.path.join(BASE_DIR, f"{agent_id}_index.faiss")
    blob = bucket.blob(f"faiss_indices/{agent_id}_index.faiss")

    try:
        blob.download_to_filename(faiss_index_path)
    except Exception as e:
        raise HTTPException(
            status_code=404,
            detail=f"Failed to download FAISS index file from storage for agent {agent_id}: {str(e)}",
        )


    try:
        faiss_index = faiss.read_index(faiss_index_path)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error loading FAISS index for agent {agent_id}: {str(e)}",
        )

    # Load existing document chunks
    document_chunks = []
    if os.path.exists(chunks_path):
        with open(chunks_path, "r") as f:
            document_chunks = json.load(f)

    processed_files = []
    failed_files = []

    # Process uploaded files and update the FAISS index
    for file in files:
        content = await file.read()
        try:
            text_chunks = parse_document(content, file.filename)
            embeddings = embedding_model.encode(text_chunks)
            faiss_index.add(embeddings)
            document_chunks.extend(text_chunks)
            processed_files.append(file.filename)
        except Exception as e:
            failed_files.append(file.filename)

    # Save updated FAISS index locally
    faiss.write_index(faiss_index, faiss_index_path)

    # Save updated document chunks
    with open(chunks_path, "w") as f:
        json.dump(document_chunks, f)

    # Upload the updated FAISS index back to GCS
    try:
        blob.upload_from_filename(faiss_index_path)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to upload updated FAISS index to storage for agent {agent_id}: {str(e)}",
        )

    return {
        "status": "success",
        "processed_files": processed_files,
        "failed_files": failed_files,
        "message": f"Agent's FAISS index updated for {agent_id}",
    }


@app.post("/agents")
async def list_agents():
    agents_ref = db.collection("agent_store")
    agents = agents_ref.stream()
    agents_data = []
    for agent in agents:
        agent_info = agent.to_dict()
        agents_data.append({
            "agent_id": agent.id,
            "user_id": agent_info["user_id"],
            "faiss_index_url": agent_info["faiss_index_url"],
            "chunks_path": agent_info["chunks_path"],
            "title": agent_info.get("title", "N/A"),
            "description": agent_info.get("description", "N/A"),
        })
    return {"agents": agents_data}
@app.post("/agents/conversations")
async def start_conversation(agent_id: str = Form(...), query: str = Form(...),user_id: str= Form(...),conversation_id: Optional[str] = Form(None)):
   
    agent_data = db.collection("agent_store").document(agent_id).get()
    if not agent_data.exists:
        raise HTTPException(status_code=404, detail="Agent data not found in Firebase")
    agent_info = agent_data.to_dict()
    index_url = agent_info["faiss_index_url"]
    chunks_path = agent_info["chunks_path"]
    # Download the FAISS index from the URL
    local_faiss_path = os.path.join(BASE_DIR, f"{agent_id}_index.faiss")
    blob = bucket.blob(f"faiss_indices/{agent_id}_index.faiss")
    blob.download_to_filename(local_faiss_path)

    # Load FAISS index and document chunks
    faiss_index = faiss.read_index(local_faiss_path)
    with open(chunks_path, "r") as f:
        document_chunks = json.load(f)

    if conversation_id:
        # Check if the conversation_id exists
        conversation_ref = db.collection("Polaris").document(user_id).collection("conversations").document(conversation_id)
        if not conversation_ref.get().exists:
            raise HTTPException(status_code=404, detail=f"Conversation ID {conversation_id} not found.")
    else:
        # Create a new conversation if no ID is provided
        conversation_id = create_new_conversation(user_id)
        create_firestore_conversation(user_id, conversation_id)
        

    conversation_history = get_firestore_conversation_history(user_id, conversation_id)


    # Encode the query and retrieve relevant context
    query_embedding = embedding_model.encode([query])
    k = 2  # Retrieve fewer chunks for more focused context
    distances, indices = faiss_index.search(query_embedding, k)
    retrieved_context = [document_chunks[index] for index in indices[0] if index != -1]

    if not retrieved_context:
        return {
            "conversation_id": conversation_id,
            "answer": "No relevant information found in the uploaded documents."
        }

    # Prepare context for the pipeline
    context_for_llm = "\n".join([chunk.strip() for chunk in retrieved_context[:k]])
    input_text = f"{conversation_history}\nContext:\n{context_for_llm}\n\nQuery: {query}\n\nAnswer:"

  
    # Generate the response
    answer =process_files_and_create_json(input_text)
    user_message = {
            "role": "user",
            "content": query,
            "timestamp": str(firestore.SERVER_TIMESTAMP)
        }
    assistant_message = {
            "role": "assistant",
            "content": answer[0]["message"],
            "timestamp": str(firestore.SERVER_TIMESTAMP)}

    # Update the conversation context
    update_firestore_conversation(user_id, conversation_id, user_message)
    update_firestore_conversation(user_id, conversation_id, assistant_message)
    
    return {
        "conversation_id": conversation_id,
        "answer": answer[0]["message"]
    }


@app.post("/users/conversations")
async def get_user_conversations(user_id: str=Form(...)):
    conversations_ref = db.collection("Polaris").document(user_id).collection("conversations")
    conversations = conversations_ref.stream()
    conversations_data = []
    for conversation in conversations:
        conversation_info = conversation.to_dict()
        agent_id = conversation_info.get("agent_id", "N/A")
        conversations_data.append({
            "conversation_id": conversation.id,
            "created_at": conversation_info["created_at"],
            "updated_at": conversation_info["updated_at"],
            "messages": conversation_info["messages"],
        })
    return {"conversations": conversations_data}

def run_with_ngrok():
    # Configure Ngrok token
    NGROK_TOKEN = "2qX6f75l3fYfD7Xn9Q80KiFV1i7_yTTB7Q2pZxfgKRvS1TbJ"
    conf.get_default().auth_token = NGROK_TOKEN
    # Create Ngrok tunnel
    public_url = ngrok.connect(8000).public_url
    print(f"NGROK public URL: {public_url}")

    # Get or generate the node ID
    node_id = get_node_id()

    # Save or update node status in Firebase
    save_or_update_node_status(node_id, public_url, "online")

    # Handle graceful shutdown
    def shutdown_handler(signal, frame):
        print("Shutting down...")
        set_node_offline(node_id)
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown_handler)
    signal.signal(signal.SIGTERM, shutdown_handler)

    # Start FastAPI app
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    run_with_ngrok()
