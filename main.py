from fastapi import FastAPI,UploadFile, File,HTTPException,Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import faiss
from tempfile import NamedTemporaryFile
from typing import List,Optional,Dict
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import uuid
from utils.document_processing import parse_document
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
from flask_ngrok import run_with_ngrok


app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
run_with_ngrok(app)

embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
# Base directory to store FAISS indices
BASE_DIR = "./faiss_indices"
os.makedirs(BASE_DIR, exist_ok=True)
huggingface_token = "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"
llama_model_name = "NousResearch/Llama-2-7b-hf"
llama_tokenizer = AutoTokenizer.from_pretrained(llama_model_name, token=huggingface_token)
llama_model = AutoModelForCausalLM.from_pretrained(llama_model_name,token=huggingface_token)
conversation_contexts: Dict[str, List[Dict[str, str]]] = {}


@app.post("/upload")
async def upload_files(user_id: str = Form(...), files: List[UploadFile] = File(...)):
    faiss_index_path = os.path.join(BASE_DIR, f"{user_id}_index.faiss")
    chunks_path = os.path.join(BASE_DIR, f"chunks_{user_id}.json")

    dimension = embedding_model.get_sentence_embedding_dimension()
    if os.path.exists(faiss_index_path):
        faiss_index = faiss.read_index(faiss_index_path)
        with open(chunks_path, "r") as f:
            document_chunks = json.load(f)
    else:
        faiss_index = faiss.IndexFlatL2(dimension)
        document_chunks = []

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

    total_files = len(files)
    processed_percentage = (len(processed_files) / total_files) * 100

    return {
        "status": "success",
        "processed_files": processed_files,
        "failed_files": failed_files,
        "processed_percentage": processed_percentage,
        "message": f"Documents uploaded and indexed for user {user_id}",
        "index_path": faiss_index_path
    }

@app.post("/agents/{agent_id}/conversations")
async def start_conversation(agent_id: str, query: str = Form(...), conversation_id: Optional[str] = None):
    faiss_index_path = os.path.join(BASE_DIR, f"{agent_id}_index.faiss")
    chunks_path = os.path.join(BASE_DIR, f"chunks_{agent_id}.json")

    if not os.path.exists(faiss_index_path) or not os.path.exists(chunks_path):
        raise HTTPException(status_code=404, detail="FAISS index or document chunks not found for agent")

    faiss_index = faiss.read_index(faiss_index_path)
    with open(chunks_path, "r") as f:
        document_chunks = json.load(f)

    if not conversation_id:
        conversation_id = str(uuid.uuid4())
        conversation_contexts[conversation_id] = []

    context_history = conversation_contexts.get(conversation_id, [])
    query_embedding = embedding_model.encode([query])
    k = 5
    distances, indices = faiss_index.search(query_embedding, k)

    retrieved_context = [document_chunks[index] for index in indices[0] if index != -1]

    if not retrieved_context:
        response_text = "No relevant information found in the uploaded documents."
    else:
        context_for_llm = "\n".join(retrieved_context)
        input_text = f"Context: {context_for_llm}\n\nQuery: {query}\n\nAnswer:"
        inputs = llama_tokenizer.encode(input_text, return_tensors="pt")
        outputs = llama_model.generate(inputs, max_length=500, num_return_sequences=1, do_sample=True)
        response_text = llama_tokenizer.decode(outputs[0], skip_special_tokens=True)

    context_history.append({"query": query, "response": response_text})
    conversation_contexts[conversation_id] = context_history

    summary = response_text[:100]

    return {
        "conversation_id": conversation_id,
        "summary": summary,
        "response": response_text,
        "references": retrieved_context,
        "context_history": context_history
    }

app.run()