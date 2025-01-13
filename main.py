from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from pyngrok import ngrok
import os
import faiss
from typing import List, Optional, Dict
from sentence_transformers import SentenceTransformer
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
import uuid
from utils.document_processing import parse_document
import json
import torch


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

huggingface_token = "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"
llama_model_name = "meta-llama/Llama-3.2-1B"
llama_tokenizer = AutoTokenizer.from_pretrained(llama_model_name, use_auth_token=huggingface_token)
llama_model = AutoModelForCausalLM.from_pretrained(llama_model_name, use_auth_token=huggingface_token)
conversation_contexts: Dict[str, List[Dict[str, str]]] = {}


@app.post("/upload")
async def upload_files(user_id: str = Form(...), files: List[UploadFile] = File(...)):
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

    # Load FAISS index and document chunks
    faiss_index = faiss.read_index(faiss_index_path)
    with open(chunks_path, "r") as f:
        document_chunks = json.load(f)

    if not conversation_id:
        conversation_id = str(uuid.uuid4())
        conversation_contexts[conversation_id] = []

    context_history = conversation_contexts.get(conversation_id, [])
    past_conversation = "\n".join([f"Q: {c['query']} A: {c['response']}" for c in context_history]) if context_history else ""

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
    input_text = f"{past_conversation}\nContext:\n{context_for_llm}\n\nQuery: {query}\n\nAnswer:"

    # Initialize the text-generation pipeline
    model_id = "meta-llama/Meta-Llama-3-8B"
    pipeline = transformers.pipeline("text-generation", model=model_id, model_kwargs={"torch_dtype": torch.bfloat16}, device_map="auto")

    # Generate the response
    response = pipeline(input_text, max_length=400, num_return_sequences=1, do_sample=True)
    answer = response[0]['generated_text'].split("Answer:")[-1].strip()

    # Update the conversation context
    context_history.append({"query": query, "response": answer})
    conversation_contexts[conversation_id] = context_history

    return {
        "conversation_id": conversation_id,
        "answer": answer
    }



def run_with_ngrok():
    public_url = ngrok.connect(8000).public_url
    print(f"NGROK public URL: {public_url}")
    print("Access your FastAPI app at this URL.")

    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    run_with_ngrok()
