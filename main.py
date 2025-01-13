from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from pyngrok import ngrok
import os
import faiss
import numpy as np
from typing import List, Optional, Dict
from sentence_transformers import SentenceTransformer
import json
import uuid
import torch
from functools import lru_cache
import asyncio
from concurrent.futures import ThreadPoolExecutor
import threading
import uvicorn
from transformers import AutoModelForCausalLM, AutoTokenizer
from datetime import datetime
from pydantic import BaseModel

# Initialize FastAPI app
app = FastAPI(
    title="AI Agent API",
    description="An optimized AI agent using FAISS and LLaMA2",
    version="2.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Constants
EMBEDDING_MODEL_NAME = 'sentence-transformers/all-MiniLM-L6-v2'
BASE_DIR = "./faiss_indices"
HUGGINGFACE_TOKEN = "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"
LLAMA_MODEL_NAME = "NousResearch/Llama-2-7b-hf"
SIMILARITY_THRESHOLD = 0.6
TOP_K_RESULTS = 3
MAX_CONTEXT_WINDOW = 1000
BATCH_SIZE = 32

# Create base directory
os.makedirs(BASE_DIR, exist_ok=True)

# Initialize thread pool
thread_pool = ThreadPoolExecutor(max_workers=4)

# Response Models
class UploadResponse(BaseModel):
    status: str
    processed_files: List[str]
    failed_files: List[Dict[str, str]]
    processed_percentage: float

class ConversationResponse(BaseModel):
    conversation_id: str
    answer: str
    response_time: float
    context_found: bool

class ModelManager:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(ModelManager, cls).__new__(cls)
                    cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        print("Initializing models...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        
        # Load embedding model
        self.embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        self.embedding_model.to(self.device)
        
        # Load LLM with optimizations
        self.llama_tokenizer = AutoTokenizer.from_pretrained(
            LLAMA_MODEL_NAME, 
            token=HUGGINGFACE_TOKEN
        )
        self.llama_model = AutoModelForCausalLM.from_pretrained(
            LLAMA_MODEL_NAME,
            token=HUGGINGFACE_TOKEN,
            device_map="auto",
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True
        )
        
        if self.device == "cuda":
            self.llama_model = torch.compile(self.llama_model)
        print("Models initialized successfully")

    @lru_cache(maxsize=1000)
    def get_embedding(self, text: str) -> np.ndarray:
        return self.embedding_model.encode(text, convert_to_numpy=True)

    def batch_get_embeddings(self, texts: List[str]) -> np.ndarray:
        embeddings = []
        for i in range(0, len(texts), BATCH_SIZE):
            batch = texts[i:i + BATCH_SIZE]
            batch_embeddings = self.embedding_model.encode(
                batch, 
                convert_to_numpy=True
            )
            embeddings.append(batch_embeddings)
        return np.vstack(embeddings)

class OptimizedVectorDB:
    def __init__(self, base_dir: str):
        self.base_dir = base_dir
        self.model_manager = ModelManager()
        self.index_cache = {}
        self.chunk_cache = {}
    
    def get_index(self, user_id: str):
        if user_id not in self.index_cache:
            index_path = os.path.join(self.base_dir, f"{user_id}_index.faiss")
            if os.path.exists(index_path):
                self.index_cache[user_id] = faiss.read_index(index_path)
                if torch.cuda.is_available():
                    res = faiss.StandardGpuResources()
                    self.index_cache[user_id] = faiss.index_cpu_to_gpu(res, 0, self.index_cache[user_id])
        return self.index_cache.get(user_id)
    
    def get_chunks(self, user_id: str):
        if user_id not in self.chunk_cache:
            chunks_path = os.path.join(self.base_dir, f"chunks_{user_id}.json")
            if os.path.exists(chunks_path):
                with open(chunks_path, 'r') as f:
                    self.chunk_cache[user_id] = json.load(f)
        return self.chunk_cache.get(user_id, [])

    async def search(self, user_id: str, query: str) -> List[str]:
        index = self.get_index(user_id)
        chunks = self.get_chunks(user_id)
        
        if not index or not chunks:
            return []
        
        query_embedding = self.model_manager.get_embedding(query)
        distances, indices = index.search(
            query_embedding.reshape(1, -1), 
            TOP_K_RESULTS
        )
        
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx != -1 and dist < SIMILARITY_THRESHOLD:
                results.append(chunks[idx])
        
        return results[:TOP_K_RESULTS]

    async def add_documents(self, user_id: str, documents: List[str]) -> bool:
        try:
            dimension = self.model_manager.get_embedding("test").shape[0]
            index = self.get_index(user_id) or faiss.IndexFlatL2(dimension)
            chunks = self.get_chunks(user_id)
            
            embeddings = self.model_manager.batch_get_embeddings(documents)
            index.add(embeddings)
            chunks.extend(documents)
            
            # Save updated index and chunks
            index_path = os.path.join(self.base_dir, f"{user_id}_index.faiss")
            chunks_path = os.path.join(self.base_dir, f"chunks_{user_id}.json")
            
            faiss.write_index(faiss.index_gpu_to_cpu(index) if torch.cuda.is_available() else index, index_path)
            with open(chunks_path, 'w') as f:
                json.dump(chunks, f)
            
            # Update cache
            self.index_cache[user_id] = index
            self.chunk_cache[user_id] = chunks
            
            return True
        except Exception as e:
            print(f"Error adding documents: {str(e)}")
            return False

class OptimizedConversationManager:
    def __init__(self):
        self.conversations = {}
        self.max_history = 3
    
    def get_or_create_conversation(self, conversation_id: Optional[str] = None) -> str:
        if conversation_id and conversation_id in self.conversations:
            return conversation_id
        new_id = str(uuid.uuid4())
        self.conversations[new_id] = []
        return new_id
    
    def add_interaction(self, conversation_id: str, query: str, response: str):
        conv = self.conversations.get(conversation_id, [])
        conv.append({
            "query": query,
            "response": response,
            "timestamp": datetime.now().isoformat()
        })
        self.conversations[conversation_id] = conv[-self.max_history:]

    def get_context(self, conversation_id: str) -> str:
        if conversation_id not in self.conversations:
            return ""
        history = self.conversations[conversation_id]
        return "\n".join([
            f"Q: {h['query']}\nA: {h['response']}" 
            for h in history[-2:]
        ])

async def generate_response(
    model_manager: ModelManager, 
    query: str, 
    context: str, 
    history: str
) -> str:
    prompt = f"Context: {context}\nHistory: {history}\nQuestion: {query}\nAnswer:"
    
    inputs = model_manager.llama_tokenizer(
        prompt,
        return_tensors="pt",
        max_length=MAX_CONTEXT_WINDOW,
        truncation=True
    ).to(model_manager.device)
    
    with torch.inference_mode():
        outputs = model_manager.llama_model.generate(
            **inputs,
            max_new_tokens=150,
            num_return_sequences=1,
            temperature=0.7,
            do_sample=True,
            pad_token_id=model_manager.llama_tokenizer.eos_token_id
        )
    
    response = model_manager.llama_tokenizer.decode(
        outputs[0], 
        skip_special_tokens=True
    )
    return response.split("Answer:")[-1].strip()

# Initialize global managers
model_manager = ModelManager()
vector_db = OptimizedVectorDB(BASE_DIR)
conversation_manager = OptimizedConversationManager()

@app.post("/upload")
async def upload_files(
    user_id: str = Form(...), 
    files: List[UploadFile] = File(...)
) -> UploadResponse:
    processed_files = []
    failed_files = []
    
    for file in files:
        try:
            content = await file.read()
            text_chunks = content.decode().split("\n")  # Simple chunking
            success = await vector_db.add_documents(user_id, text_chunks)
            if success:
                processed_files.append(file.filename)
            else:
                failed_files.append({
                    "filename": file.filename,
                    "error": "Failed to process file"
                })
        except Exception as e:
            failed_files.append({
                "filename": file.filename,
                "error": str(e)
            })
    
    return UploadResponse(
        status="success",
        processed_files=processed_files,
        failed_files=failed_files,
        processed_percentage=(len(processed_files) / len(files)) * 100
    )

@app.post("/agents/{agent_id}/conversations")
async def handle_conversation(
    agent_id: str,
    query: str = Form(...),
    conversation_id: Optional[str] = None
) -> ConversationResponse:
    start_time = datetime.now()
    
    conversation_id = conversation_manager.get_or_create_conversation(conversation_id)
    history = conversation_manager.get_context(conversation_id)
    
    # Parallel processing of vector search
    context_future = asyncio.create_task(vector_db.search(agent_id, query))
    retrieved_contexts = await context_future
    
    # Generate response
    context = "\n".join(retrieved_contexts) if retrieved_contexts else "No relevant context found."
    response = await generate_response(model_manager, query, context, history)
    
    # Update conversation
    conversation_manager.add_interaction(conversation_id, query, response)
    
    end_time = datetime.now()
    response_time = (end_time - start_time).total_seconds()
    
    return ConversationResponse(
        conversation_id=conversation_id,
        answer=response,
        response_time=response_time,
        context_found=bool(retrieved_contexts)
    )

def configure_ngrok(port: int = 8000, auth_token: Optional[str] = None):
    if auth_token:
        ngrok.set_auth_token(auth_token)
    
    ngrok.kill()
    
    try:
        public_url = ngrok.connect(port).public_url
        print(f"\n🚀 Ngrok tunnel established!")
        print(f"📡 Public URL: {public_url}")
        print(f"🔗 API endpoint: {public_url}/docs")
        return public_url
    except Exception as e:
        print(f"❌ Failed to establish ngrok tunnel: {str(e)}")
        return None

@app.on_event("startup")
async def startup_event():
    print("🔧 Initializing AI Agent...")
    # Models are initialized through singleton
    print("✅ All components initialized")

def run_with_ngrok(port: int = 8000, auth_token: Optional[str] = None):
    try:
        public_url = configure_ngrok(port, auth_token)
        if not public_url:
            raise Exception("Failed to establish ngrok tunnel")

        print(f"\n🌟 Starting FastAPI server on port {port}...")
        uvicorn.run(app, host="0.0.0.0", port=port)

    except Exception as e:
        print(f"❌ Error starting server: {str(e)}")
        ngrok.kill()

if __name__ == "__main__":
    print("🚀 Starting AI Agent with Ngrok...")
    run_with_ngrok(port=8000)