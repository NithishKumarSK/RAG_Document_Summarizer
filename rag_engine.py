# rag_engine.py


import os
import chromadb
from chromadb.config import Settings
from pypdf import PdfReader
from docx import Document
import ollama
import hashlib

class RAGEngine:
    def __init__(self, persist_dir="./chroma_db"):
        self.persist_dir = persist_dir
        self.client = chromadb.Client(Settings(
            persist_directory=persist_dir,
            anonymized_telemetry=False
        ))
        
        try:
            self.collection = self.client.get_collection("documents")
        except:
            self.collection = self.client.create_collection(
                name="documents",
                metadata={"hnsw:space": "cosine"}
            )
    
    def extract_text(self, filepath):
        """Extract text from PDF, DOCX, or TXT files"""
        ext = os.path.splitext(filepath)[1].lower()
        
        if ext == '.pdf':
            reader = PdfReader(filepath)
            text = ""
            for page in reader.pages:
                text += page.extract_text() + "\n"
            return text
        
        elif ext == '.docx':
            doc = Document(filepath)
            return "\n".join([para.text for para in doc.paragraphs])
        
        elif ext == '.txt':
            with open(filepath, 'r', encoding='utf-8') as f:
                return f.read()
        
        else:
            raise ValueError(f"Unsupported file type: {ext}")
    
    def chunk_text(self, text, chunk_size=1000, overlap=200):
        """Split text into overlapping chunks"""
        chunks = []
        start = 0
        text_len = len(text)
        
        while start < text_len:
            end = start + chunk_size
            chunk = text[start:end]
            
            if chunk.strip():
                chunks.append(chunk)
            
            start += (chunk_size - overlap)
        
        return chunks
    
    def get_embedding(self, text):
        """Generate embedding using Ollama"""
        response = ollama.embeddings(
            model="nomic-embed-text",
            prompt=text
        )
        return response["embedding"]
    
    def add_document(self, filepath, filename):
        """Process and add document to vector database"""
        text = self.extract_text(filepath)
        chunks = self.chunk_text(text)
        
        doc_id = hashlib.md5(filename.encode()).hexdigest()
        
        for i, chunk in enumerate(chunks):
            chunk_id = f"{doc_id}_{i}"
            embedding = self.get_embedding(chunk)
            
            self.collection.add(
                ids=[chunk_id],
                embeddings=[embedding],
                documents=[chunk],
                metadatas=[{"filename": filename, "chunk_index": i}]
            )
        
        return len(chunks)
    
    def query(self, question, n_results=10):
        """Query the vector database"""
        query_embedding = self.get_embedding(question)
        
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results
        )
        
        if not results['documents'][0]:
            return None, []
        
        context = "\n\n".join(results['documents'][0])
        sources = [meta['filename'] for meta in results['metadatas'][0]]
        
        prompt = f"""Based on the following context from multiple documents, answer the question accurately and concisely.

Context:
{context}

Question: {question}

Answer (be specific and cite which document if relevant):"""
        
        response = ollama.generate(
            model="llama3.2",  # Faster model
            prompt=prompt
        )
        
        return response['response'], list(set(sources))
    
    def summarize(self, filename=None):
        """Generate summary of documents"""
        if filename:
            results = self.collection.get(
                where={"filename": filename}
            )
        else:
            results = self.collection.get()
        
        if not results['documents']:
            return "No documents found."
        
        # Get more chunks for better summary
        sample_text = "\n".join(results['documents'][:20])
        
        prompt = f"""Summarize the following document content in 5-7 bullet points, highlighting the main points and key information:

{sample_text}

Summary:"""
        
        response = ollama.generate(
            model="llama3.2",  # Faster model
            prompt=prompt
        )
        
        return response['response']
    
    def get_stats(self):
        """Get database statistics"""
        count = self.collection.count()
        return {"total_chunks": count}