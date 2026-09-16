from fastapi import FastAPI, UploadFile, File
import fitz
import numpy as np
import ollama
from sentence_transformers import SentenceTransformer

app = FastAPI()

# Load embedding model once
model = SentenceTransformer("all-MiniLM-L6-v2")

# Store document chunks and embeddings
stored_chunks = []
stored_embeddings = []


@app.get("/")
def home():
    return {"message": "Document Q&A API is running"}

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "message": "Document Q&A API is running"
    }


def split_text(text, chunk_size=1000):
    chunks = []

    for i in range(0, len(text), chunk_size):
        chunk = text[i:i + chunk_size]
        chunks.append(chunk)

    return chunks


@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    global stored_chunks, stored_embeddings

    contents = await file.read()

    document = fitz.open(
        stream=contents,
        filetype="pdf"
    )

    text = ""

    for page in document:
        text += page.get_text()

    chunks = split_text(text)

    # Generate embeddings
    embeddings = model.encode(chunks)

    # Store chunks and embeddings
    stored_chunks = chunks
    stored_embeddings = embeddings

    return {
        "filename": file.filename,
        "total_characters": len(text),
        "total_chunks": len(chunks),
        "embedding_dimensions": len(embeddings[0]),
        "message": "Document stored successfully"
    }


@app.post("/ask")
async def ask_question(question: str):

    if not stored_chunks:
        return {
            "error": "Please upload a document first"
        }

    # Convert question into embedding
    question_embedding = model.encode([question])[0]

    # Calculate similarity
    similarities = []

    for embedding in stored_embeddings:

        similarity = np.dot(
            question_embedding,
            embedding
        ) / (
            np.linalg.norm(question_embedding)
            * np.linalg.norm(embedding)
        )

        similarities.append(similarity)

    # Find most relevant chunk
    top_indices = np.argsort(similarities)[-5:][::-1]

    relevant_chunks = []

    for index in top_indices:
        relevant_chunks.append(stored_chunks[index])

    relevant_chunk = "\n\n".join(relevant_chunks)
    # Send retrieved information to Llama
    prompt = f"""
You are a document question-answering assistant.

Answer the question using ONLY the information explicitly stated in the document.

IMPORTANT RULES:
- Do not add information from your own knowledge.
- Do not guess or assume anything.
- Do not mention technologies, features, or details unless they appear in the document information below.
- If the information is not clearly present, say:
"I could not find the answer in the document."

Document information:
{relevant_chunk}

Question:
{question}

Answer:
"""

    response = ollama.chat(
        model="llama3.2:3b",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    answer = response["message"]["content"]

    return {
    "question": question,
    "answer": answer,
    "similarity_score": float(max(similarities)),
    "sources": relevant_chunks
}