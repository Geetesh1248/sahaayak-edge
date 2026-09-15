import gradio as gr
import pdfplumber
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import ollama
import socket
import time
import os

# --- Configurations ---
MODEL_NAME = "llama3.2:1b"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
CHUNK_SIZE = 500
CONFIDENCE_THRESHOLD = 1.5  # L2 distance threshold for low confidence

# Global state for index and chunks
faiss_index = None
chunk_data = [] # List of dicts: {"text": chunk, "page": page_number}
embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)

# --- Helper Functions ---
def check_offline_mode():
    """Checks if there's internet connectivity to determine online/offline mode."""
    try:
        # Try to resolve a known host
        socket.create_connection(("1.1.1.1", 53), timeout=2)
        return "Offline Mode: OFF 🟢"
    except OSError:
        pass
    return "Offline Mode: ON 🔴"

def process_pdf(pdf_path):
    """Extracts text from PDF, chunks it, and builds a FAISS index."""
    global faiss_index, chunk_data
    
    if not pdf_path or not os.path.exists(pdf_path):
        return "Error: Please upload a valid PDF."
    
    chunk_data = []
    
    # Extract text with page numbers
    try:
        with pdfplumber.open(pdf_path) as pdf:
            if len(pdf.pages) == 0:
                return "Error: The uploaded PDF has no pages."
                
            for page_num, page in enumerate(pdf.pages, start=1):
                text = page.extract_text()
                if text:
                    # Simple chunking by splitting text
                    words = text.split()
                    for i in range(0, len(words), int(CHUNK_SIZE / 5)): # Roughly 100 words per chunk
                        chunk_words = words[i:i + int(CHUNK_SIZE / 5)]
                        chunk_text = " ".join(chunk_words)
                        if len(chunk_text.strip()) > 10:
                            chunk_data.append({
                                "text": chunk_text,
                                "page": page_num
                            })
    except Exception as e:
        return f"Error processing PDF: {e}"

    if not chunk_data:
        return "Error: No text could be extracted. The PDF might be empty or image-only."

    # Create Embeddings
    texts = [c["text"] for c in chunk_data]
    embeddings = embedder.encode(texts)
    
    # Build FAISS index (L2 distance)
    dimension = embeddings.shape[1]
    faiss_index = faiss.IndexFlatL2(dimension)
    faiss_index.add(np.array(embeddings).astype("float32"))
    
    return f"PDF processed successfully. Extracted {len(chunk_data)} chunks."

def answer_question(question):
    """Retrieves relevant context and generates an answer using Ollama."""
    global faiss_index, chunk_data
    
    if faiss_index is None or not chunk_data:
        return "Please upload and process a lab manual PDF first.", ""

    start_time = time.time()
    retrieval_start = time.time()

    # 1. Retrieve Context
    query_embedding = embedder.encode([question]).astype("float32")
    distances, indices = faiss_index.search(query_embedding, k=3)
    
    best_distance = distances[0][0]
    
    retrieval_latency = (time.time() - retrieval_start) * 1000
    print(f"Retrieval Latency: {retrieval_latency:.2f} ms")
    print(f"Retrieval - Best Distance: {best_distance:.4f}")

    # Fallback for low confidence
    if best_distance > CONFIDENCE_THRESHOLD:
        total_latency = (time.time() - start_time) * 1000
        print(f"Total Response Latency: {total_latency:.2f} ms")
        return "I'm not confident about this — please verify with your instructor", "Source: N/A (Low Confidence)"

    # Gather context chunks
    context = ""
    source_pages = set()
    for idx in indices[0]:
        if idx < len(chunk_data):
            chunk = chunk_data[idx]
            context += chunk["text"] + "\n\n"
            source_pages.add(chunk["page"])
            
    source_str = "Source: Page(s) " + ", ".join(map(str, sorted(source_pages)))

    # 2. Generate Answer with Local LLM
    prompt = f"""You are a helpful engineering lab assistant. Answer the student's question based strictly on the provided lab manual context. If the answer is not in the context, say you don't know based on the manual.
    
Context:
{context}

Question: {question}

Answer:"""
    
    try:
        response = ollama.generate(model=MODEL_NAME, prompt=prompt)
        answer = response['response']
    except ollama.ResponseError as e:
        if 'not found' in str(e).lower():
            answer = f"Error: Model '{MODEL_NAME}' not found. Please run 'ollama pull {MODEL_NAME}' in your terminal."
        else:
            answer = f"Error calling Ollama API: {e}"
    except Exception as e:
        answer = f"Error: Could not connect to Ollama. Is the Ollama app running? Details: {e}"

    total_latency = (time.time() - start_time) * 1000
    print(f"Total Response Latency: {total_latency:.2f} ms")

    return answer, source_str

# --- Gradio UI ---
with gr.Blocks(title="Sahaayak Edge") as demo:
    gr.Markdown("# Sahaayak Edge — Offline Lab Assistant")
    gr.Markdown("*CPU prototype validated locally. Snapdragon deployment is the next validation stage.*")
    
    offline_status = gr.Markdown(check_offline_mode())
    
    with gr.Row():
        with gr.Column():
            pdf_input = gr.File(label="Upload Lab Manual (PDF)", file_types=[".pdf"])
            process_btn = gr.Button("Process PDF")
            process_status = gr.Textbox(label="Status", interactive=False)
            
            question_input = gr.Textbox(label="Ask a question about the lab manual")
            ask_btn = gr.Button("Ask")
            
        with gr.Column():
            answer_output = gr.Textbox(label="Answer", lines=10)
            source_output = gr.Textbox(label="Source", interactive=False)

    # Event handlers
    process_btn.click(
        fn=process_pdf,
        inputs=[pdf_input],
        outputs=[process_status]
    )
    
    ask_btn.click(
        fn=answer_question,
        inputs=[question_input],
        outputs=[answer_output, source_output]
    )

if __name__ == "__main__":
    demo.launch()
