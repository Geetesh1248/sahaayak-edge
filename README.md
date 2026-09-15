# Sahaayak Edge

**Sahaayak Edge** is an offline, local RAG-based (Retrieval-Augmented Generation) Q&A assistant designed specifically for engineering lab manuals. 

## Problem Statement
Engineering students frequently face unreliable internet access in hardware laboratories and workshops, making it difficult to access online documentation or cloud-based AI assistants when troubleshooting equipment. Sahaayak Edge solves this by providing a completely offline, low-latency AI assistant that references local lab manuals securely on the user's machine.

## Architecture

```text
[User PDF] --> (pdfplumber) --> [Text Chunks] 
                                      |
                                      v
[User Query] --> (sentence-transformers) --> [FAISS Vector Store]
                                      |
                                      v
[Retrieved Context + Query] --> (Local LLM via Ollama) --> [Answer Output]
```

## Features
### Implemented
- **100% Offline Mode**: Operates entirely locally with an offline status indicator.
- **Local RAG**: Uses `sentence-transformers` for embeddings and a local `FAISS` index for fast retrieval.
- **Local LLM**: Generates answers using `llama3.2:1b` (via Ollama) to ensure privacy and offline capability.
- **Citation & Source Page**: Displays the exact page number from the uploaded PDF for the referenced answer.
- **Low Confidence Fallback**: Refuses to guess when the retrieval confidence is low, advising the student to ask an instructor.
- **Latency Tracking**: Measures and logs response generation times.

### Planned / Future Work
- Camera recognition (for equipment identification).
- Voice I/O (hands-free interaction during lab experiments).
- Automatic safety rules injection and verification.

## Tech Stack
- **UI**: Gradio
- **PDF Extraction**: pdfplumber
- **Embeddings**: sentence-transformers (all-MiniLM-L6-v2)
- **Vector Store**: FAISS (cpu)
- **LLM Engine**: Ollama (llama3.2:1b)
- **Language**: Python 3.10+

## Setup Instructions

### 1. Prerequisites
Ensure you have **Python 3.10+** installed.
You must also have **Ollama** installed on your system. You can download it from [ollama.com](https://ollama.com).

Once Ollama is installed, open your terminal and pull the required model:
```bash
ollama pull llama3.2:1b
```

### 2. Install Dependencies
Clone this repository and install the required Python packages:

```bash
git clone https://github.com/Geetesh1248/sahaayak-edge.git
cd sahaayak-edge
pip install -r requirements.txt
```

### 3. Run the Application
Execute the main application file:
```bash
python app.py
```
Open the provided local URL (typically `http://127.0.0.1:7860`) in your web browser.

## Hardware & Deployment Note
This application was developed and tested on a Windows machine. It is architected with a lightweight model and local retrieval setup, designed specifically for future deployment and acceleration via **ONNX Runtime** and **Qualcomm AI Hub** on **Snapdragon NPU** hardware to maximize performance and battery life on edge devices.

## Known Limitations
- Heavy PDFs might take a moment to process on older CPUs.
- The default LLM (`llama3.2:1b`) is small to prioritize speed and low resource usage, so very complex reasoning might be limited compared to larger cloud models.

## License
MIT License
