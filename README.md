# Sahaayak Edge — Offline Lab Assistant

Sahaayak Edge is an offline, local RAG-based Q&A assistant for engineering lab manuals.

## Problem Statement

Engineering students often need answers from long laboratory manuals while working in environments with unreliable internet connectivity. Cloud AI may expose lab documents and student questions, so Sahaayak Edge provides a local, privacy-preserving alternative.

## What is implemented

- Gradio web UI
- PDF upload
- pdfplumber page-aware text extraction
- Text chunking
- sentence-transformers all-MiniLM-L6-v2 local embeddings
- FAISS local vector search
- Ollama local LLM using llama3.2:1b
- Page-number citations
- Offline status indicator
- Retrieval confidence fallback
- Response latency logging

## What is not implemented yet (Future Work)

- Camera/component recognition
- Voice input/output
- Advanced laboratory safety rules
- Snapdragon NPU benchmark
- Qualcomm AI Hub production deployment

## Architecture

```text
  PDF manual
      |
      v
  pdfplumber page-aware extraction
      |
      v
  chunking + local embeddings
      |
      v
  FAISS local retrieval
      |
      v
  confidence check
      |
      v
  Ollama llama3.2:1b local generation
      |
      v
  answer + source page citation
```

## Privacy & Offline Behavior

After the one-time model download, document extraction, retrieval, and answer generation are intended to run locally. This preserves privacy and ensures the application can function entirely offline.

## Measured Results (CPU-Only)

The current prototype has been validated with the following CPU-only measurements:
- 7 chunks extracted from the sample lab manual
- approximately 30 ms retrieval-only latency
- 3.3–63 seconds full LLM generation on CPU, with high variance
- offline behavior confirmed with Wi-Fi disabled

## Snapdragon Deployment Plan

The next step is to replace or export compatible model components using ONNX Runtime and/or Qualcomm AI Hub, then validate the application on Snapdragon-powered HP PC hardware. Sub-second generation is a target for this future deployment, not a measured result of the current CPU prototype.

## Test Environment

- CPU-only Windows laptop
- No Snapdragon NPU was available during development

## Known Limitations

- Current answer quality depends on the uploaded manual
- Scanned/image-only PDFs may not extract correctly
- The safety fallback is not a substitute for an instructor
- Current prototype uses Ollama and requires the local model to be installed
- Snapdragon performance is not yet validated

## Setup Instructions

### Prerequisites
- Python 3.10+
- Ollama: Download and install from [ollama.com](https://ollama.com).

### Pull the Local LLM
Once Ollama is installed, open your terminal and run the exact command to pull the required model:
```bash
ollama pull llama3.2:1b
```

### Install Dependencies
Clone the repository, then install requirements:
```bash
pip install -r requirements.txt
```

### Run the App
```bash
python app.py
```

## Troubleshooting
- **Missing Ollama**: If you see connection errors during answer generation, ensure the Ollama app is running in the background.
- **Missing Model**: If Ollama is running but generation fails, ensure you ran `ollama pull llama3.2:1b`.
- **Invalid/Image-only PDF**: If no text is extracted, ensure the PDF contains text layers (not just scanned images).
- **Port Conflicts**: If port 7860 is in use, Gradio will automatically try the next available port. Check the terminal output for the correct URL.

## License
MIT License
