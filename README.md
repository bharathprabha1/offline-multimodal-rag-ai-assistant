# Offline Multimodal RAG AI Assistant

A privacy-focused Offline Retrieval-Augmented Generation (RAG) AI Assistant designed for intelligent document search, contextual question answering, semantic retrieval, and local AI processing.

## Features

- Offline Retrieval-Augmented Generation (RAG)
- Local Large Language Model (LLM) integration
- Semantic document search
- PDF and DOCX document processing
- Speech-to-Text using Vosk
- Text-to-Speech using pyttsx3
- OCR support using Tesseract
- Knowledge visualization
- Conversation history
- Interactive desktop GUI
- Privacy-focused local processing

## Technology Stack

- Python
- Retrieval-Augmented Generation (RAG)
- Sentence Transformers
- FAISS / Vector Search
- PyTorch
- Transformers
- Vosk
- Tesseract OCR
- NetworkX
- Graphviz
- CustomTkinter
- PyPDF2
- python-docx

## RAG Pipeline

```text
Documents
    ↓
Document Processing
    ↓
Text Extraction / OCR
    ↓
Chunking
    ↓
Semantic Embeddings
    ↓
Vector Retrieval
    ↓
Relevant Context
    ↓
Local LLM
    ↓
Contextual Answer