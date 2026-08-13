# 🧠 Offline Multimodal RAG AI Assistant

> A privacy-focused, offline AI knowledge assistant for intelligent document retrieval, contextual question answering, semantic search, and local LLM inference.

---

## 📌 Overview

The **Offline Multimodal RAG AI Assistant** is a desktop-based Retrieval-Augmented Generation (RAG) application designed to provide AI-powered question answering without depending on cloud-based AI services.

The system allows users to provide documents and audio files, extracts meaningful information, converts the content into semantic embeddings, retrieves relevant knowledge, and generates contextual responses using a locally running Large Language Model (LLM).

The project focuses on **privacy, offline accessibility, intelligent retrieval, and multimodal interaction**.

---

## ✨ Key Features

### 📚 Intelligent Document Processing

- PDF document extraction
- DOCX document processing
- TXT file processing
- WAV audio transcription
- Automatic text cleaning
- Intelligent text chunking
- Local document indexing

### 🔎 Semantic Search

- Transformer-based text embeddings
- Semantic similarity matching
- Context-aware retrieval
- Relevant knowledge selection
- Cached embeddings for faster processing

### 🤖 Local AI / RAG

- Retrieval-Augmented Generation architecture
- Local Large Language Model inference
- Context-grounded responses
- Offline AI processing
- No mandatory cloud AI API dependency

### 🎤 Voice Interaction

- Offline speech recognition using Vosk
- Text-to-speech responses
- Audio document transcription
- Voice-based AI interaction

### 🌐 Multilingual Support

- Text translation capabilities
- Multilingual interaction support using translation services

### 📊 Visualization

- RAG-related visualizations
- Graph-based knowledge visualization
- Matplotlib-based analytics

### 💬 Conversation Management

- Persistent chat history
- Multiple conversation sessions
- Automatic conversation titles
- Timestamped messages

### 🖥️ Modern Desktop GUI

- Built with CustomTkinter
- Dark-mode interface
- Interactive AI assistant experience
- File/folder selection
- Processing status feedback
- Voice interaction controls

---

# 🏗️ System Architecture

```text
                    ┌──────────────────────────┐
                    │      User Interface      │
                    │      CustomTkinter       │
                    └────────────┬─────────────┘
                                 │
                  ┌──────────────┼──────────────┐
                  │              │              │
                  ▼              ▼              ▼
              Documents        Audio         User Query
             PDF/DOCX/TXT       WAV              │
                  │              │               │
                  ▼              ▼               │
            Text Extraction   Vosk ASR           │
                  │              │               │
                  └───────┬──────┘               │
                          ▼                      │
                   Text Processing              │
                          │                      │
                          ▼                      │
                    Text Chunking               │
                          │                      │
                          ▼                      │
                Sentence Transformer            │
                  Semantic Embeddings            │
                          │                      │
                          ▼                      ▼
                    Vector Memory        Query Embedding
                          │                      │
                          └──────────┬───────────┘
                                     ▼
                             Similarity Retrieval
                                     │
                                     ▼
                              Relevant Context
                                     │
                                     ▼
                              Local LLM
                             llama.cpp / GGUF
                                     │
                                     ▼
                              AI Response
                                     │
                       ┌─────────────┴─────────────┐
                       ▼                           ▼
                Chat History                 Text-to-Speech