# Local RAG Chat System

This project provides a fully local Retrieval-Augmented Generation (RAG) application built with Streamlit and LangChain.  
It allows you to upload PDF and DOCX files, converts them into vector embeddings, stores them locally using ChromaDB, and lets you chat with your documents using an Ollama-powered model.

The project is designed to run fully locally. No cloud dependencies are required.

## Features

- Upload and process PDF or DOCX documents  
- Automatically split documents into overlapping chunks for better retrieval  
- Use HuggingFace embeddings (MiniLM-L6-v2) for vector encoding  
- Persistent Chroma vector database  
- Simple conversational chat interface for querying your knowledge base  
- Local language model inference via Ollama  
- Option to fully clear knowledge base and reset embeddings

## Project Structure

```
project/
│
├── rag_system.py
├── app.py
├── chroma_db/
├── requirements.txt
└── README.md
```

## Installation

### 1. Install Python
Python 3.10+ is required.

### 2. Install dependencies
```
pip install -r requirements.txt
```

### 3. Install and configure Ollama
Download Ollama from:
https://ollama.com/download

Then pull a model:
```
ollama pull llama3.2
```

## How It Works

### 1. Document Loading  
PDFs use PyPDFLoader, DOCX uses Docx2txtLoader.

### 2. Text Splitting  
Documents are broken into chunks (1000 characters, 200 overlap).

### 3. Embedding  
Chunks are embedded using MiniLM-L6-v2.

### 4. Vector Storage  
Stored in a local Chroma database.

### 5. Retrieval  
Relevant chunks are retrieved based on similarity using a `Retriever`.

### 6. Generation  
The LLM answers using the retrieved context.

### 7. Knowledge Base Management
The system allows clearing the entire Chroma database and session state to remove all embeddings and chunks at any time.

## Running the App

```
streamlit run app.py
```

## Using the Interface

1. Upload documents (PDF/DOCX)  
2. System processes and stores them  
3. Ask questions in the chat  
4. Model retrieves relevant chunks and responds  
5. Clear knowledge base anytime using the sidebar button

## Troubleshooting

### Slow ingestion  
Large PDFs may take longer due to splitting and embedding.

### LangChain errors  
Ensure required LangChain modules are installed.

### Chroma lock issues  
Delete the `chroma_db/` directory and restart the app.

### Ollama issues  
Ensure Ollama is running:
```
ollama serve
```


