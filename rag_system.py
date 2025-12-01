import os
from pathlib import Path
from typing import List
import tempfile
import shutil

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser


class LocalRAGSystem:
    def __init__(self, persist_directory="./chroma_db"):
        self.persist_directory = persist_directory
        self.vectorstore = None
        self.retriever = None
        self.llm = None
        self.rag_chain = None

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )

    def load_documents(self, file_paths: List[str]):
        """Load documents from the given file paths."""
        documents = []

        for file_path in file_paths:
            file_path = Path(file_path)
            if file_path.suffix.lower() == ".pdf":
                loader = PyPDFLoader(str(file_path))
            elif file_path.suffix.lower() in [".docx", ".doc"]:
                loader = Docx2txtLoader(str(file_path))
            else:
                raise ValueError(f"Unsupported file type: {file_path.suffix}")

            documents.extend(loader.load())

        return documents

    def create_vectorstore(self, documents):
        """Create a new vectorstore from documents."""
        embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"},
        )

        splits = self.text_splitter.split_documents(documents)

        self.vectorstore = Chroma.from_documents(
            documents=splits,
            embedding=embeddings,
            persist_directory=self.persist_directory,
        )
        self.vectorstore.persist()
        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 4})

    def setup_rag_chain(self, model_name: str = "llama3.2"):
        """Setup the RAG chain with the specified model."""
        if self.retriever is None:
            raise ValueError("Vectorstore must be created before setting up RAG chain")

        self.llm = OllamaLLM(model=model_name)

        prompt_template = """You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Do not make up an answer.

Context: {context}

Question: {question}

Answer:"""

        prompt = PromptTemplate.from_template(prompt_template)

        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)

        self.rag_chain = (
            {"context": self.retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | self.llm
            | StrOutputParser()
        )

    def query(self, question: str) -> str:
        """Query the RAG system."""
        if self.rag_chain is None:
            raise ValueError("RAG chain must be set up before querying")
        return self.rag_chain.invoke(question)

    def load_existing_vectorstore(self):
        """Load an existing vectorstore."""
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        self.vectorstore = Chroma(
            persist_directory=self.persist_directory,
            embedding_function=embeddings,
        )
        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 4})

    def get_document_count(self):
        """Get the number of documents in the vectorstore."""
        if self.vectorstore is None:
            return 0
        try:
            return len(self.vectorstore.get()["ids"])
        except Exception:
            return 0
