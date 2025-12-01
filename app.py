# app.py
import streamlit as st
import os
import shutil
from pathlib import Path
import tempfile
from rag_system import LocalRAGSystem

st.set_page_config(page_title="Local RAG Chat", page_icon="Books", layout="wide")

# Cache the RAG system (but allow reset)
def load_rag_system():
    return LocalRAGSystem(persist_directory="./chroma_db")

# Clear everything: Chroma DB + session state
def clear_knowledge_base():
    chroma_path = "./chroma_db"
    if os.path.exists(chroma_path):
        shutil.rmtree(chroma_path)
        st.success("Knowledge base cleared! All chunks and embeddings deleted.")
    else:
        st.info("No knowledge base to clear.")
    
    # Reset session state
    if "messages" in st.session_state:
        st.session_state.messages = []
    st.rerun()

def main():
    st.title("Local RAG with Ollama")

    # Initialize or reload RAG system
    if "rag_system" not in st.session_state:
        st.session_state.rag_system = load_rag_system()

    rag_system = st.session_state.rag_system

    # Sidebar
    with st.sidebar:
        st.header("Document Management")

        uploaded_files = st.file_uploader(
            "Upload PDF/DOCX files",
            type=['pdf', 'docx'],
            accept_multiple_files=True,
            help="Files are processed temporarily and never stored permanently."
        )

        st.markdown("---")
        st.subheader("Knowledge Base Control")

        col1, col2 = st.columns(2)
        with col1:
            if st.button("Clear All Data", type="secondary"):
                st.session_state.show_confirm = True

        if st.session_state.get("show_confirm", False):
            st.warning("Are you sure you want to delete the entire knowledge base?")
            col_a, col_b = st.columns(2)
            with col_a:
                if st.button("Yes, Delete Everything", type="primary"):
                    clear_knowledge_base()
                    st.session_state.show_confirm = False
            with col_b:
                if st.button("Cancel"):
                    st.session_state.show_confirm = False
                    st.rerun()

        # Show current size
        try:
            count = rag_system.get_document_count()
            st.metric("Chunks in Memory", count)
        except:
            st.metric("Chunks in Memory", 0)

    # Main area
    col1, col2 = st.columns([3, 1])
    with col2:
        st.caption("lucas")

    # Process uploaded files
    if uploaded_files:
        with st.spinner(f"Processing {len(uploaded_files)} document(s)..."):
            with tempfile.TemporaryDirectory() as temp_dir:
                file_paths = []
                for uploaded_file in uploaded_files:
                    file_path = Path(temp_dir) / uploaded_file.name
                    with open(file_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    file_paths.append(str(file_path))

                documents = rag_system.load_documents(file_paths)
                rag_system.create_vectorstore(documents)
                rag_system.setup_rag_chain()

            st.success(f"Loaded {len(uploaded_files)} document(s) → {rag_system.get_document_count()} chunks ready!")

    # Chat interface
    if rag_system.get_document_count() > 0:
        st.success("Knowledge base ready! Ask questions below.")

        if "messages" not in st.session_state:
            st.session_state.messages = []

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        if prompt := st.chat_input("Ask a question about your documents..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    try:
                        response = rag_system.query(prompt)
                        st.markdown(response)
                        st.session_state.messages.append({"role": "assistant", "content": response})
                    except Exception as e:
                        st.error(f"Error: {e}")

    else:
        st.info("Upload some PDFs or DOCX files to start chatting with your documents!")

    # Footer
    st.markdown("---")
    st.caption("Your uploaded files are never stored permanently. Chroma DB is stored locally but can be fully cleared anytime.")

if __name__ == "__main__":
    main()