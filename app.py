import streamlit as st
import os
import shutil
from pathlib import Path
import tempfile
from rag_system import LocalRAGSystem

st.set_page_config(page_title="Local RAG Chat", page_icon="📚", layout="wide")

@st.cache_resource
def load_rag_system():
    return LocalRAGSystem()

def clear_vectorstore():
    if os.path.exists("./chroma_db"):
        shutil.rmtree("./chroma_db")
        st.success("Vectorstore cleared successfully!")

def main():
    st.title("Local RAG")
    
    rag_system = load_rag_system()
    
    # Sidebar
    with st.sidebar:
        st.header("Document Management")
        
        uploaded_files = st.file_uploader(
            "Upload PDF or DOCX files",
            type=['pdf', 'docx'],
            accept_multiple_files=True,
            help="Upload documents to create a knowledge base for chatting"
        )
        
        if st.button("Clear Knowledge Base", type="primary"):
            if st.button("Confirm Clear Knowledge Base", type="primary"):
                clear_vectorstore()
                st.session_state.clear()
                st.rerun()
    
    # Main interface
    col1, col2 = st.columns([3, 1])
    
    with col2:
        document_count = rag_system.get_document_count()
        st.metric("Documents in Knowledge Base", document_count)
    
    # File processing
    if uploaded_files:
        with st.spinner("Processing uploaded documents..."):
            # Create a temporary directory for uploaded files
            with tempfile.TemporaryDirectory() as temp_dir:
                file_paths = []
                for uploaded_file in uploaded_files:
                    file_path = Path(temp_dir) / uploaded_file.name
                    with open(file_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    file_paths.append(str(file_path))
                
                # Load and process documents
                documents = rag_system.load_documents(file_paths)
                rag_system.create_vectorstore(documents)
                rag_system.setup_rag_chain()
                
                st.success(f"Successfully processed {len(uploaded_files)} document(s). "
                         f"Knowledge base contains {rag_system.get_document_count()} chunks.")
    
    # Check if vectorstore exists and is ready for querying
    try:
        if rag_system.get_document_count() > 0:
            st.success("Knowledge base is ready. You can now ask questions about the uploaded documents.")
            
            # Chat interface
            if "messages" not in st.session_state:
                st.session_state.messages = []
            
            # Display chat messages
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])
            
            # Chat input
            if prompt := st.chat_input("Ask a question about the uploaded documents:"):
                st.session_state.messages.append({"role": "user", "content": prompt})
                with st.chat_message("user"):
                    st.markdown(prompt)
                
                with st.chat_message("assistant"):
                    with st.spinner("Thinking..."):
                        response = rag_system.query(prompt)
                    st.markdown(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})
                    
        else:
            st.info(
                "Please upload PDF or DOCX documents to create a knowledge base. "
                "Once documents are uploaded, you can ask questions about their content."
            )
            
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()