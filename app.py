import streamlit as st
import tempfile
import time
import os
import shutil
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate

st.set_page_config(page_title="Local RAG", layout="wide")
DB = "./chroma_db_app"
st.title("Local PDF RAG")

def reset_db():
    try:
        if "vs" in st.session_state:
            st.session_state.vs = None
    except:
        pass

    for _ in range(10):
        try:
            if os.path.exists(DB):
                shutil.rmtree(DB)
            return
        except PermissionError:
            time.sleep(0.15)

with st.sidebar:
    file = st.file_uploader("Upload PDF", type="pdf")
    model = st.selectbox("Model", ["llama3.2", "mistral", "gemma2"])
    if st.button("Process") and file:
        reset_db()
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(file.getvalue())
            path = tmp.name

        loader = PyMuPDFLoader(path)
        pages = loader.load()
        text = "".join([p.page_content for p in pages]).strip()
        if not text:
            st.error("PDF has no extractable text.")
            st.stop()

        splitter = RecursiveCharacterTextSplitter(chunk_size=650, chunk_overlap=80)
        chunks = splitter.split_documents(pages)

        emb = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        st.session_state.vs = Chroma.from_documents(chunks, emb, persist_directory=DB)
        st.success("Ready.")

if "msgs" not in st.session_state:
    st.session_state.msgs = []

for m in st.session_state.msgs:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

q = st.chat_input("Ask about the PDF...")
if q:
    st.session_state.msgs.append({"role": "user", "content": q})
    with st.chat_message("user"):
        st.markdown(q)

    if "vs" not in st.session_state:
        with st.chat_message("assistant"):
            st.markdown("Upload a PDF first.")
    else:
        retriever = st.session_state.vs.as_retriever(search_kwargs={"k": 4})
        docs = retriever.invoke(q)
        if not docs:
            ans = "No answer found in the PDF."
        else:
            context = "\n\n".join([d.page_content for d in docs])
            llm = OllamaLLM(model=model)

            prompt = ChatPromptTemplate.from_template(
                "Use only the context.\n\n{context}\n\nQuestion: {question}\nAnswer:"
            )

            chain = prompt | llm
            out = ""
            box = st.chat_message("assistant")
            slot = box.empty()

            for chunk in chain.stream({"context": context, "question": q}):
                out += chunk
                slot.markdown(out)

            st.session_state.msgs.append({"role": "assistant", "content": out})
