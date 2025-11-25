import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate

pdf = input("PDF file: ").strip()
if not os.path.exists(pdf):
    raise FileNotFoundError(pdf)

loader = PyPDFLoader(pdf)
pages = loader.load()

splitter = RecursiveCharacterTextSplitter(chunk_size=650, chunk_overlap=80)
chunks = splitter.split_documents(pages)

emb = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vs = Chroma.from_documents(chunks, emb, persist_directory="./chroma_db")

retriever = vs.as_retriever(search_kwargs={"k": 4})
llm = OllamaLLM(model="llama3.2")

prompt = PromptTemplate(
    template="Use only this:\n\n{context}\n\nQuestion: {question}\nAnswer:",
    input_variables=["context", "question"]
)

while True:
    q = input("\nAsk: ").strip()
    if q.lower() in ["exit", "quit"]:
        break
    docs = retriever.invoke(q)
    if not docs:
        print("No answer found.")
        continue
    ctx = "\n\n".join([d.page_content for d in docs])
    final = prompt.format(context=ctx, question=q)
    print("\n", llm.invoke(final))
