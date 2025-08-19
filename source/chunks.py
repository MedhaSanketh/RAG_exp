from dotenv import load_dotenv
load_dotenv()

from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from datetime import datetime

file_path = "rainforests.pdf"  

if file_path.endswith(".txt"):
    loader = TextLoader(file_path)
elif file_path.endswith(".pdf"):
    loader = PyPDFLoader(file_path)
else:
    raise ValueError("Unsupported file type. Use txt or pdf.")

documents = loader.load()

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


def split_and_store(chunk_size):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=50
    )
    docs = splitter.split_documents(documents)

    for i, doc in enumerate(docs):
        doc.metadata = {
            "source": file_path,
            "chunk_id": i,
            "timestamp": datetime.now().strftime("%Y-%m-%d"),
            "chunk_size": chunk_size
        }
    return FAISS.from_documents(docs, embeddings)


faiss_200 = split_and_store(200)
faiss_500 = split_and_store(500)


def compare_results(query):
    print(f"\n Query: {query}")
    
    res_200 = faiss_200.similarity_search(query, k=2)
    print("\n Results with chunk size = 200 ")
    for r in res_200:
        print(f"[Metadata: {r.metadata}]")
        print(r.page_content, "\n")
    
    res_500 = faiss_500.similarity_search(query, k=2)
    print("\n Results with chunk size = 500 ")
    for r in res_500:
        print(f"[Metadata: {r.metadata}]")
        print(r.page_content, "\n")


while True:
    query = input("\nAsk a question to compare (or type 'exit'): ")
    if query.lower() == "exit":
        break
    compare_results(query)
