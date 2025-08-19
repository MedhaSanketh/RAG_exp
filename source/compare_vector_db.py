from dotenv import load_dotenv
load_dotenv()

from datetime import datetime
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS, Chroma

# from langchain_community.vectorstores import Weaviate, Pinecone

from transformers import pipeline
from langchain_community.llms import HuggingFacePipeline
from langchain.prompts import PromptTemplate
import os


FILE_PATH = "rainforests.pdf"
CHUNK_OVERLAP = 50
CHUNK_SIZE = 200
TOP_K = 3
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "google/flan-t5-large"  # fast, good at concise QA



if FILE_PATH.endswith(".pdf"):
    loader = PyPDFLoader(FILE_PATH)
else:
    raise ValueError("Only PDF supported in this demo")

documents = loader.load()

splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
docs = splitter.split_documents(documents)
for i, d in enumerate(docs):
    d.metadata = {
        "source": FILE_PATH,
        "chunk_id": i,
        "created": datetime.now().strftime("%Y-%m-%d")
    }

# 2) Embeddings (same for all DBs)
embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)

# 3) Build vector DBs
print("[info] Building FAISS index...")
faiss_store = FAISS.from_documents(docs, embeddings)

print("[info] Building Chroma index...")
chroma_store = Chroma.from_documents(
    docs,
    embeddings,
    collection_name="rainforest_test",
    persist_directory="./chroma_test"
)
chroma_store.persist()

# Optionally:
# print("[info] Building Weaviate index...")
# weaviate_store = Weaviate.from_documents(...)


print(f"[info] Loading LLM: {LLM_MODEL}")
llm_pipe = pipeline(
    "text2text-generation",  # suited for instruction-style models
    model=LLM_MODEL,
    max_new_tokens=200,
    temperature=0.3,
    device=-1
)
llm = HuggingFacePipeline(pipeline=llm_pipe)

prompt_template = """You are a helpful expert on rainforests.

Answer the question based ONLY on the context below.
If you cannot find the answer, say "I don't know".

Context:
{context}

Question:
{question}

Answer in a short, clear numbered list without repeating items.
"""
PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

def build_prompt_from_docs(docs, question):
    blocks = []
    for d in docs:
        tag = f"[src={d.metadata.get('source')}|chunk={d.metadata.get('chunk_id')}]"
        blocks.append(f"{tag}\n{d.page_content.strip()}")
    context = "\n\n".join(blocks)
    return PROMPT.format(context=context, question=question)
def run_query_and_compare(query):
    print(f"\n=== QUERY ===\n{query}")

    dbs = {
        "FAISS": faiss_store,
        "Chroma": chroma_store,
        # "Weaviate": weaviate_store
    }

    for name, store in dbs.items():
        print(f"\n--- {name} Retrieval ---")
        hits = store.similarity_search(query, k=TOP_K)
        for i, h in enumerate(hits, 1):
            snippet = h.page_content.replace("\n", " ")[:120]
            print(f"[{name}-{i}] chunk_id={h.metadata['chunk_id']} | {snippet}...")

        
        clean_context = "\n".join(
            [f"{i+1}. {h.page_content.strip()}" for i, h in enumerate(hits)]
        )

        prompt = f"""You are a helpful expert on rainforests.

Answer the question based ONLY on the context below.
If you cannot find the answer, say "I don't know".

Context:
{clean_context}

Question:
{query}

Answer in a short, clear numbered list without repeating items.
"""

        answer = llm_pipe(prompt)[0]['generated_text']
        print(f"\n[{name} Answer]\n{answer}")



if __name__ == "__main__":
    while True:
        q = input("\nQuery (or 'exit'): ").strip()
        if not q or q.lower() == "exit":
            break
        run_query_and_compare(q)
