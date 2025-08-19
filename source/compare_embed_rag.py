from dotenv import load_dotenv
load_dotenv()

from datetime import datetime
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from transformers import pipeline
from langchain_community.llms import HuggingFacePipeline
from langchain.prompts import PromptTemplate
import re


FILE_PATH = "rainforests.pdf"        
CHUNK_OVERLAP = 50
TOP_K = 3                            
LLM_MODEL = "EleutherAI/gpt-neo-125M"
GEN_MAX_TOKENS = 180
GEN_TEMP = 0.3
GEN_REP_PENALTY = 1.2
NO_REPEAT_NGRAM = 3

if FILE_PATH.endswith(".txt"):
    loader = TextLoader(FILE_PATH)
elif FILE_PATH.endswith(".pdf"):
    loader = PyPDFLoader(FILE_PATH)
else:
    raise ValueError("Unsupported file type. Use .txt or .pdf")

documents = loader.load()
print(f"[info] Loaded {len(documents)} document(s) from {FILE_PATH}")


def split_documents(chunk_size):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=CHUNK_OVERLAP)
    docs = splitter.split_documents(documents)
    for i, d in enumerate(docs):
        d.metadata.update({
            "source": FILE_PATH,
            "chunk_id": i,
            "chunk_size": chunk_size,
            "created": datetime.now().strftime("%Y-%m-%d")
        })
    return docs

# same chunking for both embeddings 
chunked_docs = split_documents(chunk_size=200)

#2 FAISS indexes
print("[info] Building index A with all-MiniLM-L6-v2")
embA = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
faiss_A = FAISS.from_documents(chunked_docs, embA)

print("[info] Building index B with multi-qa-MiniLM-L6-cos-v1")
embB = HuggingFaceEmbeddings(model_name="sentence-transformers/multi-qa-MiniLM-L6-cos-v1")
faiss_B = FAISS.from_documents(chunked_docs, embB)

#LLM pipeline
print(f"[info] Loading local LLM: {LLM_MODEL}")
llm_pipe = pipeline(
    "text-generation",
    model=LLM_MODEL,
    max_new_tokens=GEN_MAX_TOKENS,
    temperature=GEN_TEMP,
    repetition_penalty=GEN_REP_PENALTY,
    no_repeat_ngram_size=NO_REPEAT_NGRAM,
    do_sample=True,
    device=-1
)
llm = HuggingFacePipeline(pipeline=llm_pipe)

# 5) Prompt template
prompt_template = """Use the context below to answer the question concisely.
If the answer is not in the context, say "I don't know".

Context:
{context}

Question:
{question}

Answer:"""
PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

# ------ Output cleanup helpers ------
def remove_prompt_echo(generated_text: str, prompt_text: str) -> str:
    if generated_text.startswith(prompt_text):
        return generated_text[len(prompt_text):].lstrip()
    return generated_text

def collapse_consecutive_duplicates(text: str) -> str:
    if not text:
        return ""
    lines = text.splitlines()
    new_lines = []
    prev = None
    for line in lines:
        sline = line.strip()
        if sline != prev:
            new_lines.append(line)
        prev = sline
    cleaned = "\n".join(new_lines).strip()
    sents = re.split(r'(?<=[.!?])\s+', cleaned)
    final_sents, prev_sent = [], None
    for s in sents:
        if s != prev_sent:
            final_sents.append(s)
        prev_sent = s
    return " ".join(final_sents).strip()

def generate_with_fallback(prompt_text: str) -> str:
    try:
        out = llm.invoke(prompt_text)
        if hasattr(out, "generations"):  # LLMResult case
            candidate = out.generations[0][0].text
        elif isinstance(out, dict):
            candidate = out.get("result") or out.get("text") or ""
        elif isinstance(out, str):
            candidate = out
        else:
            candidate = str(out)
    except Exception:
        res = llm_pipe(prompt_text, return_full_text=False)
        candidate = res[0]["generated_text"] if res else ""
    candidate = remove_prompt_echo(candidate, prompt_text)
    candidate = collapse_consecutive_duplicates(candidate)
    return candidate.strip()

#Retrieval + comp
def run_query_and_compare(query):
    print(f"\n\n=== QUERY ===\n{query}\n")
    hits_A = faiss_A.similarity_search(query, k=TOP_K)
    print("--- Top retrievals (A) ---")
    for h in hits_A:
        print(f"[A] chunk_id={h.metadata['chunk_id']}")
        print(h.page_content.strip()[:200])
    contextA = "\n\n".join(h.page_content for h in hits_A)
    promptA = PROMPT.format(context=contextA, question=query)
    answerA = generate_with_fallback(promptA)

    hits_B = faiss_B.similarity_search(query, k=TOP_K)
    print("\n--- Top retrievals (B) ---")
    for h in hits_B:
        print(f"[B] chunk_id={h.metadata['chunk_id']}")
        print(h.page_content.strip()[:200])
    contextB = "\n\n".join(h.page_content for h in hits_B)
    promptB = PROMPT.format(context=contextB, question=query)
    answerB = generate_with_fallback(promptB)

    print("\n=== FINAL ANSWERS ===")
    print("[A] all-MiniLM-L6-v2:\n", answerA)
    print("\n[B] multi-qa-MiniLM-L6-cos-v1:\n", answerB)

if __name__ == "__main__":
    print("\nReady. Enter queries to compare embedding models (type 'exit' to quit).")
    while True:
        q = input("\nQuery: ").strip()
        if q.lower() == "exit":
            break
        run_query_and_compare(q)
