from dotenv import load_dotenv
load_dotenv()

from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFacePipeline
from transformers import pipeline

file_path = "rainforests.pdf"  
if file_path.endswith(".txt"):
    loader = TextLoader(file_path)
elif file_path.endswith(".pdf"):
    loader = PyPDFLoader(file_path)
else:
    raise ValueError("Unsupported file type. Use txt or pdf.")

documents = loader.load()


splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)
docs = splitter.split_documents(documents)

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


vectorstore = FAISS.from_documents(docs, embeddings)

#smaller but better model
local_llm_pipeline = pipeline(
    "text-generation",
    model="distilgpt2",  # cleaner output
    max_new_tokens=250,
    temperature=0.3,
    repetition_penalty=1.2,
    device=-1  # because only CPU
)
llm = HuggingFacePipeline(pipeline=local_llm_pipeline)


prompt_template = """
Use the following context to answer the question.
If you don't know, say "I don't know".
Give the answer in 4-5 sentences maximum, without repeating facts.

Context:
{context}

Question:
{question}

Answer:
"""

PROMPT = PromptTemplate(
    template=prompt_template,
    input_variables=["context", "question"]
)


qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectorstore.as_retriever(),
    chain_type_kwargs={"prompt": PROMPT}
)

while True:
    query = input("\nAsk a question (or type 'exit'): ")
    if query.lower() == "exit":
        break
    answer = qa_chain.invoke({"query": query})  
    print("\nAnswer:", answer["result"])
