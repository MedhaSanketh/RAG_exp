# Rainforest QA 🌳

A mini RAG (Retrieval-Augmented Generation) project where a PDF about **rainforests** is converted into vector embeddings, stored in FAISS/Chroma, and queried via a local LLM.

## Features
- Load and chunk PDF text
- Compare embedding models (`MiniLM`, `multi-qa`)
- Compare vector DBs (`FAISS`, `Chroma`)
- Retrieval-based QA with HuggingFace local models
- Bonus: Ollama prompt variations

## Quickstart
```bash
git clone https://github.com/YOUR_USERNAME/rainforest-qa.git
cd rainforest-qa
pip install -r requirements.txt
cp .env.example .env
python src/qa_simple.py
