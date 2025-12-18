# QA with Documents (Streamlit + LlamaIndex)

Retrieval-augmented QA app that lets you upload a document (PDF/TXT), indexes it with LlamaIndex + OpenAI embeddings, and answers questions via an OpenAI LLM. Frontend is Streamlit.

## Quickstart

```bash
# 1) Create and activate venv (example)
python3 -m venv venv
source venv/bin/activate

# 2) Install deps
pip install -r requirements.txt

# 3) Set your OpenAI key
export OPENAI_API_KEY="sk-..."
# or create a .env file with OPENAI_API_KEY=...

# 4) Run the app
streamlit run StreamlitApp.py
```

## How it works
- Upload a document in the UI; it is saved into `Data/`.
- `load_data` reads the file(s) and produces documents.
- `create_or_load_index` builds (or reloads) a vector index using OpenAI embeddings and persists it to `storage/`.
- Queries are sent to the index; responses come from the OpenAI LLM.

## Project layout
- `StreamlitApp.py` – Streamlit UI and request flow
- `QAWithPDF/data_ingestion.py` – loads uploaded files into LlamaIndex documents
- `QAWithPDF/embedding.py` – builds/loads the vector index and sets embedding/LLM settings
- `QAWithPDF/model_api.py` – loads the OpenAI LLM
- `QAWithPDF/logger.py` – basic logging configuration
- `requirements.txt` – full dependency list
- `storage/` – persisted index artifacts (created at runtime)
- `Data/` – uploaded documents are written here

## Configuration
- `OPENAI_API_KEY` must be set in your environment or `.env`.
- Adjust embedding/LLM models or chunking in `QAWithPDF/embedding.py` if needed.

## Troubleshooting
- If imports fail, ensure you run from the project root so `QAWithPDF` is on `PYTHONPATH`.
- Delete `storage/` to rebuild the index from fresh documents if you change embeddings or data.
