# Start Here - Nivora Local RAG

Nivora is an offline-first document assistant. It indexes PDF, Markdown, TXT and Excel files locally, searches them with hybrid semantic and keyword retrieval, and generates source-grounded answers with a local model.

## Project structure

```text
Local RAG/
├── backend/   FastAPI API, indexing, SQLite database and document processing
├── frontend/  React/Vinext interface and Electron desktop shell
└── README.md  Detailed architecture and command reference
```

## Requirements

- Windows 10/11 recommended
- Python 3.11+
- Node.js 22.13+
- Internet only for the first dependency/model/data download

After models and documents are available locally, Nivora can run without internet access.

## 1. Install and start the backend

Open PowerShell in the project root:

```powershell
cd backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e .
```

Start the local API:

```powershell
.\.venv\Scripts\local-rag-api.exe
```

The API runs at `http://127.0.0.1:8000`.

Check that it is ready:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/api/health
Invoke-RestMethod http://127.0.0.1:8000/api/status
```

## 2. Start the web interface

Open a second PowerShell window:

```powershell
cd frontend
npm install
npm run dev
```

Open `http://localhost:3000` in a browser. The frontend connects to the local API automatically.

## 3. Add documents

You can upload documents from the Nivora interface. Supported formats:

- PDF
- Markdown and TXT
- Excel `.xlsx`

Documents can be assigned to categories, selected for search, inspected through sources, and deleted from the interface.

For the original RAG research collection, use the CLI:

```powershell
cd backend
.\.venv\Scripts\Activate.ps1
local-rag download-hf --topic rag --limit 20
local-rag ingest documents
```

The Hugging Face download requires internet. Ingested files and the SQLite index remain local.

## 4. Test a question from the CLI

```powershell
local-rag status
local-rag search "What is corrective RAG?" --top-k 3
local-rag ask "What is corrective RAG?" --top-k 2
local-rag --chat-model qwen2.5-1.5b chat --top-k 2
```

Available local chat models include:

- `qwen2.5-1.5b`: better quality, higher memory use
- `qwen2.5-0.5b`: faster and lighter, lower answer quality

The Nivora UI and API default to `qwen2.5-1.5b`.

## 5. Run the desktop application

For development:

```powershell
cd frontend
npm install
npm run desktop:dev
```

To create a Windows installer:

```powershell
npm run desktop:package
```

The installer is generated under `frontend/release/`. Electron starts the packaged FastAPI executable as a local sidecar process. The application ID is `com.nivora.localrag` and the product name is `Nivora`.

## 6. Run tests

Backend:

```powershell
cd backend
$env:PYTHONPATH="src"
python -m unittest discover -s tests -v
```

Frontend:

```powershell
cd frontend
npm test
npm run lint
```

## Important local paths

- SQLite index: `backend/data/knowledge.db`
- User documents: `backend/documents/`
- Hugging Face papers: `backend/documents/huggingface/`
- Desktop backend build: `backend/dist/local-rag-api.exe`
- Windows installer: `frontend/release/`

Do not commit `.venv`, `node_modules`, local databases, uploaded private documents, model caches or generated release artifacts. The root `.gitignore` contains the relevant rules.

## Troubleshooting

- If port 8000 is busy, stop the old `local-rag-api` process before starting the backend.
- If the browser shows old behavior, restart the backend and refresh the frontend with `Ctrl + F5`.
- The first question can be slower because the local model is loaded lazily; later questions are usually faster.
- If the virtual environment was moved from another folder, recreate `.venv` in the current `backend` directory.
