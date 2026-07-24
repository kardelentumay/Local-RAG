# Local RAG

The project is organized under one main folder:

```text
Local RAG/
├── backend/   Local RAG API, indexing, database, and documents
└── frontend/  Nivora user interface
```

## Backend

Open a PowerShell terminal in `backend`:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e .
local-rag-api
```

The API runs locally at `http://127.0.0.1:8000`.

## Frontend

Open a second PowerShell terminal in `frontend`:

```powershell
npm install
npm run dev
```

The interface runs at `http://localhost:3000`.

The frontend connects to the backend at `http://127.0.0.1:8000` by default.

## Notes

- The existing indexed database and 20 RAG papers are preserved in `backend`.
- Python and Node dependency folders are intentionally not copied. Install them
  in their new locations using the commands above.
- The previous Desktop copies have not been deleted.
