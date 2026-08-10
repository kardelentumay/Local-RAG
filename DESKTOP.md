# Nivora Desktop

Nivora Desktop packages the React interface, local FastAPI backend, and Foundry
Local runtime integration as a Windows application.

## Development

Install the frontend and backend dependencies first, then run:

```powershell
cd frontend
npm run desktop:dev
```

The Electron process uses an API already listening on `127.0.0.1:8000`. If no
API is running, it starts `backend/.venv/Scripts/local-rag-api.exe` automatically.

## Windows installer

From `frontend` run:

```powershell
npm run desktop:package
```

The command performs three steps:

1. Packages the Python backend and Foundry/WinML native runtime with PyInstaller.
2. Creates a static renderer build with Vite.
3. Produces an NSIS installer with Electron Builder.

The installer is written to `frontend/release/Nivora-Setup-<version>.exe`.

## Local data

The installed application stores uploaded documents and the SQLite database
under Electron's Windows user-data directory. Development mode continues to use
the repository's `backend/data` and `backend/documents` directories.

Models already cached by Foundry Local can be used without internet access. A
clean machine still needs internet for its first model download unless a
separately licensed offline model bundle is provided.
