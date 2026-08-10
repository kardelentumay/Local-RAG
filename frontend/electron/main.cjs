const { app, BrowserWindow, dialog, net, protocol } = require("electron");
const { spawn } = require("node:child_process");
const { execFileSync } = require("node:child_process");
const { randomUUID } = require("node:crypto");
const fs = require("node:fs");
const path = require("node:path");
const { pathToFileURL } = require("node:url");

const API_BASE_URL = "http://127.0.0.1:8000";
const INSTANCE_TOKEN = randomUUID();
let backendProcess = null;

protocol.registerSchemesAsPrivileged([
  {
    scheme: "nivora",
    privileges: {
      standard: true,
      secure: true,
      supportFetchAPI: true,
      corsEnabled: true,
      stream: true,
    },
  },
]);

async function backendHealth() {
  try {
    const response = await fetch(`${API_BASE_URL}/api/health`, {
      signal: AbortSignal.timeout(1200),
    });
    return response.ok ? await response.json() : null;
  } catch {
    return null;
  }
}

async function backendIsReady() {
  const health = await backendHealth();
  return health?.application === "nivora-local-rag"
    && health.instance_token === INSTANCE_TOKEN;
}

function backendExecutable() {
  if (app.isPackaged) {
    return path.join(process.resourcesPath, "backend", "local-rag-api.exe");
  }
  return path.resolve(__dirname, "..", "..", "backend", ".venv", "Scripts", "local-rag-api.exe");
}

async function startBackend() {
  const existing = await backendHealth();
  if (existing?.application === "nivora-local-rag" && existing.instance_token === INSTANCE_TOKEN) {
    return;
  }
  if (existing?.application === "nivora-local-rag" && existing.instance_token && existing.pid) {
    execFileSync("taskkill.exe", ["/PID", String(existing.pid), "/T", "/F"], {
      windowsHide: true,
      stdio: "ignore",
    });
    for (let attempt = 0; attempt < 20 && await backendHealth(); attempt += 1) {
      await new Promise((resolve) => setTimeout(resolve, 100));
    }
  } else if (existing) {
    throw new Error("Port 8000 is being used by another or outdated backend. Close it and reopen Nivora.");
  }

  const executable = backendExecutable();
  if (!fs.existsSync(executable)) {
    throw new Error(`Local RAG backend was not found at ${executable}`);
  }

  const backendRoot = app.isPackaged
    ? path.join(app.getPath("userData"), "backend")
    : path.resolve(__dirname, "..", "..", "backend");
  const dataRoot = path.join(backendRoot, "data");
  const documentsRoot = path.join(backendRoot, "documents");
  fs.mkdirSync(dataRoot, { recursive: true });
  fs.mkdirSync(documentsRoot, { recursive: true });

  backendProcess = spawn(executable, [], {
    cwd: backendRoot,
    env: {
      ...process.env,
      LOCAL_RAG_DB: path.join(dataRoot, "knowledge.db"),
      LOCAL_RAG_DOCUMENTS: documentsRoot,
      LOCAL_RAG_INSTANCE_TOKEN: INSTANCE_TOKEN,
    },
    windowsHide: true,
    stdio: "ignore",
  });

  for (let attempt = 0; attempt < 60; attempt += 1) {
    if (await backendIsReady()) return;
    if (backendProcess.exitCode !== null) break;
    await new Promise((resolve) => setTimeout(resolve, 250));
  }
  throw new Error("The local RAG backend did not become ready.");
}

function registerDesktopProtocol() {
  const rendererRoot = path.resolve(__dirname, "..", "dist-desktop");
  protocol.handle("nivora", (request) => {
    const requestUrl = new URL(request.url);
    const relativePath = decodeURIComponent(requestUrl.pathname).replace(/^\/+/, "") || "desktop.html";
    const filePath = path.resolve(rendererRoot, relativePath);
    if (filePath !== rendererRoot && !filePath.startsWith(`${rendererRoot}${path.sep}`)) {
      return new Response("Invalid path", { status: 400 });
    }
    return net.fetch(pathToFileURL(filePath).toString());
  });
}

function createWindow() {
  const window = new BrowserWindow({
    width: 1440,
    height: 920,
    minWidth: 1050,
    minHeight: 680,
    backgroundColor: "#f2f2f2",
    autoHideMenuBar: true,
    show: true,
    webPreferences: {
      contextIsolation: true,
      nodeIntegration: false,
      sandbox: true,
    },
  });
  window.webContents.on("did-fail-load", (_event, code, description) => {
    dialog.showErrorBox("Nivora interface could not load", `${code}: ${description}`);
  });
  window.loadURL("nivora://app/desktop.html").catch((error) => {
    dialog.showErrorBox("Nivora interface could not load", String(error.message || error));
  });
}

const hasSingleInstanceLock = app.requestSingleInstanceLock();
if (!hasSingleInstanceLock) {
  app.quit();
}

app.on("second-instance", () => {
  const window = BrowserWindow.getAllWindows()[0];
  if (window) {
    if (window.isMinimized()) window.restore();
    window.focus();
  }
});

app.whenReady().then(async () => {
  if (!hasSingleInstanceLock) return;
  registerDesktopProtocol();
  try {
    await startBackend();
    createWindow();
  } catch (error) {
    dialog.showErrorBox("Nivora could not start", String(error.message || error));
    app.quit();
  }
});

app.on("window-all-closed", () => {
  if (process.platform !== "darwin") app.quit();
});

app.on("before-quit", () => {
  if (backendProcess && backendProcess.exitCode === null) backendProcess.kill();
});
