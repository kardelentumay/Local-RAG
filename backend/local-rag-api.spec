from PyInstaller.utils.hooks import collect_all


foundry_datas = []
foundry_binaries = []
foundry_hiddenimports = []
for package in (
    "foundry_local_sdk",
    "foundry_local_core_winml",
    "onnxruntime_core",
    "onnxruntime_genai_core",
):
    datas, binaries, hiddenimports = collect_all(package)
    foundry_datas += datas
    foundry_binaries += binaries
    foundry_hiddenimports += hiddenimports

a = Analysis(
    ["desktop_entry.py"],
    pathex=["src"],
    binaries=foundry_binaries,
    datas=foundry_datas,
    hiddenimports=foundry_hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name="local-rag-api",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
)
