const { app, BrowserWindow, ipcMain, dialog, shell } = require('electron');
const path = require('path');
const { spawn } = require('child_process');
const http = require('http');
const fs = require('fs');

// -------------------------------------------------------------
// Ultra-Lite & Low-Resource Chromium Engine Configuration
// -------------------------------------------------------------
app.commandLine.appendSwitch('disable-background-networking');
app.commandLine.appendSwitch('disable-component-update');
app.commandLine.appendSwitch('disable-breakpad');
app.commandLine.appendSwitch('no-default-browser-check');
app.commandLine.appendSwitch('disable-features', 'Translate,CalculateNativeWinOcclusion,SpareRendererForSitePerProcess,MediaRouter');
app.commandLine.appendSwitch('js-flags', '--max-old-space-size=256 --optimize_for_size');

let mainWindow = null;
let pythonProcess = null;
const BACKEND_PORT = 8008;

// Determine backend executable path (supports packaged standalone exe, portable exe, and dev mode)
function getBackendExecutable() {
  const baseAppDir = process.env.PORTABLE_EXECUTABLE_DIR || (app.isPackaged ? path.dirname(app.getPath('exe')) : path.join(__dirname, '..'));

  // 1. Packaged resource directory (electron-builder mode)
  if (process.resourcesPath) {
    const packagedExe = path.join(process.resourcesPath, 'backend', 'server.exe');
    if (fs.existsSync(packagedExe)) {
      return { exe: packagedExe, args: [], cwd: baseAppDir };
    }
  }
  // 2. Relative dist_backend directory (local built binary test)
  const distExe = path.join(__dirname, '..', 'dist_backend', 'server', 'server.exe');
  if (fs.existsSync(distExe)) {
    return { exe: distExe, args: [], cwd: baseAppDir };
  }
  // 3. Virtual environment Python (development mode)
  const venvPython = path.join(__dirname, '..', '.venv', 'Scripts', 'python.exe');
  const serverScript = path.join(__dirname, '..', 'src', 'server.py');
  if (fs.existsSync(venvPython)) {
    return { exe: venvPython, args: [serverScript], cwd: path.join(__dirname, '..') };
  }
  // 4. System python fallback
  return { exe: 'python', args: [serverScript], cwd: path.join(__dirname, '..') };
}

// Spawn Python backend server
function startPythonBackend() {
  const target = getBackendExecutable();
  console.log(`[Electron] Spawning Python Backend: ${target.exe} ${target.args.join(' ')}`);

  // Ensure data folders exist in cwd
  try {
    fs.mkdirSync(path.join(target.cwd, 'data', 'sessions'), { recursive: true });
    fs.mkdirSync(path.join(target.cwd, 'models'), { recursive: true });
  } catch (e) {}

  pythonProcess = spawn(target.exe, target.args, {
    cwd: target.cwd,
    env: { ...process.env, PORT: String(BACKEND_PORT), PYTHONIOENCODING: 'utf-8', PYTHONUTF8: '1' },
    stdio: ['ignore', 'pipe', 'pipe']
  });

  pythonProcess.stdout.on('data', (data) => {
    console.log(`[Python Server] ${data.toString().trim()}`);
  });

  pythonProcess.stderr.on('data', (data) => {
    console.error(`[Python Error] ${data.toString().trim()}`);
  });

  pythonProcess.on('close', (code) => {
    console.log(`[Python Server] Process exited with code ${code}`);
  });
}

// Wait for Python backend to be healthy
function waitForBackend(callback, retries = 30) {
  if (retries <= 0) {
    console.error('[Electron] Failed to connect to Python backend.');
    callback();
    return;
  }

  http.get(`http://127.0.0.1:${BACKEND_PORT}/health`, (res) => {
    if (res.statusCode === 200) {
      console.log('[Electron] Python Backend is ready and healthy!');
      callback();
    } else {
      setTimeout(() => waitForBackend(callback, retries - 1), 200);
    }
  }).on('error', () => {
    setTimeout(() => waitForBackend(callback, retries - 1), 200);
  });
}

function createWindow() {
  const iconPath = path.join(__dirname, '..', 'assets', 'icon.png');

  mainWindow = new BrowserWindow({
    width: 1340,
    height: 880,
    minWidth: 1040,
    minHeight: 680,
    frame: false,
    titleBarStyle: 'hidden',
    icon: iconPath,
    backgroundColor: '#0b0f17',
    show: false, // Prevent white flash and render when ready
    webPreferences: {
      preload: path.join(__dirname, 'preload.js'),
      contextIsolation: true,
      nodeIntegration: false,
      sandbox: false,
      webSecurity: true,
      allowRunningInsecureContent: false,
      backgroundThrottling: true,
      spellcheck: false,
    }
  });

  mainWindow.loadFile(path.join(__dirname, 'renderer', 'index.html'));

  mainWindow.once('ready-to-show', () => {
    mainWindow.show();
  });

  // Security: Prevent unauthorized in-app navigation
  mainWindow.webContents.on('will-navigate', (event, navigationUrl) => {
    const parsedUrl = new URL(navigationUrl);
    if (parsedUrl.protocol !== 'file:') {
      event.preventDefault();
      if (navigationUrl.startsWith('http://') || navigationUrl.startsWith('https://')) {
        shell.openExternal(navigationUrl);
      }
    }
  });

  // Security: Intercept and validate external link popups
  mainWindow.webContents.setWindowOpenHandler(({ url }) => {
    if (url.startsWith('https://') || url.startsWith('http://')) {
      shell.openExternal(url);
    }
    return { action: 'deny' };
  });

  // Security: Block creation of unapproved webviews
  mainWindow.webContents.on('will-attach-webview', (event) => {
    event.preventDefault();
  });

  mainWindow.on('maximize', () => {
    mainWindow.webContents.send('window-maximized-state', true);
  });

  mainWindow.on('unmaximize', () => {
    mainWindow.webContents.send('window-maximized-state', false);
  });

  mainWindow.on('closed', () => {
    mainWindow = null;
  });
}

// App Lifecycle
app.whenReady().then(() => {
  startPythonBackend();
  waitForBackend(() => {
    createWindow();
  });

  app.on('activate', () => {
    if (BrowserWindow.getAllWindows().length === 0) {
      createWindow();
    }
  });
});

// Terminate Python process on quit
function cleanupBackend() {
  if (pythonProcess) {
    console.log('[Electron] Terminating Python backend...');
    try {
      if (process.platform === 'win32') {
        spawn('taskkill', ['/pid', pythonProcess.pid, '/f', '/t']);
      } else {
        pythonProcess.kill('SIGKILL');
      }
    } catch (e) {
      console.error('[Electron] Cleanup error:', e);
    }
    pythonProcess = null;
  }
}

app.on('before-quit', cleanupBackend);
app.on('will-quit', cleanupBackend);
app.on('window-all-closed', () => {
  cleanupBackend();
  if (process.platform !== 'darwin') {
    app.quit();
  }
});

// IPC Handlers
ipcMain.on('window-minimize', () => {
  if (mainWindow) mainWindow.minimize();
});

ipcMain.on('window-maximize', () => {
  if (mainWindow) {
    if (mainWindow.isMaximized()) {
      mainWindow.unmaximize();
    } else {
      mainWindow.maximize();
    }
  }
});

ipcMain.on('window-close', () => {
  if (mainWindow) mainWindow.close();
});

ipcMain.handle('select-file', async (event, options) => {
  try {
    const targetWin = BrowserWindow.fromWebContents(event.sender) || mainWindow;
    
    let defaultPath = options?.defaultPath;
    if (!defaultPath) {
      try {
        const appDir = app.isPackaged ? path.dirname(app.getPath('exe')) : path.join(__dirname, '..');
        const modelsDir = path.join(appDir, 'models');
        if (fs.existsSync(modelsDir)) {
          defaultPath = modelsDir;
        }
      } catch (e) {}
    }

    const filters = options?.filters || [
      { name: 'GGUF / Model Files (*.gguf, *.bin)', extensions: ['gguf', 'bin', 'GGUF', 'BIN'] },
      { name: 'All Files (*.*)', extensions: ['*'] }
    ];

    const result = await dialog.showOpenDialog(targetWin, {
      title: options?.title || 'Select GGUF Model File',
      defaultPath: defaultPath,
      filters: filters,
      properties: ['openFile', 'dontAddToRecent']
    });

    if (result.canceled || !result.filePaths || result.filePaths.length === 0) {
      return null;
    }
    return result.filePaths[0];
  } catch (err) {
    console.error('[Electron] select-file error:', err);
    return null;
  }
});

ipcMain.handle('select-directory', async (event, options) => {
  try {
    const targetWin = BrowserWindow.fromWebContents(event.sender) || mainWindow;
    const result = await dialog.showOpenDialog(targetWin, {
      title: options?.title || 'Select Folder / Drive to Scan for GGUF Models',
      defaultPath: options?.defaultPath,
      properties: ['openDirectory', 'dontAddToRecent']
    });

    if (result.canceled || !result.filePaths || result.filePaths.length === 0) {
      return null;
    }
    return result.filePaths[0];
  } catch (err) {
    console.error('[Electron] select-directory error:', err);
    return null;
  }
});

ipcMain.on('open-external', (event, url) => {
  if (url && (url.startsWith('http://') || url.startsWith('https://'))) {
    shell.openExternal(url);
  }
});
