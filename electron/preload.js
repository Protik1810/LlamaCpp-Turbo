const { contextBridge, ipcRenderer } = require('electron');

// Expose safe desktop APIs to renderer
contextBridge.exposeInMainWorld('electronAPI', {
  minimize: () => ipcRenderer.send('window-minimize'),
  maximize: () => ipcRenderer.send('window-maximize'),
  close: () => ipcRenderer.send('window-close'),
  selectFile: (options) => ipcRenderer.invoke('select-file', options),
  selectDirectory: (options) => ipcRenderer.invoke('select-directory', options),
  openExternal: (url) => ipcRenderer.send('open-external', url),
  onMaximizedState: (callback) => ipcRenderer.on('window-maximized-state', (event, isMax) => callback(isMax)),
  backendUrl: 'http://127.0.0.1:8008'
});
