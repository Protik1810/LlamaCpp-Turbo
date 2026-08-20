; Inno Setup Script for Llama.cpp Turbo Desktop
; Builds a complete Windows Setup installer from dist_app\win-unpacked

#define MyAppName "Llama.cpp Turbo Desktop"
#define MyAppVersion "1.0.0"
#define MyAppPublisher "Agentic Systems"
#define MyAppExeName "LlamaCppTurboDesktop.exe"

[Setup]
AppId={{C8A69F21-E8D5-4F8A-9A24-B1F7432890ED}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher={#MyAppPublisher}
DefaultDirName={autopf}\LlamaCppTurboDesktop
DefaultGroupName={#MyAppName}
AllowNoIcons=yes
OutputDir=dist_app
OutputBaseFilename=LlamaCppTurboDesktop-v1.0-Setup
SetupIconFile=assets\icon.ico
Compression=lzma2/ultra64
SolidCompression=yes
WizardStyle=modern
ArchitecturesInstallIn64BitMode=x64compatible
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog
WizardImageFile=assets\wizard_large.bmp
WizardSmallImageFile=assets\wizard_small.bmp
UninstallDisplayIcon={app}\{#MyAppExeName}
DisableDirPage=no
DisableProgramGroupPage=no

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked

[Files]
Source: "dist_app\win-unpacked\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs
Source: "assets\*"; DestDir: "{app}\assets"; Flags: ignoreversion recursesubdirs createallsubdirs onlyifdoesntexist; Permissions: users-full

[Dirs]
Name: "{app}\models"; Permissions: users-full
Name: "{app}\data"; Permissions: users-full
Name: "{app}\data\sessions"; Permissions: users-full
Name: "{app}\assets"; Permissions: users-full

[Icons]
Name: "{group}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"
Name: "{group}\Uninstall {#MyAppName}"; Filename: "{uninstallexe}"
Name: "{autodesktop}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; Tasks: desktopicon

[Run]
Filename: "{app}\{#MyAppExeName}"; Description: "{cm:LaunchProgram,{#StringChange(MyAppName, '&', '&&')}}"; Flags: nowait postinstall skipifsilent
