#!/usr/bin/env bash
# ==============================================================================
# ⚡ Llama.cpp Turbo Desktop — One-Line Automated Linux / macOS Installer
# ==============================================================================
# Usage:
#   curl -fsSL https://raw.githubusercontent.com/Protik1810/llamacpp-turbo/main/install.sh | bash
#   or
#   wget -qO- https://raw.githubusercontent.com/Protik1810/llamacpp-turbo/main/install.sh | bash
# ==============================================================================

set -e

CYAN='\033[0;36m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${CYAN}"
echo "============================================================"
echo "  ⚡ Llama.cpp Turbo Desktop — Automated Installer"
echo "  Google TurboQuant™ Accelerated Local AI Engine"
echo "============================================================"
echo -e "${NC}"

INSTALL_DIR="${HOME}/.llamacpp-turbo"
BIN_DIR="${HOME}/.local/bin"

# 1. Dependency Checks
echo -e "[1/4] ${YELLOW}Checking system prerequisites...${NC}"

# Check Git
if ! command -v git &>/dev/null; then
    echo -e "${RED}[ERROR] Git is not installed. Please install git first:${NC}"
    echo "  sudo apt install git  # Debian/Ubuntu"
    echo "  sudo dnf install git  # Fedora"
    echo "  sudo pacman -S git    # Arch Linux"
    exit 1
fi

# Check Python 3
PYTHON_BIN=""
if command -v python3 &>/dev/null; then
    PYTHON_BIN="python3"
elif command -v python &>/dev/null; then
    PYTHON_BIN="python"
else
    echo -e "${RED}[ERROR] Python 3 is not installed. Please install Python 3.10+:${NC}"
    echo "  sudo apt install python3 python3-venv python3-pip"
    exit 1
fi

# Check Node.js and NPM
if ! command -v npm &>/dev/null; then
    echo -e "${RED}[ERROR] Node.js and npm are required. Please install Node.js (v18+):${NC}"
    echo "  curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -"
    echo "  sudo apt install -y nodejs"
    exit 1
fi

echo -e "  ${GREEN}✓ Git, Python ($($PYTHON_BIN --version)), and Node.js ($(node --version)) found!${NC}"

# 2. Clone or Update Repository
echo -e "\n[2/4] ${YELLOW}Downloading Llama.cpp Turbo to ${INSTALL_DIR}...${NC}"
if [ -d "$INSTALL_DIR" ]; then
    echo "  Existing installation detected. Updating to latest version..."
    cd "$INSTALL_DIR"
    git pull origin main || true
else
    git clone https://github.com/Protik1810/llamacpp-turbo.git "$INSTALL_DIR"
    cd "$INSTALL_DIR"
fi

# 3. Setup Virtual Environment and Dependencies
echo -e "\n[3/4] ${YELLOW}Setting up Python & Electron environments...${NC}"
if [ ! -f ".venv/bin/python" ]; then
    "$PYTHON_BIN" -m venv .venv
fi

source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# Install Electron NPM packages
npm install

# 4. Create Terminal Launcher Symlink & Desktop Shortcut
echo -e "\n[4/4] ${YELLOW}Configuring system launcher & shortcuts...${NC}"
mkdir -p "$BIN_DIR"

LAUNCHER_SCRIPT="${BIN_DIR}/llamacpp-turbo"
cat << EOF > "$LAUNCHER_SCRIPT"
#!/usr/bin/env bash
cd "$INSTALL_DIR"
exec ./run.sh "\$@"
EOF
chmod +x "$LAUNCHER_SCRIPT"

# Linux Desktop Entry (.desktop)
if [ "$(uname -s)" = "Linux" ]; then
    DESKTOP_DIR="${HOME}/.local/share/applications"
    mkdir -p "$DESKTOP_DIR"
    cat << EOF > "${DESKTOP_DIR}/llamacpp-turbo.desktop"
[Desktop Entry]
Name=Llama.cpp Turbo Desktop
Comment=Google TurboQuant Accelerated Local LLM Desktop Studio
Exec=${BIN_DIR}/llamacpp-turbo
Icon=${INSTALL_DIR}/assets/icon.png
Terminal=false
Type=Application
Categories=Development;Science;ArtificialIntelligence;
EOF
    chmod +x "${DESKTOP_DIR}/llamacpp-turbo.desktop"
    echo -e "  ${GREEN}✓ Desktop launcher created at ~/.local/share/applications/llamacpp-turbo.desktop${NC}"
fi

echo -e "\n${GREEN}============================================================${NC}"
echo -e "${GREEN}  ⚡ INSTALLATION SUCCESSFUL!${NC}"
echo -e "${GREEN}============================================================${NC}"
echo -e "You can launch the app at any time by running:"
echo -e "  ${CYAN}llamacpp-turbo${NC}"
echo -e "\nOr launch directly from the source directory:"
echo -e "  ${CYAN}cd ~/.llamacpp-turbo && ./run.sh${NC}"
echo -e "\n(Note: Ensure ${BIN_DIR} is in your \$PATH)\n"
