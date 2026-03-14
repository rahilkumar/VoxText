#!/bin/bash
set -e

APP_DIR="$(cd "$(dirname "$0")" && pwd)"
APPS_DIR="$HOME/.local/share/applications"
DESKTOP_SHORTCUT="$HOME/Desktop/VoxText.desktop"
DESKTOP_FILE="$APPS_DIR/VoxText.desktop"

echo "=============================="
echo "Installing VoxText..."
echo "=============================="

# Fix broken apt packages if any
echo "Checking for broken packages..."
sudo apt --fix-broken install -y || true

echo "Installing system dependencies..."
sudo apt update

sudo apt install -y \
python3 \
python3-venv \
python3-pip \
libportaudio2 \
python3-tk \
git

echo "Setting permissions..."
chmod +x "$APP_DIR/run.sh" 2>/dev/null || true

# Create venv
if [ ! -d "$APP_DIR/.venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv "$APP_DIR/.venv"
fi

echo "Activating environment..."
source "$APP_DIR/.venv/bin/activate"

echo "Upgrading pip..."
pip install --upgrade pip

echo "Installing Python packages..."
pip install \
vosk \
sounddevice \
numpy \
customtkinter

echo "Creating desktop launcher..."
mkdir -p "$APPS_DIR"

cat > "$DESKTOP_FILE" <<EOF
[Desktop Entry]
Version=1.0
Type=Application
Name=VoxText
Comment=Offline Speech To Text
Path=$APP_DIR
Exec=/bin/bash $APP_DIR/run.sh
Icon=audio-input-microphone
Terminal=false
Categories=Utility;
EOF

chmod +x "$DESKTOP_FILE"

# Copy to desktop
cp "$DESKTOP_FILE" "$DESKTOP_SHORTCUT" 2>/dev/null || true
chmod +x "$DESKTOP_SHORTCUT" 2>/dev/null || true

# Trust launcher (Raspberry Pi OS / GNOME)
gio set "$DESKTOP_FILE" metadata::trusted true 2>/dev/null || true
gio set "$DESKTOP_SHORTCUT" metadata::trusted true 2>/dev/null || true

echo ""
echo "=============================="
echo "VoxText installation complete!"
echo "=============================="
echo ""
echo "You can now launch VoxText from:"
echo "Desktop → VoxText"
echo ""
