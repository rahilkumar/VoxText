#!/bin/bash
set -e

APP_DIR="$(cd "$(dirname "$0")" && pwd)"
APPS_DIR="$HOME/.local/share/applications"
DESKTOP_SHORTCUT="$HOME/Desktop/VoxText.desktop"
DESKTOP_FILE="$APPS_DIR/VoxText.desktop"

echo "Setting up VoxText..."

# Make scripts executable automatically
chmod +x "$APP_DIR/run.sh" 2>/dev/null || true
chmod +x "$APP_DIR/install_desktop.sh" 2>/dev/null || true

# Create virtual environment if missing
if [ ! -d "$APP_DIR/.venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv "$APP_DIR/.venv"
fi

# Activate venv
source "$APP_DIR/.venv/bin/activate"

# Upgrade pip
pip install --upgrade pip

# Install dependencies
pip install vosk sounddevice numpy customtkinter

# Create desktop launcher
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

# Mark as trusted where supported
gio set "$DESKTOP_FILE" metadata::trusted true 2>/dev/null || true
gio set "$DESKTOP_SHORTCUT" metadata::trusted true 2>/dev/null || true

echo "VoxText setup complete."
echo "You can now launch it from the Desktop or Applications menu."