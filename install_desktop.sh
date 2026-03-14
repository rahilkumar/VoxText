#!/bin/bash
set -e

APP_DIR="$(cd "$(dirname "$0")" && pwd)"
APPS_DIR="$HOME/.local/share/applications"
DESKTOP_SHORTCUT="$HOME/Desktop/VoxText.desktop"
DESKTOP_FILE="$APPS_DIR/VoxText.desktop"

echo "Setting up VoxText..."
echo "App directory: $APP_DIR"

# Install system packages
echo "Installing system packages..."
sudo apt update
sudo apt install -y python3 python3-venv python3-pip portaudio19-dev python3-tk libatlas-base-dev python3-dev

# Fix script permissions
chmod +x "$APP_DIR/run.sh" 2>/dev/null || true
chmod +x "$APP_DIR/install_desktop.sh" 2>/dev/null || true

# Create virtual environment if missing
if [ ! -d "$APP_DIR/.venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv "$APP_DIR/.venv"
fi

# Activate venv
echo "Activating virtual environment..."
source "$APP_DIR/.venv/bin/activate"

# Upgrade pip tools
echo "Upgrading pip..."
python -m pip install --upgrade pip setuptools wheel

# Install Python dependencies
if [ -f "$APP_DIR/requirements.txt" ]; then
    echo "Installing Python packages from requirements.txt..."
    pip install -r "$APP_DIR/requirements.txt"
else
    echo "requirements.txt not found, installing default packages..."
    pip install vosk sounddevice numpy customtkinter
fi

# Ensure run.sh exists
if [ ! -f "$APP_DIR/run.sh" ]; then
    cat > "$APP_DIR/run.sh" <<'EOF'
#!/bin/bash
set -e
cd "$(dirname "$0")"
source .venv/bin/activate
exec python Voxtext_app.py
EOF
    chmod +x "$APP_DIR/run.sh"
fi

# Create applications folder
mkdir -p "$APPS_DIR"

# Create launcher in applications menu
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

# Copy launcher to desktop if desktop exists
if [ -d "$HOME/Desktop" ]; then
    cp "$DESKTOP_FILE" "$DESKTOP_SHORTCUT"
    chmod +x "$DESKTOP_SHORTCUT" 2>/dev/null || true
    gio set "$DESKTOP_SHORTCUT" metadata::trusted true 2>/dev/null || true
fi

# Trust application launcher where supported
gio set "$DESKTOP_FILE" metadata::trusted true 2>/dev/null || true

echo
echo "VoxText setup complete."
echo "Try launching from:"
echo "  1) Desktop -> VoxText.desktop"
echo "  2) Applications menu -> VoxText"
echo "  3) Terminal -> $APP_DIR/run.sh"
