#!/bin/bash
set -e

APP_DIR="$(cd "$(dirname "$0")" && pwd)"
DESKTOP_FILE="$APP_DIR/VoxText.desktop"
APPS_DIR="$HOME/.local/share/applications"
DESKTOP_SHORTCUT="$HOME/Desktop/VoxText.desktop"

mkdir -p "$APPS_DIR"

sed "s|__APP_DIR__|$APP_DIR|g" "$APP_DIR/VoxText.desktop.template" > "$DESKTOP_FILE"

chmod +x "$DESKTOP_FILE"

cp "$DESKTOP_FILE" "$APPS_DIR/VoxText.desktop"
cp "$DESKTOP_FILE" "$DESKTOP_SHORTCUT" 2>/dev/null || true

chmod +x "$APPS_DIR/VoxText.desktop"
chmod +x "$DESKTOP_SHORTCUT" 2>/dev/null || true

gio set "$APPS_DIR/VoxText.desktop" metadata::trusted true 2>/dev/null || true
gio set "$DESKTOP_SHORTCUT" metadata::trusted true 2>/dev/null || true

echo "VoxText launcher installed."
echo "You can open it from the Applications menu or Desktop."