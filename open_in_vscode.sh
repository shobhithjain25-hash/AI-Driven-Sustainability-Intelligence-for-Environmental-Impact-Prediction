#!/usr/bin/env bash
# Open EcoSphere directly inside Visual Studio Code
PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "📂 Opening EcoSphere in Visual Studio Code: $PROJECT_DIR"

if [ -d "/Applications/Visual Studio Code.app" ]; then
  "/Applications/Visual Studio Code.app/Contents/Resources/app/bin/code" "$PROJECT_DIR"
elif which code >/dev/null 2>&1; then
  code "$PROJECT_DIR"
else
  open -a "Visual Studio Code" "$PROJECT_DIR"
fi

echo "✅ EcoSphere opened in VS Code!"
