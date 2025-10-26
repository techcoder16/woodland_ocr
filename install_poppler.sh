#!/bin/bash

# Install poppler-utils for PDF to image conversion
# This script installs poppler on different operating systems

echo "Installing poppler-utils for PDF conversion..."

# Detect operating system
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    # Linux
    if command -v apt-get &> /dev/null; then
        # Ubuntu/Debian
        echo "Installing poppler-utils on Ubuntu/Debian..."
        sudo apt-get update
        sudo apt-get install -y poppler-utils
    elif command -v yum &> /dev/null; then
        # CentOS/RHEL
        echo "Installing poppler-utils on CentOS/RHEL..."
        sudo yum install -y poppler-utils
    elif command -v dnf &> /dev/null; then
        # Fedora
        echo "Installing poppler-utils on Fedora..."
        sudo dnf install -y poppler-utils
    else
        echo "Unsupported Linux distribution. Please install poppler-utils manually."
        exit 1
    fi
elif [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS
    if command -v brew &> /dev/null; then
        echo "Installing poppler-utils on macOS with Homebrew..."
        brew install poppler
    else
        echo "Homebrew not found. Please install Homebrew first or install poppler manually."
        exit 1
    fi
elif [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "cygwin" ]]; then
    # Windows (Git Bash/Cygwin)
    echo "Windows detected. Please install poppler-utils manually:"
    echo "1. Download from: https://blog.alivate.com.au/poppler-windows/"
    echo "2. Extract and add to PATH"
    echo "3. Or use conda: conda install -c conda-forge poppler"
else
    echo "Unsupported operating system: $OSTYPE"
    echo "Please install poppler-utils manually for your system."
    exit 1
fi

echo "Poppler installation completed!"
echo "You can now use PDF to image conversion in the OCR system."
