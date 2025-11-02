#!/bin/bash
# Quick start script for BL Handwriting Instruction Extractor

echo "🚀 Starting BL Handwriting Instruction Extractor..."
echo ""
echo "📦 Checking dependencies..."

# Check if poppler is installed
if ! command -v pdftoppm &> /dev/null; then
    echo "❌ poppler-utils not found!"
    echo "Please install it:"
    echo "  Ubuntu/Debian: sudo apt-get install -y poppler-utils"
    echo "  macOS: brew install poppler"
    exit 1
fi

echo "✅ poppler-utils installed"
echo ""
echo "🌐 Starting Streamlit app on http://localhost:8501"
echo ""
streamlit run app.py
