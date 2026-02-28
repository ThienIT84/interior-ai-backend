#!/bin/bash
# Run backend WITHOUT auto-reload to prevent restart during long operations

cd "$(dirname "$0")"

echo "🚀 Starting Backend (Stable Mode - No Auto-Reload)"
echo "=================================================="
echo ""
echo "⚠️  Changes to code will NOT auto-reload"
echo "   Press Ctrl+C to stop, then restart manually"
echo ""

~/miniconda3/envs/interior_ai/bin/python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
