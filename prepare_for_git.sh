#!/bin/bash
# Script to prepare backend for git push

echo "🧹 Cleaning up before git push..."

# 1. Remove Python cache
echo "Removing __pycache__ and .pyc files..."
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null
find . -type f -name "*.pyc" -delete
find . -type f -name "*.pyo" -delete

# 2. Check .env is not being committed
if [ -f ".env" ]; then
    echo "⚠️  WARNING: .env file exists. Make sure it's in .gitignore!"
fi

# 3. Verify .gitkeep files exist
echo "Checking .gitkeep files..."
for dir in data/inputs data/outputs data/masks data/temp weights; do
    if [ ! -f "$dir/.gitkeep" ]; then
        echo "Creating $dir/.gitkeep"
        touch "$dir/.gitkeep"
    fi
done

# 4. Show what will be committed
echo ""
echo "📊 Files to be committed:"
git status --short

echo ""
echo "✅ Cleanup complete! Ready to commit."
echo ""
echo "Next steps:"
echo "  1. Review files above"
echo "  2. git add ."
echo "  3. git commit -m 'feat: implement SAM segmentation and SD inpainting backend'"
echo "  4. git push origin main"
