#!/bin/bash

# ─────────────────────────────────────────────
# push.sh — Stage, commit, and push all changes
# Usage:  bash push.sh "your commit message"
#         bash push.sh          (uses a default timestamped message)
# ─────────────────────────────────────────────

set -e

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_DIR"

# ── Commit message: use first argument or auto-generate ──
if [ -n "$1" ]; then
    COMMIT_MSG="$1"
else
    COMMIT_MSG="Update: $(date '+%Y-%m-%d %H:%M:%S')"
fi

echo "──────────────────────────────────────"
echo "  Repository : $REPO_DIR"
echo "  Remote     : $(git remote get-url origin 2>/dev/null || echo 'No remote set')"
echo "  Branch     : $(git branch --show-current)"
echo "  Message    : $COMMIT_MSG"
echo "──────────────────────────────────────"

# ── Check for anything to commit ──
if git diff --quiet && git diff --cached --quiet && [ -z "$(git ls-files --others --exclude-standard)" ]; then
    echo "✓ Nothing to commit — working tree is clean."
    exit 0
fi

# ── Stage all changes ──
git add .

# ── Show what is being committed ──
echo ""
echo "Staged changes:"
git status --short
echo ""

# ── Commit ──
git commit -m "$COMMIT_MSG"

# ── Push ──
git push origin "$(git branch --show-current)"

echo ""
echo "✓ Pushed successfully to origin/$(git branch --show-current)"
