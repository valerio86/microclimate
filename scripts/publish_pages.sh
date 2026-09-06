#!/bin/bash
# Publish data/dashboard.html to the gh-pages branch (served at
# https://valerio86.github.io/microclimate/) via a dedicated worktree at
# .gh-pages-worktree/, so it doesn't disturb the main checkout.
#
# Run after `microclimate refresh` has rebuilt data/dashboard.html. See
# refresh_and_publish.sh for the combined daily job.
set -euo pipefail
cd "$(dirname "$0")/.."

if [ ! -d .gh-pages-worktree ]; then
    git worktree add .gh-pages-worktree gh-pages
fi

if [ ! -f data/dashboard.html ]; then
    echo "publish_pages: data/dashboard.html does not exist, nothing to publish" >&2
    exit 1
fi

cp data/dashboard.html .gh-pages-worktree/index.html
cd .gh-pages-worktree

if git diff --quiet -- index.html && [ -z "$(git status --porcelain)" ]; then
    echo "publish_pages: no change since last publish"
    exit 0
fi

git add index.html
git commit -q -m "Publish dashboard $(date -u +%Y-%m-%dT%H:%M:%SZ)"
git push -q origin gh-pages
echo "publish_pages: pushed at $(date -u +%Y-%m-%dT%H:%M:%SZ)"
