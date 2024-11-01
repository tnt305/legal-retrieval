#!/bin/bash

# Pull the latest changes
git pull

# Add all changes
git add .

# Prompt user for a custom commit message
echo "Enter your commit message: "
read commit_message

# Commit changes with the custom message
git commit -m "$commit_message"

echo "And commit branch ?: "
read branch
# Push changes to the main branch
git push -u origin "$branch"
