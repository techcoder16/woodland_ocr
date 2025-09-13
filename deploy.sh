#!/bin/bash
set -e  # Exit on any error

echo "Navigating to project directory..."
cd ./woodland/woodland_ocr || { echo "Directory not found!"; exit 1; }



echo "Pulling latest changes..."
# Check if credentials exist in environment variables
if [[ -n "$GITHUB_USERNAME" && -n "$GITHUB_PASSWORD" ]]; then
    echo "Using credentials from environment variables..."
    git pull https://$GITHUB_USERNAME:$GITHUB_PASSWORD@github.com/$(git config --get remote.origin.url | sed 's/.*github.com[:/]\(.*\)\.git/\1/')
else
    # Handle interactive input if environment variables are not set
    echo "Environment credentials not found. Using interactive mode for git pull..."
    # This allows the default git credential prompt to appear
    git pull origin main




fi






echo "Stopping running container..."
docker-compose down --rmi all --volumes --remove-orphans || true

echo "Starting a new..."
docker-compose up --build -d || true


# echo "Displaying logs..."
# docker compose logs --tail 10 || true
# echo "Deployment complete."