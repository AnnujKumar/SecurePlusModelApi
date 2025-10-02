#!/bin/bash

# Install system dependencies from packages.txt
apt-get update && apt-get install -y --no-install-recommends $(cat packages.txt)

# Install Python dependencies from requirements.txt
pip install -r requirements.txt