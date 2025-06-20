#!/bin/bash
source .venv/bin/activate
# Install PyTorch (2.7.1 during dev of the project) with CUDA 12.8 support
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
