#!/usr/bin/env python
"""
Download CLIP RN50 pretrained model.
This script downloads the CLIP ResNet50 model and saves it to the specified location.
"""

import clip
import torch
import os

# Model name to download
model_name = "RN50"

# Target directory
target_path = "/home/dell/gitrepos/MdaCD/RN50.pt"

print(f"Downloading CLIP {model_name} model...")
print("This may take a few minutes depending on your internet connection.")

try:
    # Load the model (this will download it automatically if not cached)
    # The jit parameter returns a jit model which can be saved
    model, preprocess = clip.load(model_name, device='cpu', jit=True)
    
    print(f"Model downloaded successfully!")
    print(f"Saving to {target_path}...")
    
    # Save the JIT model
    model.save(target_path)
    
    print(f"✓ Model saved successfully to {target_path}")
    print(f"File size: {os.path.getsize(target_path) / (1024*1024):.2f} MB")
    
except Exception as e:
    print(f"✗ Error occurred: {e}")
    print("\nAlternative: You can manually download the model from:")
    print("https://openaipublic.azureedge.net/clip/models/afeb0e10f9e5a86da6080e35cf09123aca3b358a0c3e3b6c78a7b63bc04b6762/RN50.pt")
    print(f"And save it to: {target_path}")
