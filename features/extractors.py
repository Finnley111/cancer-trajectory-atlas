"""Feature extraction helpers for Phikon and ResNet backbones."""

import torch
import numpy as np
from tqdm import tqdm
from PIL import Image


def get_model(model_name: str, device: torch.device):
    """Load a model and optional processor."""
    print(f"  Loading {model_name}...")

    if model_name in ("resnet18", "resnet50", "resnet101"):
        from torchvision import models
        from torchvision.models import (
            ResNet18_Weights, ResNet50_Weights, ResNet101_Weights,
        )
        weights_map = {
            "resnet18":  (models.resnet18,  ResNet18_Weights.IMAGENET1K_V1),
            "resnet50":  (models.resnet50,  ResNet50_Weights.IMAGENET1K_V2),
            "resnet101": (models.resnet101, ResNet101_Weights.IMAGENET1K_V2),
        }
        factory, weights = weights_map[model_name]
        model = factory(weights=weights)
        model = torch.nn.Sequential(*list(model.children())[:-1])
        return model.to(device), None

    if model_name in ("phikon", "phikon-v2"):
        from transformers import AutoModel, AutoImageProcessor
        hub_name = f"owkin/{model_name}"
        model = AutoModel.from_pretrained(hub_name, trust_remote_code=True)
        processor = AutoImageProcessor.from_pretrained(hub_name, use_fast=True)
        return model.to(device), processor

    raise ValueError(f"Unsupported model: {model_name}")


def extract_features(
    patches: np.ndarray,
    model_name: str = "phikon",
    batch_size: int = 32,
) -> np.ndarray:
    """
    Extract feature vectors for a batch of patches.

    Args:
        patches: (N, H, W, 3) uint8 array
        model_name: Model identifier
        batch_size: GPU batch size

    Returns:
        features: (N, D) float32 array  (D=768 for Phikon)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Using device: {device}")

    model, processor = get_model(model_name, device)
    model.eval()

    features = []
    with torch.no_grad():
        for i in tqdm(range(0, len(patches), batch_size), desc="Extracting features"):
            batch = patches[i : i + batch_size]

            if processor:  # Transformers (Phikon)
                pil_batch = [Image.fromarray(p) for p in batch]
                inputs = processor(images=pil_batch, return_tensors="pt")
                inputs = {k: v.to(device) for k, v in inputs.items()}
                outputs = model(**inputs)
                batch_feats = outputs.last_hidden_state[:, 0].cpu().numpy()  # CLS token
            else:  # ResNet
                tensors = [
                    torch.from_numpy(p).permute(2, 0, 1).float() / 255.0
                    for p in batch
                ]
                batch_input = torch.stack(tensors).to(device)
                output = model(batch_input)
                batch_feats = output.squeeze(-1).squeeze(-1).cpu().numpy()

            features.append(batch_feats)

    return np.vstack(features)
