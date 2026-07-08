import torch
import torch.nn as nn
import os
import json


class BinaryMlp(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.input_dim = input_dim

        self.net = nn.Sequential(
            nn.Linear(self.input_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, x):
        return self.net(x)
    
    def save(self, save_path, meta=None):
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)

        save_dict = {
            "state_dict": self.state_dict(),
            "input_dim": self.input_dim,
        }

        torch.save(save_dict, save_path)

        if meta is not None:
            meta_path = os.path.splitext(save_path)[0] + ".meta.json"

            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(meta, f, ensure_ascii=False, indent=2)

        print(f">>> Best model saved to: {save_path}")

    @classmethod
    def load(cls, load_path, map_location="cpu"):
        checkpoint = torch.load(load_path, map_location=map_location)

        model = cls(
            input_dim=checkpoint["input_dim"]
        )

        model.load_state_dict(checkpoint["state_dict"])
        model.to(map_location)

        print(f">>> Model loaded from: {load_path} (device: {map_location})")

        return model
