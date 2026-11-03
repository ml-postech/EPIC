"""MiDaS depth estimation used by the EPIC visual-program verifier."""

import numpy as np
import torch
from PIL import Image


class MidasDepthEstimator:
    """
    Robust MiDaS wrapper: PIL.Image -> 1-channel PIL.Image ('L')
    - Handles both transform APIs: returns dict or tensor
    - Handles input as PIL or NumPy
    """

    def __init__(self, model_type: str = "DPT_Large", device: str | None = None):
        self.device = (
            torch.device(device)
            if device
            else (
                torch.device("cuda")
                if torch.cuda.is_available()
                else torch.device("cpu")
            )
        )
        self.model = (
            torch.hub.load("intel-isl/MiDaS", model_type, trust_repo=True)
            .to(self.device)
            .eval()
        )

        midas_transforms = torch.hub.load(
            "intel-isl/MiDaS", "transforms", trust_repo=True
        )
        if model_type in ("DPT_Large", "DPT_Hybrid"):
            self.transform = midas_transforms.dpt_transform
        else:
            self.transform = midas_transforms.small_transform

    @torch.inference_mode()
    def __call__(
        self, img_pil: Image.Image, invert_for_visual: bool = True
    ) -> Image.Image:
        """
        Returns: 1-channel PIL 'L' image with values in [0,255].
        invert_for_visual=True -> smaller values indicate nearer regions.
        """
        # Keep original size
        orig_w, orig_h = img_pil.size
        img_rgb = img_pil.convert("RGB")

        # ---- Robust transform handling (dict or tensor; PIL or NumPy) ----
        inp_any = None
        try:
            # Most hubs accept PIL directly
            inp_any = self.transform(img_rgb)
        except Exception:
            # Fallback: some transforms expect NumPy
            inp_any = self.transform(np.array(img_rgb))

        if isinstance(inp_any, dict):
            inp = inp_any["image"]
        elif torch.is_tensor(inp_any):
            inp = inp_any
        elif isinstance(inp_any, (list, tuple)) and len(inp_any) > 0:
            # Very rare: choose first tensor-like item
            inp = inp_any[0] if torch.is_tensor(inp_any[0]) else inp_any
            if isinstance(inp, dict):
                inp = inp["image"]
        else:
            raise TypeError(f"Unexpected transform output type: {type(inp_any)}")

        # Ensure shape [1,3,H,W], dtype float32, on device
        if inp.ndim == 3:  # [3,H,W]
            inp = inp.unsqueeze(0)
        elif inp.ndim != 4:  # anything else is invalid for the model
            raise ValueError(
                f"Unexpected input tensor shape from transform: {tuple(inp.shape)}"
            )
        inp = inp.to(self.device, dtype=torch.float32, non_blocking=True)

        # ---- Forward pass ----
        pred = self.model(inp)  # shapes vary: [1,H',W'] or [1,1,H',W']

        # Normalize shapes to [1,1,H',W'] for interpolate
        if pred.ndim == 2:  # [H',W']
            pred = pred.unsqueeze(0).unsqueeze(0)
        elif pred.ndim == 3:  # [1,H',W']
            pred = pred.unsqueeze(1)
        elif pred.ndim == 4:
            if pred.shape[1] != 1:  # sometimes [1,?,H',W'] — take channel 0
                pred = pred[:, :1, ...]
        else:
            raise ValueError(f"Unexpected prediction shape: {tuple(pred.shape)}")

        # Resize to original image size
        pred = torch.nn.functional.interpolate(
            pred, size=(orig_h, orig_w), mode="bicubic", align_corners=False
        )[0, 0]  # -> [H,W]

        # ---- Min-max normalize to 0..255 for viewable 8-bit output ----
        d = pred.float().cpu().numpy()
        d = (d - d.min()) / (d.max() - d.min() + 1e-8)
        if invert_for_visual:
            d = 1.0 - d  # invert MiDaS inverse depth: nearer = smaller
        d_u8 = (d * 255.0).clip(0, 255).astype(np.uint8)

        return Image.fromarray(d_u8, mode="L")
