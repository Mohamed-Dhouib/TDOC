import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from .lib import models
from .lib.config import config, update_config


def make_cat_config(cfg_file="experiments/CAT_full.yaml"):
    """
    Build the CAT config exactly like tools/infer.py does,
    but without argparse.
    """
    args = type("Args", (), {})()
    args.cfg = cfg_file
    args.opts = []
    cfg = config.clone()
    update_config(cfg, args)
    return cfg


# def dct_hw_to_dctvol(dct_hw: torch.Tensor, T: int = 20) -> torch.Tensor:
#     """
#     Reproduce CAT's DCTvol logic from Splicing/data/AbstractDataset.py

#     dct_hw: (B, H, W) or (B,1,H,W) float/int, values are *JPEG-like* coeffs
#     return: (B, 21, H, W) float
#     """
#     if dct_hw.dim() == 3:
#         dct_hw = dct_hw.unsqueeze(1)  # (B,1,H,W)
#     B, _, H, W = dct_hw.shape

#     dct_vol = dct_hw.new_zeros((B, T + 1, H, W))  # (B,21,H,W)

#     # channel 0: exactly 0
#     dct_vol[:, 0] = (dct_hw == 0).float().squeeze(1)

#     # 1..T-1: +i or -i
#     for i in range(1, T):
#         pos = (dct_hw == float(i)).float().squeeze(1)
#         neg = (dct_hw == float(-i)).float().squeeze(1)
#         dct_vol[:, i] = pos + neg

#     # T: >= T or <= -T
#     geT = (dct_hw >= float(T)).float().squeeze(1)
#     leT = (dct_hw <= float(-T)).float().squeeze(1)
#     dct_vol[:, T] = geT + leT

#     return dct_vol


def dct_hw_to_dctvol(
    dct_hw: torch.Tensor,
    T: int = 20,
) -> torch.Tensor:
    """
    CAT-Net-style DCT volume encoding.

    Input:
        dct_hw: (B, H, W) or (B, 1, H, W) tensor of DCT coefficients (usually ints).
                These are the *scalar* coefficients per spatial location.

        T: maximum magnitude to consider (paper uses T=20).
           Coefficients are clipped to [-T, T], then abs() is taken.

    Output:
        dct_vol: (B, T+1, H, W) binary volume.
                 For each pixel (i,j):
                   - let v = clamp(|M[i,j]|, 0, T)  (integer)
                   - channel v is 1, all others 0.
    """
    # Ensure shape (B, 1, H, W)
    if dct_hw.dim() == 3:
        dct_hw = dct_hw.unsqueeze(1)
    elif dct_hw.dim() != 4:
        raise ValueError(f"dct_hw must be (B,H,W) or (B,1,H,W), got {dct_hw.shape}")

    B, C, H, W = dct_hw.shape
    if C != 1:
        raise ValueError(f"Expected 1 channel of DCT coefficients, got C={C}")

    # 1) absolute value and clipping to [0, T]
    #    (clip(M) in paper is to [-T, T], abs() -> [0, T])
    abs_clipped = dct_hw.abs().clamp_(0, T)       # (B,1,H,W), same dtype as input

    # 2) convert to integer bin indices 0..T
    bin_idx = abs_clipped.long().squeeze(1)       # (B,H,W), each in [0, T]

    num_bins = T + 1                              # e.g. 21
    # 3) allocate (B, T+1, H, W) and scatter one-hot
    dct_vol = dct_hw.new_zeros((B, num_bins, H, W))

    # need index shape (B,1,H,W) for scatter along channel dim=1
    dct_vol.scatter_(1, bin_idx.unsqueeze(1), 1.0)

    return dct_vol
    
class Classifier(nn.Module):
    """
    Take CAT outputs:
      - seg_logits: (B, 2, h, w)
      - x_to_cls:   (B, 360, h, w)
    and produce image-level logits: (B, 2)

    Logic:
      1) seg_logits -> softmax -> take tampered prob (class 1)
      2) from that map compute:
           - mean
           - max
           - top-k mean   (k = topk_ratio * (h*w))
      3) from x_to_cls compute:
           - global avg pool  -> (B, 360)
           - global max pool  -> (B, 360)
      4) concat: 360 + 360 + 3 = 723
      5) MLP -> (B, 2)
    """
    def __init__(self, feat_ch=360, seg_ch=2, topk_ratio=0.05, hidden_dim=512, num_classes=2):
        super().__init__()
        assert seg_ch == 2, "this classifier is hardcoded for 2 seg channels (bg, tampered)"
        self.feat_ch = feat_ch
        self.seg_ch = seg_ch
        self.topk_ratio = float(topk_ratio)
        

        in_dim = feat_ch * 2 + 3   # 360 avg + 360 max + 3 stats from probs = 723
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, seg_logits: torch.Tensor, x_to_cls: torch.Tensor) -> torch.Tensor:
        """
        seg_logits: (B,2,h,w)
        x_to_cls:   (B,360,h,w)
        return:     (B,2)
        """
        B, C, H, W = seg_logits.shape
        # 1) seg -> prob -> tampered prob
        prob = seg_logits.softmax(dim=1)       # (B,2,H,W)
        p_tamp = prob[:, 1]                    # (B,H,W)

        # 2) stats on p_tamp
        p_mean = p_tamp.mean(dim=(1, 2))       # (B,)
        p_max = p_tamp.amax(dim=(1, 2))        # (B,)

        N = H * W
        k = max(1, int(self.topk_ratio * N))
        flat = p_tamp.flatten(1)               # (B, N)
        topk_vals, _ = flat.topk(k, dim=1)
        p_topk_mean = topk_vals.mean(dim=1)    # (B,)

        logit_stats = torch.stack([p_mean, p_max, p_topk_mean], dim=1)  # (B,3)

        # 3) feature-map cues
        x_avg = F.adaptive_avg_pool2d(x_to_cls, 1).flatten(1)  # (B,360)
        x_max = F.adaptive_max_pool2d(x_to_cls, 1).flatten(1)  # (B,360)

        # 4) concat
        feat = torch.cat([x_avg, x_max, logit_stats], dim=1)   # (B, 723)

        # 5) MLP -> (B,2)
        img_logits = self.mlp(feat)
        return img_logits

class CATWrapper(nn.Module):
    """
    Wrapper that exposes the *exact* CAT forward:
        y = cat(x_cat, qtable)
    but x_cat is built from:
        - RGB image         -> (B,3,H,W) normalized
        - DCT coeff (H,W)   -> converted to (B,21,H,W)
    """
    def __init__(self):
        super().__init__()
        cfg_file=os.path.join(os.path.dirname(os.path.abspath(__file__)), "experiments/CAT_full.yaml")
        self.cfg = make_cat_config(cfg_file)
        # build the real CAT model
        cat_cls = getattr(models, self.cfg.MODEL.NAME)
        self.cat = cat_cls.get_seg_model(self.cfg)   # this will try to init weights; if missing, it just warns
        self.classifier=Classifier()

        weights_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "weights", "CAT_full_v2.pth")
        if os.path.exists(weights_path):
            try:
                weights=torch.load(weights_path,weights_only=False)["state_dict"]
                self.cat.load_state_dict(weights)
                print(f"Loaded CAT weights from {weights_path}")
            except Exception:
                pass

    def forward(self, img_norm: torch.Tensor, dct_hw: torch.Tensor, qt: torch.Tensor):
        """
        img_rgb: (B,3,H,W)  raw in [0,255] or [0,1]
        dct_hw : (B,H,W)    your coefficient grid (one channel)
        qtable : (B,1,8,8) or (B,8,8) or (B,64)

        return: CAT logits (B, num_classes, H, W)
        """


        B, _, H, W = img_norm.shape

        # 2) build DCTvol from your (B,H,W)

        dct_vol= dct_hw_to_dctvol(dct_hw)
        # 3) concat to make CAT input
        x_cat = torch.cat([img_norm, dct_vol], dim=1)  # (B, 3+21, H, W)

        # 5) call the REAL CAT model
        out, x_to_cls = self.cat(x_cat,qt)     # <- this is exactly what network_CAT.forward does

        cls_logits= self.classifier(out, x_to_cls)
        out = F.interpolate(out, size=(H, W), mode='bilinear', align_corners=True)
        
        return out, cls_logits


if __name__ == "__main__":
    # dummy example
    B, H, W = 1, 1536, 1536
    img = torch.rand(B, 3, H, W) * 255.0             # fake RGB
    dct_hw = torch.randint(-256, 256, (B, H, W))        # fake DCT coeffs at block resolution expanded to HxW

    model = CATWrapper()
    model.eval()
    with torch.no_grad():
        logits, cls_logits = model(img, dct_hw)

    print("Output shape:", logits.shape, cls_logits.shape)
