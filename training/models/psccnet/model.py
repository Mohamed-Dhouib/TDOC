import torch
import torch.nn as nn

from .models.seg_hrnet import get_seg_model
from .models.seg_hrnet_config import get_hrnet_cfg
from .models.NLCDetection import NLCDetection
from .models.detection_head import DetectionHead
from .utils.config import get_pscc_args
import os


class PSCCNetCombined(nn.Module):
    """
    Faithful implementation of the official PSCC-Net architecture:
      - HRNet backbone (feature extractor)
      - NLCDetection head (segmentation)
      - DetectionHead (classification)
    Returns: seg_map, cls_logits
    """
    def __init__(self):
        super().__init__()
        # Backbone
        args = get_pscc_args()
        hrnet_cfg = get_hrnet_cfg()
        self.backbone = get_seg_model(hrnet_cfg)
        # Heads
        self.loc_head = NLCDetection(args)
        self.cls_head = DetectionHead(args)

        # Minimal checkpoint loading: try to load three expected files if they exist.
        base_dir = os.path.dirname(__file__)
        ckpt_root = os.path.join(base_dir, 'checkpoint')

        # Minimal: construct expected paths and load if files exist (no complex handling)
        hrnet_ckpt = os.path.join(ckpt_root, 'HRNet_checkpoint', 'HRNet.pth')
        if os.path.exists(hrnet_ckpt):
            try:
                state = torch.load(hrnet_ckpt, map_location='cpu')
                state = {k.replace('module.', ''): v for k, v in state.items()} if isinstance(state, dict) else state
                self.backbone.load_state_dict(state)
                print(f"Loaded HRNet checkpoint from {hrnet_ckpt}")
            except Exception:
                pass

        nlc_ckpt = os.path.join(ckpt_root, 'NLCDetection_checkpoint', 'NLCDetection.pth')
        if os.path.exists(nlc_ckpt):
            try:
                state = torch.load(nlc_ckpt, map_location='cpu')
                state = {k.replace('module.', ''): v for k, v in state.items()} if isinstance(state, dict) else state
                model_sd = self.loc_head.state_dict()
                state = {k: v for k, v in state.items() if k in model_sd and model_sd[k].shape == v.shape}
                self.loc_head.load_state_dict(state,strict=False)
                print(f"Loaded NLCDetection checkpoint from {nlc_ckpt}")
            except Exception:
                pass

        det_ckpt = os.path.join(ckpt_root, 'DetectionHead_checkpoint', 'DetectionHead.pth')
        if os.path.exists(det_ckpt):
            try:
                state = torch.load(det_ckpt, map_location='cpu')
                state = {k.replace('module.', ''): v for k, v in state.items()} if isinstance(state, dict) else state
                self.cls_head.load_state_dict(state)
                print(f"Loaded DetectionHead checkpoint from {det_ckpt}")
            except Exception:
                pass


    def forward(self, x):
        feat = self.backbone(x)
        seg_map, seg_map_1, seg_map_2, seg_map_3 = self.loc_head(feat)

        cls_logits = self.cls_head(feat)
        if self.training:
            return [seg_map, seg_map_1, seg_map_2, seg_map_3,], cls_logits
        else:
            return seg_map, cls_logits


# ----------------- Example -----------------
if __name__ == "__main__":
    
    model = PSCCNetCombined()
    model.eval()

    dummy = torch.randn(1, 3, 1536, 1536)
    seg_map, cls_logits = model(dummy)
    print("Seg map:", seg_map.shape)
    print("Cls logits:", cls_logits.shape)

    model.train()
    dummy = torch.randn(1, 3, 1536, 1536)
    seg_map, cls_logits = model(dummy)
    for index,el in enumerate(seg_map):
        print("Seg map:",index, el.shape)
