"""FFDN architecture using standard mmseg components.

This implementation uses mmseg's FPNHead and FCNHead instead of custom implementations.
Only FFDN-specific components (TimmDct, DWTFPN) are kept custom.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import torch

from mmseg.models.segmentors import EncoderDecoder
from mmseg.registry import MODELS
from mmengine.model import BaseModule

from .dwt_fpn import DWTFPN
from .timm_backbone import TimmDct
import torch.nn.functional as F
import torch.nn as nn


# ---------------------------------------------------------
# Classifier
# ---------------------------------------------------------
class Last3DecoderClassifier(nn.Module):
    """
    Take the 3 most semantic decoder maps (p2_up, p3_up, p4_up),
    do GAP+GMP on each, concat, then add logit-based stats
    (mean prob, max prob, top-k mean on tampered class),
    then MLP with GELU -> image logits.
    """
    def __init__(
        self,
        in_channels_each=256,
        num_classes=2,
        dropout=0.0,
        topk_ratio=0.05,
        hidden_dim=512,
    ):
        super().__init__()
        # 3 decoder maps * (avg+max) per map
        dec_feat_dim = 3 * (2 * in_channels_each)   # 3 * 512 = 1536 if in_ch=256
        logit_feat_dim = 3                          # mean, max, topk_mean
        total_in = dec_feat_dim + logit_feat_dim

        self.topk_ratio = float(topk_ratio)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # MLP with GELU (no direct linear)
        self.mlp = nn.Sequential(
            nn.Linear(total_in, hidden_dim),
            nn.GELU(),
            self.dropout,
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, aligned_feats, seg_logits):
        """
        aligned_feats: list/tuple like [p1, p2_up, p3_up, p4_up]
        seg_logits:    (B,2,H,W)
        returns:       (B, num_classes)
        """
        # 1) take last 3 decoder maps
        feats_last3 = aligned_feats[-3:]  # p2_up, p3_up, p4_up

        pooled = []
        for f in feats_last3:
            # f: (B, C, H, W)
            avg = F.adaptive_avg_pool2d(f, 1)  # (B, C, 1, 1)
            mx = F.adaptive_max_pool2d(f, 1)   # (B, C, 1, 1)
            v = torch.cat([avg, mx], dim=1).flatten(1)  # (B, 2C)
            pooled.append(v)

        dec_vec = torch.cat(pooled, dim=1)  # (B, 3 * 2C)

        # 2) logit->prob->tampered and stats
        # seg_logits: (B,2,H,W)
        prob = seg_logits.softmax(dim=1)    # (B,2,H,W)
        p_tamp = prob[:, 1]                 # (B,H,W)

        p_mean = p_tamp.mean(dim=(1, 2))    # (B,)
        p_max = p_tamp.amax(dim=(1, 2))     # (B,)

        B, H, W = p_tamp.shape
        N = H * W
        k = max(1, int(min(self.topk_ratio, 1.0) * N))
        flat = p_tamp.flatten(1)            # (B, N)
        topk_vals, _ = flat.topk(k, dim=1)
        p_topk_mean = topk_vals.mean(dim=1) # (B,)

        logit_stats = torch.stack([p_mean, p_max, p_topk_mean], dim=1)  # (B,3)

        # 3) concat decoder features + logit features
        x = torch.cat([dec_vec, logit_stats], dim=1)  # (B, total_in)
        x = self.mlp(x)
        return x

# Register custom FFDN components with mmseg
@MODELS.register_module(name='TimmDct')
@MODELS.register_module(name='TimmDctBackbone')
class TimmDctBackbone(TimmDct, BaseModule):
    """TimmDct backbone registered with mmseg.
    
    This is the FFDN-specific backbone with DCT fusion - not in standard mmseg.
    """
    
    def __init__(self, init_cfg=None, **kwargs):
        BaseModule.__init__(self, init_cfg=init_cfg)
        TimmDct.__init__(self, **kwargs)


@MODELS.register_module(name='DWTFPN')
@MODELS.register_module(name='DWTFPNNeck')
class DWTFPNNeck(DWTFPN, BaseModule):
    """DWTFPN neck registered with mmseg.
    
    This is the FFDN-specific wavelet neck - not in standard mmseg.
    """
    
    def __init__(self, init_cfg=None, **kwargs):
        BaseModule.__init__(self, init_cfg=init_cfg)
        DWTFPN.__init__(self, **kwargs)


class FFDNModel(EncoderDecoder):
    """FFDN segmentation network using mmseg's EncoderDecoder.
    
    Uses standard mmseg components:
    - EncoderDecoder base class
    - FPNHead for decode head
    - FCNHead for auxiliary head
    
    Only custom components:
    - TimmDctBackbone (FFDN-specific DCT fusion)
    - DWTFPNNeck (FFDN-specific wavelet processing)
    """

    def __init__(
        self,
        num_classes: int = 2,
        backbone_cfg: Optional[Dict] = None,
        neck_cfg: Optional[Dict] = None,
        decode_head_cfg: Optional[Dict] = None,
        init_cfg: Optional[Dict] = None,
    ) -> None:
        """Initialize FFDN model.
        
        Args:
            num_classes: Number of segmentation classes
            backbone_cfg: TimmDctBackbone config (if None, uses defaults)
            neck_cfg: DWTFPNNeck config (if None, uses defaults)
            decode_head_cfg: FPNHead config (if None, uses defaults)
            init_cfg: Weight initialization config
        """
        # Set default configs for FFDN
        if backbone_cfg is None:
            backbone_cfg = dict(
                type='TimmDct',
                out_indices=(0, 1, 2, 3),
                fusion='ZERO',
            )
        
        if neck_cfg is None:
            neck_cfg = dict(
                type='DWTFPN',
                in_channels=[128, 256, 512, 1024],
                out_channels=256,
                num_outs=4,
            )
        
        if decode_head_cfg is None:
            decode_head_cfg = dict(
                type='FPNHead',
                in_channels=[256, 256, 256, 256],
                in_index=[0, 1, 2, 3],
                feature_strides=[4, 8, 16, 32],
                channels=512,
                dropout_ratio=0.,
                num_classes=num_classes,
                norm_cfg=dict(type='SyncBN', requires_grad=True),
                align_corners=False,
            )
        


        # Initialize parent EncoderDecoder
        super().__init__(
            backbone=backbone_cfg,
            decode_head=decode_head_cfg,
            neck=neck_cfg,
            auxiliary_head=None,
            init_cfg=init_cfg,
        )

        self.classifier=Last3DecoderClassifier()
    
    def extract_feat(self, inputs: Dict[str, torch.Tensor]) -> List[torch.Tensor]:
        """Extract features with DCT support.
        
        Args:
            inputs: Dict with keys 'x' (image), 'dct' (DCT coefficients), 'qtb' (quantization table)
        
        Returns:
            List of feature tensors from neck
        """
        x = self.backbone(inputs)
        x = self.neck(x)
        return x
    
    def forward(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Forward pass returning segmentation logits.
        
        Args:
            inputs: Dict with keys 'x', 'dct', 'qtb'
        
        Returns:
            Segmentation logits (B, num_classes, H, W)
        """
        input_size = inputs['x'].shape[2:]
        features = self.extract_feat(inputs)
        seg_logits = self.decode_head.forward(features)

        seg_logits = F.interpolate(
                seg_logits,
                size=input_size,
                mode='bilinear',
                align_corners=False
            )

        cls_logit =self.classifier(features,seg_logits)

        return seg_logits, cls_logit


def build_ffdn_model(
    num_classes: int = 2,
) -> FFDNModel:
    """Build FFDN model with standard configurations.
    
    Args:
        num_classes: Number of segmentation classes
    
    Returns:
        FFDNModel instance
    
    Example:
        >>> model = build_ffdn_model(num_classes=2)
        >>> inputs = {'x': img, 'dct': dct, 'qtb': qtb}
        >>> logits = model(inputs)  # (B, 2, H, W)
    """
    model = FFDNModel(
        num_classes=num_classes)

    return model


__all__ = [
    'FFDNModel',
    'TimmDctBackbone',
    'DWTFPNNeck',
    'build_ffdn_model',
]

