"""
ASCFormer model for tampering detection.
Wrapper around the MyModelFull from RTM/ASCFormer with added classification head.
"""
import os
os.environ['LIGHTLY_DISABLE_VERSION_CHECK'] = '1'
import torch
import torch.nn as nn
from .my_model_full import MyModelFull


class ClassificationHead(nn.Module):
    """
    Classification head for image-level tampering detection.
    Takes multi-scale features and produces binary classification (authentic vs tampered).
    """
    
    def __init__(self, in_channels_list=[64, 128, 320, 512], hidden_dim=256, num_classes=2):
        """
        Args:
            in_channels_list: List of channel dimensions from backbone stages
            hidden_dim: Hidden dimension for fusion
            num_classes: Number of classes (2 for binary: authentic/tampered)
        """
        super().__init__()
        
        # Adaptive pooling to reduce spatial dimensions
        self.pool = nn.AdaptiveAvgPool2d(1)
        
        # Per-scale projection layers
        self.projections = nn.ModuleList([
            nn.Linear(in_c, hidden_dim) for in_c in in_channels_list
        ])
        
        # Fusion and classification layers
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * len(in_channels_list), hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, num_classes)
        )
    
    def forward(self, features):
        """
        Args:
            features: List of multi-scale features from backbone [f1, f2, f3, f4]
                     Each feature has shape (B, C_i, H_i, W_i)
        
        Returns:
            Classification logits (B, num_classes)
        """
        pooled_features = []
        
        for i, feat in enumerate(features):
            # Global average pooling: (B, C, H, W) -> (B, C, 1, 1) -> (B, C)
            pooled = self.pool(feat).flatten(1)
            # Project to hidden dim: (B, C) -> (B, hidden_dim)
            projected = self.projections[i](pooled)
            pooled_features.append(projected)
        
        # Concatenate all scales: (B, hidden_dim * num_scales)
        fused = torch.cat(pooled_features, dim=1)
        
        # Final classification: (B, hidden_dim * num_scales) -> (B, num_classes)
        logits = self.fusion(fused)
        
        return logits


class ASCFormerModel(nn.Module):
    """
    Wrapper for ASCFormer (MyModelFull) to match the Encoder interface.
    Adds a trainable classification head for image-level tampering detection.
    """
    
    def __init__(self, num_classes=2):
        super().__init__()
        
        # Build config matching ascformer_rtm.py
        norm_cfg = dict(type='SyncBN', requires_grad=True)
        # checkpoint = 'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/segformer/mit_b2_20220624-66e8bf70.pth'
        
        # Backbone config
        backbone = dict(
            type='AsymCMNeXtV2',
            use_rectifier=False,
            in_stages=(1, 0, 0),
            extra_patch_embed=dict(
                in_channels=64,
                embed_dims=128,
                kernel_size=3,
                stride=1,
                reshape=True,
            ),
            out_indices=(0, 1, 2, 3),
            backbone_main=dict(
                type='MixVisionTransformer',
                # pretrained=checkpoint,
                in_channels=3,
                embed_dims=64,
                num_stages=4,
                num_layers=[3, 4, 6, 3],
                num_heads=[1, 2, 5, 8],
                patch_sizes=[7, 3, 3, 3],
                sr_ratios=[8, 4, 2, 1],
                out_indices=(0, 1, 2, 3),
                mlp_ratio=4,
                qkv_bias=True,
                drop_rate=0.0,
                attn_drop_rate=0.0,
                drop_path_rate=0. #0.1
            ),
            backbone_extra=dict(
                type='HubVisionTransformer',
                # pretrained=checkpoint,
                in_channels=3,
                embed_dims=64,
                modals=['dct', 'srm', 'ela'],
                in_modals=(2, 3, 3, 3),
                skip_patch_embed_stage=1,
                num_stages=4,
                num_layers=[3, 4, 6, 3],
                num_heads=[1, 2, 5, 8],
                patch_sizes=[7, 3, 3, 3],
                strides=[4, 2, 2, 2],
                sr_ratios=[8, 4, 2, 1],
                out_indices=(0, 1, 2, 3),
                mlp_ratio=4,
                qkv_bias=True,
                drop_rate=0.0,
                attn_drop_rate=0.0,
                drop_path_rate=0. #0.1
            ),
            fuser=dict(
                type='NATFuserBlock',
                kernel_size=5,
                gated=True,
                post_attn=True,
                attn_mode='cross',
            ),
        )
        
        # Preprocessors for secondary stream (match ascformer_rtm.py)
        preprocessor_sec = [
            ['dct', dict(
                type='DCTProcessor',
                in_channels=1,
                embed_dims=64,
                num_heads=1,
                patch_size=3,
                stride=1,
                sr_ratio=4,
                out_channels=64,
                norm_cfg=norm_cfg,
                reduce_neg=False,
            )],
            ['ela', dict(
                type='NoFilter',
            )],
            ['img', dict(
                type='SRMConv2d_simple',
                inc=3,
                learnable=False,
            )],
        ]
        
        # Decode head config
        decode_head = dict(
            type='ContrastiveHeadV2',
            in_channels=[64, 128, 320, 512],
            in_index=[0, 1, 2, 3],
            channels=256,
            dropout_ratio=0., #0.1
            num_classes=num_classes,
            norm_cfg=norm_cfg,
            align_corners=False,
            use_cl=False,
            up_decode=True,
            loss_decode=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0
            ),
        )
        
        # Initialize MyModelFull with kwargs
        self.model = MyModelFull(
            backbone=backbone,
            decode_head=decode_head,
            preprocessor_sec=preprocessor_sec,
            merge_input=True,
            train_cfg=dict(),
            test_cfg=dict(mode='whole')
        )
        
        # Add trainable classification head
        # ASCFormer uses MiT-B2 backbone with output channels [64, 128, 320, 512]
        self.classification_head = ClassificationHead(
            in_channels_list=[64, 128, 320, 512],
            hidden_dim=256,
            num_classes=num_classes
        )

        pretrained_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "weights",
            "mit_b2_20220624-66e8bf70.pth",
        )
        if os.path.exists(pretrained_path):
            try:
                checkpoint = torch.load(pretrained_path, map_location='cpu', weights_only=False)
                self.model.backbone.main_branch.load_state_dict(checkpoint)
                self.model.backbone.extra_branch.load_state_dict(checkpoint, strict=False)
                print(f"Loaded ASCFormer backbone weights from {pretrained_path}")
            except Exception:
                pass
    
    def forward(self, x, extras, dwt=None):
        """
        Forward pass matching the Encoder interface.
        
        Args:
            x: RGB image (B, 3, H, W)
            extras: Dictionary containing 'dct', 'ela', 'img', and 'qtable' features
            dwt: DWT features (optional, not used by ASCFormer, kept for compatibility)
            
        Returns:
            (segmentation_output, classification_output)
        """
        # Ensure dct has correct shape (B, 1, H, W)

        B, C, H_orig, W_orig = x.shape

        if 'dct' in extras:
            dct = extras['dct']
            if len(dct.shape) == 3:
                extras['dct'] = dct.unsqueeze(1)

        # Forward through the RTM encoder to include auxiliary modalities
        # This mirrors MyModelFull._forward and ensures the fuser/preprocessors run
        backbone_features = self.model.forward_encoder(x, extras)

        # Segmentation logits from ContrastiveHeadV2 (ignore contrastive feats)
        seg_output, _ = self.model.decode_head.forward(backbone_features)

        seg_output = torch.nn.functional.interpolate(
            seg_output,
            size=(H_orig, W_orig),
            mode='bilinear',
            align_corners=False
        )

        # Trainable image-level classification head shares the same features
        cls_logits = self.classification_head(backbone_features)

        return seg_output, cls_logits


if __name__ == '__main__':
    """
    ASCFormer inference with defaults from RTM config.
    
    Usage:
        python -m training_v2_fraud.models.ascformer.ascformer_model \
            --image path/to/image.jpg \
            --checkpoint path/to/model.pth
    """
    import argparse
    import torch.nn.functional as F
    from .preprocessing import preprocess_image, visualize_results
    
    parser = argparse.ArgumentParser(description='ASCFormer Inference')
    parser.add_argument('--image', type=str, required=True, help='Path to input image')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--output', type=str, default='output/result', help='Output path')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'])
    args = parser.parse_args()
    
    # Defaults from RTM config (ascformer_rtm.py)
    QUALITY = 80  # From config: quality = 80
    NUM_CLASSES = 2  # From config: num_classes = 2
    
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        args.device = 'cpu'
    
    print("=" * 60)
    print("ASCFormer Inference")
    print("=" * 60)
    print(f"Image: {args.image}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Device: {args.device}")
    print("=" * 60)
    
    # Initialize model
    print("\n[1/4] Initializing model...")
    model = ASCFormerModel(num_classes=NUM_CLASSES)
    
    # Load checkpoint
    print(f"[2/4] Loading checkpoint...")
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    state_dict = checkpoint.get('state_dict', checkpoint.get('model', checkpoint))
    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    if not any(k.startswith('model.') for k in state_dict.keys()):
        state_dict = {f'model.{k}': v for k, v in state_dict.items()}
    model.load_state_dict(state_dict,strict=False) # strict=False
    model = model.to(args.device)
    model.eval()
    print("  ✓ Model loaded")
    
    # Preprocess
    print(f"\n[3/4] Preprocessing...")
    data = preprocess_image(args.image, quality=QUALITY, device=args.device)
    print(f"  Image shape: {data['ori_shape']}")
    
    # Inference
    print(f"\n[4/4] Running inference...")
    with torch.no_grad():
        extras = {'dct': data['dct'], 'ela': data['ela'], 'img': data['img']}
        seg_output, cls_output = model(data['img'], extras)
        
        seg_probs = F.softmax(seg_output, dim=1)
        seg_prediction = torch.argmax(seg_output, dim=1)[0].cpu().numpy()
        seg_confidence = seg_probs[0, 1].cpu().numpy()
        
        cls_probs = F.softmax(cls_output, dim=1)
        cls_prediction = torch.argmax(cls_output, dim=1)[0].item()
        cls_confidence = cls_probs[0, 1].item()
    
    print("\n" + "=" * 60)
    print("Results:")
    print("=" * 60)
    print(f"Classification: {'TAMPERED' if cls_prediction == 1 else 'AUTHENTIC'}")
    print(f"  Confidence: {cls_confidence:.4f}")
    print(f"\nSegmentation:")
    print(f"  Tampered pixels: {(seg_prediction == 1).sum()} / {seg_prediction.size}")
    print(f"  Tampered ratio: {(seg_prediction == 1).mean():.2%}")
    print("=" * 60)
    
    visualize_results(seg_prediction, seg_confidence, args.output)
    print(f"\n✓ Results saved to {args.output}*")
