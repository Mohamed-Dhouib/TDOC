"""
ASCFormer model for tampering detection.
Files copied from RTM/ASCFormer/mmseg/models/
"""
# Register all model components with mmseg registry
from mmseg.registry import MODELS

# Import and register backbone
from .aysm_cmnext import AsymCMNeXtV2
MODELS.register_module(module=AsymCMNeXtV2, force=True)

# Import and register decode head
from .contrastive_head import ContrastiveHeadV2
MODELS.register_module(module=ContrastiveHeadV2, force=True)


from .ascformer_model import ASCFormerModel

__all__ = ['ASCFormerModel']
