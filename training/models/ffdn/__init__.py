"""FFDN implementation using standard mmseg components.

This module now uses mmseg's FPNHead and FCNHead directly.
Only FFDN-specific components (TimmDct, DWTFPN) remain custom.

Architecture:
- FFDNModel (EncoderDecoder base)
  ├── TimmDctBackbone (custom - FFDN DCT fusion)
  ├── DWTFPNNeck (custom - FFDN wavelet processing)
  ├── FPNHead (standard mmseg decode head)
  └── FCNHead (standard mmseg auxiliary head)
"""

from .ffdn_model import (
    FFDNModel,
    TimmDctBackbone,
    DWTFPNNeck,
    build_ffdn_model,
)
from .timm_backbone import TimmDct
from .dwt_fpn import DWTFPN, DWTNeck
from .fph import FPH, SCSEModule
from .preprocessing import jpeg_compress_and_load_info, preprocess_image, visualize_results

__all__ = [
    # Main model (uses mmseg EncoderDecoder + FPNHead + FCNHead)
    'FFDNModel',
    'build_ffdn_model',
    # Custom FFDN components (not in standard mmseg)
    'TimmDctBackbone',
    'TimmDct',
    'DWTFPNNeck',
    'DWTFPN',
    'DWTNeck',
    'FPH',
    'SCSEModule',
    # Utilities
    'preprocess_image',
    'jpeg_compress_and_load_info',
    'visualize_results',
]

