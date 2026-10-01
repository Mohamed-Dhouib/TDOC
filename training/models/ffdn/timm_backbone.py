"""ConvNeXtV2 backbone with FPH fusion copied from the official FFDN repo."""

from __future__ import annotations

from typing import Dict, Iterable, Tuple

import torch
import torch.nn as nn

from .fph import FPH, SCSEModule
from .convnextv2 import convnextv2_base


class TimmDct(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        out_indices: Iterable[int] = (0, 1, 2, 3),
        init_cfg=None,
        fusion: str = 'ZERO',
        **kwargs,
    ) -> None:
        super().__init__()

        self.out_indices = tuple(out_indices)
        self.fusion = fusion.upper()
        self.init_cfg = init_cfg

        norm_layer = kwargs.get('norm_layer')
        if isinstance(norm_layer, str):
            raise ValueError('String norm_layer is not supported without mmengine registry support')

        # Create ConvNeXtV2 Base model directly (no timm dependency, works offline)
        self.timm_model = convnextv2_base(in_chans=in_channels, num_classes=0)
        # Remove head since we only need features
        if hasattr(self.timm_model, 'head'):
            self.timm_model.head = None
            self.timm_model.norm = None

        self._is_init = False

        self.fph = FPH()
        
        if self.fusion == 'ZERO':
            fusion_units = [
                nn.Sequential(
                    SCSEModule(512),
                    nn.Conv2d(512, 256, 3, 1, 1),
                    nn.BatchNorm2d(256),
                    nn.ReLU(inplace=True),
                )
            ]
            proj = nn.Conv2d(256, 256, 1, 1, 0)
            nn.init.zeros_(proj.weight)
            if proj.bias is not None:
                nn.init.zeros_(proj.bias)
            fusion_units.append(proj)
            self.FU = nn.ModuleList(fusion_units)
        elif self.fusion in {'SCSE', 'ADD'}:
            raise NotImplementedError(f'Fusion mode {self.fusion} is not implemented')
        else:
            raise NotImplementedError(f'Unknown fusion mode {self.fusion}')

    def forward(self, inputs: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, ...]:
        x = inputs['x']
        dct = inputs['dct']
        qtb = inputs['qtb']
        f_dct = self.fph(dct, qtb)

        outs = []
        # Use Meta's ConvNeXtV2 components to mirror timm's forward pass
        # Meta uses downsample_layers[0] as stem, downsample_layers[1-3] for stage transitions
        # timm does: stem(x) then for each stage: stage(x)
        # Meta does: for each i: downsample_layers[i](x) then stages[i](x)
        
        # Apply stem (downsample_layers[0])
        x = self.timm_model.downsample_layers[0](x)
        x = self.timm_model.stages[0](x)
        if 0 in self.out_indices:
            outs.append(x)
        
        # Apply remaining stages with their downsampling
        for idx in range(1, 4):
            x = self.timm_model.downsample_layers[idx](x)
            x = self.timm_model.stages[idx](x)
            
            # Apply DCT fusion at stage 1 (idx=1)
            if self.fusion == 'ZERO' and idx == 1:
                ext = self.FU[0](torch.cat((x, f_dct), dim=1))
                x = self.FU[1](ext) + x
            
            if idx in self.out_indices:
                outs.append(x)
        
        return tuple(outs)


__all__ = ['TimmDct']
