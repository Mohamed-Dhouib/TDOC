import torch
import math
import torch.nn as nn
from timm.models.layers import trunc_normal_
import torch.nn.functional as F



class LayerNorm(nn.Module):
    """Layer normalization supporting channels-first format."""

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape, )

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x

class ConvBlock(nn.Module):
    """ConvNeXt-style block with depthwise conv and layer scale."""

    def __init__(self, dim, layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True) if layer_scale_init_value > 0 else None

    def forward(self, x):
        ipt = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = self.pwconv1(x)
        x=torch.nn.functional.gelu(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)
        x = ipt + x
        return x

    

class RGB(nn.Module):
    """ConvNeXt-style encoder for crop embedding in G_theta (Appendix C.2)."""
    def __init__(self, dims=[32,96, 192], layer_scale_init_value=1e-6,max_h: int = 16, max_w: int = 256,):
        super().__init__()
        self.dims=dims
        self.downsample_layers = nn.ModuleList([nn.Sequential(nn.Conv2d(3, dims[0], kernel_size=2, stride=2), LayerNorm(dims[0], eps=1e-6, data_format="channels_first")), nn.Sequential(LayerNorm(dims[0], eps=1e-6, data_format="channels_first"),nn.Conv2d(dims[0], dims[1], kernel_size=2, stride=2)), nn.Sequential(LayerNorm(dims[1], eps=1e-6, data_format="channels_first"),nn.Conv2d(dims[1], dims[2], kernel_size=2, stride=2))])
        self.stages = nn.ModuleList([nn.Sequential(*[ConvBlock(dim=dims[0],layer_scale_init_value=layer_scale_init_value) for j in range(3)]), nn.Sequential(*[ConvBlock(dim=dims[1],layer_scale_init_value=layer_scale_init_value) for j in range(6)]),nn.Sequential(*[ConvBlock(dim=dims[2],layer_scale_init_value=layer_scale_init_value) for j in range(27)])])

        pe = torch.zeros(max_h, max_w, dims[1])
        h_pos = torch.arange(max_h, dtype=torch.float).unsqueeze(1)
        w_pos = torch.arange(max_w, dtype=torch.float).unsqueeze(1)
        d_half = dims[1] // 2
        div_h  = torch.exp(torch.arange(0, d_half, 2, dtype=torch.float) * (-math.log(10000.0) / d_half))
        div_w  = torch.exp(torch.arange(0, d_half, 2, dtype=torch.float) * (-math.log(10000.0) / d_half))
        pe[:, :, 0:d_half:2]  = torch.sin(h_pos * div_h).unsqueeze(1)
        pe[:, :, 1:d_half:2]  = torch.cos(h_pos * div_h).unsqueeze(1)
        pe[:, :, d_half::2]    = torch.sin(w_pos * div_w).unsqueeze(0)
        pe[:, :, d_half+1::2]  = torch.cos(w_pos * div_w).unsqueeze(0)
        self.register_buffer('pos_emb', pe.permute(2, 0, 1).unsqueeze(0)) 


        self.unvalid_input_embed = nn.Parameter(
            torch.randn(3, 1, 1) * 0.01
        )  # small noise around 0


        self.apply(self._init_weights)
 

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Conv1d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @staticmethod
    def _reflect_pad_if_needed(x: torch.Tensor) -> torch.Tensor:
        """Pad right/bottom by 1 with reflect if H or W is odd (for k=2,s=2): avoid loosing pixels: really important for thism odel to work!!"""  
        _, _, H, W = x.shape  
        pad_h = H % 2  
        pad_w = W % 2  
        if pad_h or pad_w:  
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")  
        return x  

    @staticmethod
    def _masked_global_avg_pool(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        m = mask.unsqueeze(1).to(x.dtype)
        num = (x * m).sum(dim=(2, 3))
        den = m.sum(dim=(2, 3)).clamp_min(1e-6)
        return num / den  

    @staticmethod
    def _masked_global_max_pool(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x_masked = x.masked_fill(~mask.unsqueeze(1), float('-inf'))
        return torch.amax(x_masked, dim=(2, 3))  
        
    @staticmethod
    def _downsample_mask(mask: torch.Tensor) -> torch.Tensor:
        B, H, W = mask.shape
        pad_h = H % 2
        pad_w = W % 2
        mask_f = mask.float()                              
        if pad_h or pad_w:
            mask_f = F.pad(mask_f, (0, pad_w, 0, pad_h), mode="replicate") 
        pooled = F.max_pool2d(mask_f.unsqueeze(1), 2, 2)    
        return (pooled.squeeze(1) > 0)                      

    def forward(self, x, image_mask):

        

        m = image_mask.bool()  

        inv = (~image_mask).unsqueeze(1).float()     
        val = image_mask.unsqueeze(1).float()

        x = x * val + self.unvalid_input_embed * inv


        x = self._reflect_pad_if_needed(x)
        x = self.downsample_layers[0](x)
        m = self._downsample_mask(m) 
        x = self.stages[0](x)

        x = self._reflect_pad_if_needed(x)
        x = self.downsample_layers[1](x)
        m = self._downsample_mask(m)  
        x = self.stages[1](x)

        B, C, Hf, Wf = x.shape

        pe = self.pos_emb[:,:,:Hf, :Wf].clone()
        x = x + pe
        
        x = self._reflect_pad_if_needed(x)
        x = self.downsample_layers[2](x)
        m = self._downsample_mask(m) 
        x = self.stages[2](x)

        B, C, _, _ = x.shape

        gap = self._masked_global_avg_pool(x, m)  
        gmp = self._masked_global_max_pool(x, m)  
        x = torch.cat((gap, gmp), dim=1)                                              
        return x



class ConvNeXt1DBlock(nn.Module):
    """1D ConvNeXt block for edge stripe processing."""

    def __init__(self, dim, layer_scale_init_value=1e-6):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, kernel_size=7, padding=3,groups=dim)
        self.norm   = nn.LayerNorm(dim, eps=1e-6)
        self.fc1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones(dim), requires_grad=True)

    def forward(self, x):
        shortcut = x
        x = self.conv(x)
        x = x.permute(0, 2, 1)
        x = self.norm(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = x.permute(0, 2, 1)
        x = self.gamma.view(1, -1, 1) * x
        return shortcut + x

class SurroundingsEncoder(nn.Module):
    """Encodes edge stripes of thickness t pixels capturing crop neighborhood (Section 3.2)."""
    def __init__(
        self,
        in_channels=3,
        target_pixels=9,
        depth=27, 
        layer_scale_init_value: float = 1e-6,
        hidden_dim=64,
    ):
        super().__init__()
        mid_ch = in_channels * target_pixels
        self.hidden_dim = hidden_dim or mid_ch

        self.proj_tb     = nn.Conv1d(mid_ch, self.hidden_dim, kernel_size=1)
        self.blocks_tb   = nn.Sequential(*[
            ConvNeXt1DBlock(self.hidden_dim, layer_scale_init_value)
            for _ in range(depth)
        ])
        self.proj_lr     = nn.Conv1d(mid_ch, self.hidden_dim, kernel_size=1)
        self.blocks_lr   = nn.Sequential(*[
            ConvNeXt1DBlock(self.hidden_dim, layer_scale_init_value)
            for _ in range(depth)
        ])

        self.apply(self._init_weights)
 

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Conv1d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @staticmethod
    def _masked_avg_pool1d(x: torch.Tensor, mask_1d: torch.Tensor) -> torch.Tensor:
        m = mask_1d.to(x.dtype).unsqueeze(1)
        num = (x * m).sum(-1)
        den = m.sum(-1).clamp_min(1e-6)
        return num / den 

    @staticmethod
    def _masked_max_pool1d(x: torch.Tensor, mask_1d: torch.Tensor) -> torch.Tensor:
        x_masked = x.masked_fill(~mask_1d.unsqueeze(1), float('-inf'))
        return torch.amax(x_masked, dim=-1)  

    def forward(self, top_bottom: torch.Tensor, left_right: torch.Tensor, tb_mask: torch.Tensor,lr_mask: torch.Tensor):
        B, C, tp, W_tb = top_bottom.shape
        _, _, H_lr, tp2 = left_right.shape
        assert tp == tp2, "target_pixels must match"

        tb = top_bottom.view(B, C * tp, W_tb)
        lr = left_right.permute(0, 1, 3, 2).contiguous().view(B, C * tp, H_lr)

        tb = self.proj_tb(tb)
        tb = self.blocks_tb(tb)

        lr = self.proj_lr(lr)
        lr = self.blocks_lr(lr)

        tb_mask_1d = tb_mask.squeeze(1)             
        lr_mask_1d = lr_mask.squeeze(-1)            

        tb_avg = self._masked_avg_pool1d(tb, tb_mask_1d) 
        tb_max = self._masked_max_pool1d(tb, tb_mask_1d) 
        lr_avg = self._masked_avg_pool1d(lr, lr_mask_1d) 
        lr_max = self._masked_max_pool1d(lr, lr_mask_1d) 

        return torch.cat([tb_avg, tb_max, lr_avg, lr_max], dim=1)
        


class MetaNetwork(nn.Module):
    """G_theta: Bounding box quality function (Section 3.2). Outputs in [0,1] where low values indicate ill-defined crops."""
    def __init__(self):
        super().__init__()

        self.rgb_backbone= RGB()

        self.surroundings_backbone= SurroundingsEncoder()

        in_dim  = 192*2 + 4*64  
        hidden  = in_dim*4           
        out_dim = 2

        self.classifier = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, out_dim)
        )

        self.classifier_only_image = nn.Sequential(
            nn.Linear(192*2, hidden),
            nn.GELU(),
            nn.Linear(hidden, out_dim)
        )



    def forward(self, x,micro_top_bottom,micro_left_right,micro_image_mask, micro_tb_mask, micro_lr_mask, use_only_image=False):
     
        rgb_features = self.rgb_backbone(x, micro_image_mask)

        if use_only_image:
            if self.training:
                print("[Warning] use_only_image is set to true while training")
            output=self.classifier_only_image(rgb_features)
            return output

        if self.training:  # only apply during training (use only image)
            if torch.rand(1).item() < 0.5:
                output=self.classifier_only_image(rgb_features)
                return output

        sur_features = self.surroundings_backbone(
            micro_top_bottom, micro_left_right, micro_tb_mask, micro_lr_mask
        )       
        
        feats = torch.cat((rgb_features, sur_features), dim=1)

        output = self.classifier(feats)

        return output
