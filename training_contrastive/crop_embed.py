import torch
import math
from torch import nn
from torch.nn import functional as F
from timm.models.layers import trunc_normal_

class LayerNorm(nn.Module):
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
    """ConvNeXt-style backbone for F_theta crop embedding (Appendix C.2)."""
    def __init__(self, dims=[32,96, 192], layer_scale_init_value=1e-6,max_h: int = 16, max_w: int = 256, num_queries=8, att_dim=128):
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

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.pool_max = nn.AdaptiveMaxPool2d(1)

        self.num_queries = num_queries
        self.query_embed = nn.Parameter(torch.randn(self.num_queries, att_dim))
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=att_dim,
            num_heads=8,
            batch_first=True
        )
        self.k_proj = nn.Linear(dims[2], att_dim)
        self.v_proj = nn.Linear(dims[2], att_dim)

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
        """Pad right/bottom by 1 with reflect if H or W is odd (for k=2,s=2): avoid loosing pixels"""  
        _, _, H, W = x.shape  
        pad_h = H % 2  
        pad_w = W % 2  
        if pad_h or pad_w:  
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")  
        return x  

    def forward(self, x):
        x = self._reflect_pad_if_needed(x)
        x = self.downsample_layers[0](x)
        x = self.stages[0](x)
    
        x = self._reflect_pad_if_needed(x)
        x = self.downsample_layers[1](x)
        x = self.stages[1](x)

        B, C, Hf, Wf = x.shape

        pe = self.pos_emb[:,:,:Hf, :Wf].clone()
        x = x + pe
        
        x = self._reflect_pad_if_needed(x)
        x = self.downsample_layers[2](x)
        x = self.stages[2](x)


        B, C, Hf, Wf = x.shape
    
        feat = x.flatten(2).permute(0, 2, 1)
    
        queries = self.query_embed.unsqueeze(0).expand(B, -1, -1)
        k = self.k_proj(feat)
        v = self.v_proj(feat)

        attn_out, _ = self.cross_attn(queries, k, v)
    
        query_feat = attn_out.reshape(B, -1)


        output = torch.cat((self.global_pool(x).view(B, C),self.pool_max(x).view(B, C),query_feat),dim=1)
        return output



class MetaNetwork(nn.Module):
    """F_theta: Crop similarity network (Section 3.1). Outputs foreground/background embeddings for S(x,u) computation."""
    def __init__(self):
        super().__init__()

        self.rgb_backbone= RGB()

        C = self.rgb_backbone.dims[-1]         # 192
        self.C= C
        Q = self.rgb_backbone.num_queries      # 8
        A = self.rgb_backbone.k_proj.out_features  # 128
        self.FF = nn.Sequential(
            nn.Linear(2*C + Q*A, 4*C),
            nn.GELU(),
            nn.Linear(4*C, C),
        )



    def forward(self, x):


        rgb_features = self.rgb_backbone(x)

        rgb_features=self.FF(rgb_features)
        

        rgb_features_0,rgb_features_1=rgb_features[:,:int(self.C/2)],rgb_features[:,int(self.C/2):]

        rgb_features_0 = F.normalize(rgb_features_0, dim=-1)
        rgb_features_1 = F.normalize(rgb_features_1, dim=-1)
        output=torch.cat([rgb_features_0,rgb_features_1],dim=-1)
    
        return output