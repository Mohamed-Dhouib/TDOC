import os
from typing import Union
import PIL
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms
from transformers.modeling_utils import PretrainedConfig, PreTrainedModel
import cv2
import torchvision 
from meta_cropped import MetaNetwork
import torch.nn.functional as F

class Encoder(nn.Module):
    """Encoder wrapping G_theta for bounding box quality assessment (Section 3.2)."""

    def __init__(
        self,
        pretrained_model_name_or_path: Union[str, bytes, os.PathLike] = None,
    ):
        super().__init__()
        self.to_tensor = transforms.Compose(
            [
                transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=(0.485, 0.455, 0.406), std=(0.229, 0.224, 0.225)),
            ]
        )

        self.model = MetaNetwork()

      
        if pretrained_model_name_or_path:
            old_state_dict = torch.load(pretrained_model_name_or_path,map_location='cpu')
            
            
    
            new_state_dict = self.model.state_dict()
            for x in new_state_dict:

                if "module.model.encoder.model."+x in old_state_dict.keys():
                    key="module.model.encoder.model."+x
                elif "model.encoder.model."+x in old_state_dict.keys():
                    key="model.encoder.model."+x
                elif "module.model.encoder._orig_mod.model."+x in old_state_dict.keys():
                    key="module.model.encoder._orig_mod.model."+x
                else :
                    print(f"{x} not found")
                    key=False
                if key :
                    old_dims=old_state_dict[key].shape
                    new_dims=new_state_dict[x].shape
                    slices = tuple(slice(0, min(old_dim, new_dim)) for old_dim, new_dim in zip(old_dims, new_dims))
                    new_state_dict[x][slices] = old_state_dict[key][slices]
                    
            self.model.load_state_dict(new_state_dict)
            print("Weights loaded")
        else:
            print("!!!No weights detected — crop_quality model has been newly initialized.!!!")   
    
            
    def forward(self, x,micro_top_bottom,micro_left_right,micro_image_mask, micro_tb_mask, micro_lr_mask, use_only_image=False):
        
        output = self.model(x,micro_top_bottom,micro_left_right,micro_image_mask, micro_tb_mask, micro_lr_mask, use_only_image=use_only_image)

        return output

    def prepare_input(self, img,pad=True ):
      
        if type(img)==str :
            img=cv2.imread(img)
         
        if not isinstance(img, PIL.Image.Image) :
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)

        tensor = self.to_tensor(img)           
        _, h, w = tensor.shape
        mask = torch.ones((h, w), dtype=torch.bool)

        if pad:
            pad_w = max(0, 16 - w)
            pad_h = max(0, 16 - h)
            if pad_w or pad_h:
                tensor = F.pad(tensor, (0, pad_w, 0, pad_h), value=0.0)
                mask   = F.pad(mask,   (0, pad_w, 0, pad_h), value=False)
        return tensor, mask


class DocMDetectorConfig(PretrainedConfig):
    """Configuration for G_theta bounding box quality model."""

    model_type = "DocMDetector"

    def __init__(
        self,
        pretrained_model_name_or_path: Union[str, bytes, os.PathLike] = "",
        **kwargs,
    ):
        super().__init__()
        self.pretrained_model_name_or_path = pretrained_model_name_or_path



class DocMDetector(PreTrainedModel):
    """G_theta model wrapper for bounding box quality (Section 3.2)."""

    config_class = DocMDetectorConfig
    base_model_prefix = "DocMDetector"

    def __init__(self, config: DocMDetectorConfig):
        super().__init__(config)
        self.config = config
        self.encoder = Encoder(
            pretrained_model_name_or_path=self.config.pretrained_model_name_or_path,
        )



