import os
from typing import Union
import PIL
from PIL import Image, ImageOps
import torch
import torch.nn as nn
from torchvision import transforms
from transformers.modeling_utils import PretrainedConfig, PreTrainedModel
import cv2
import torchvision 
from crop_embed import MetaNetwork


class Encoder(nn.Module):
    """Encoder wrapping F_theta for crop similarity (Section 3.1, Appendix C.2)."""

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
            print("!!!No weights detected — crop_embed model has been newly initialized.!!!")    
    def forward(self, x):
        
        output = self.model(x)

        return output

    def prepare_input(self, img ):
      
        if type(img)==str :
            img=cv2.imread(img)
         
        if not isinstance(img, PIL.Image.Image) :
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)

        w, h = img.size
        pad_w = max(0, 16 - w)
        pad_h = max(0, 16 - h)
        if pad_w or pad_h:
            img = ImageOps.expand(img, border=(0, 0, pad_w, pad_h), fill=0)

        return self.to_tensor(img)

class DocMDetectorConfig(PretrainedConfig):
    """Configuration for F_theta crop similarity model."""

    model_type = "DocMDetector"

    def __init__(
        self,
        pretrained_model_name_or_path: Union[str, bytes, os.PathLike] = "",
        **kwargs,
    ):
        super().__init__()
        self.pretrained_model_name_or_path = pretrained_model_name_or_path



class DocMDetector(PreTrainedModel):
    """F_theta model wrapper for crop similarity (Section 3.1)."""

    config_class = DocMDetectorConfig
    base_model_prefix = "DocMDetector"

    def __init__(self, config: DocMDetectorConfig):
        super().__init__(config)
        self.config = config
        self.encoder = Encoder(
            pretrained_model_name_or_path=self.config.pretrained_model_name_or_path,
        )
        

  
