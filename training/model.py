import os
import io
try:
    import jpegio
    use_pillow_for_qt = False
except ImportError:
    jpegio = None
    use_pillow_for_qt = True
import re
from typing import List, Union
import numpy as np
import PIL
import torch
import torch.nn as nn
import torchvision 
from torchvision import transforms
from transformers.modeling_utils import PretrainedConfig, PreTrainedModel
import cv2
import random
from PIL import Image
import torchvision.transforms.functional as TF
from PIL import ImageColor
import copy


brightness=transforms.ColorJitter(brightness=[0.3,2.0],contrast=0,saturation=0) 
contrast=transforms.ColorJitter(contrast=[0.3,2.0],brightness=0,saturation=0) 
saturation=transforms.ColorJitter(saturation=[0.3,2.0],brightness=0,contrast=0) 



qt_per_path_dict={}


def qt_from_pillow(image_path):
    """Extract the luminance quantization table from a JPEG file using Pillow."""
    try:
        with Image.open(image_path) as im:
            qtables = im.quantization
            if qtables is None:
                return None
            # Key 0 = luminance table; Pillow returns it as a 64-element sequence
            lum = qtables.get(0)
            if lum is None:
                return None
            return np.array(lum, dtype=np.int32).reshape(8, 8)
    except Exception:
        return None


def color_customized(color, mode):
    if isinstance(color, str):
        color = ImageColor.getcolor(color, mode)
    return color

def expand_customized(image, border, fill=0):
 
    left, top, right, bottom = border
    width = left + image.size[0] + right
    height = top + image.size[1] + bottom
    color = color_customized(fill, image.mode)
    if image.palette:
        raise RuntimeError("didn't expect palette")
    out = Image.new(image.mode, (width, height), color)
    out.paste(image, (left, top))
    return out

def generate_homography(image: Image.Image, mask: Image.Image):
    """Apply random homography transformation to image and mask."""
    image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    mask_cv = cv2.cvtColor(np.array(mask), cv2.COLOR_GRAY2BGR)

    src_pts = np.float32([[0, 0], [0, image.width], [image.width, image.height], [image.height, 0]])

    dst_pts = np.float32([
        [np.random.randint(50, 100), np.random.randint(100, 150)],
        [0, image.width],
        [image.width, np.random.randint(image.width-100, image.width)],
        [np.random.randint(image.height-100, image.height), 0]
    ])

    H, _ = cv2.findHomography(src_pts, dst_pts)

    image_transformed = cv2.warpPerspective(image_cv, H, (image.width, image.height))
    mask_transformed = cv2.warpPerspective(mask_cv, H, (mask.width, mask.height))

    image_transformed = Image.fromarray(cv2.cvtColor(image_transformed, cv2.COLOR_BGR2RGB))
    mask_transformed = Image.fromarray(cv2.cvtColor(mask_transformed, cv2.COLOR_BGR2GRAY))

    return image_transformed, mask_transformed


def random_crop_pad_white(image, mask, base_width, base_height):
    """Pad image/mask if smaller than base size, random crop if larger."""
    width, height = image.size

    left, top, right, bottom= 0,0,0,0
    if width < base_width or height < base_height:

        left = max(0, (base_width - width) // 2)
        top = max(0, (base_height - height) // 2)
        right = max((base_width - width) - left,0)
        bottom = max((base_height - height) - top,0)
        
        image = expand_customized(image, border=(left, top, right, bottom), fill='white')
        mask = expand_customized(mask, border=(left, top, right, bottom), fill='black')
        

    w_img, h_img = image.size
    mask_array = np.array(mask)

    if w_img > base_width or h_img > base_height:

        def random_crop_coords():
            x0 = random.randint(0, w_img - base_width)
            y0 = random.randint(0, h_img - base_height)
            return x0, y0

        total_pos = int((mask_array > 0).sum())
        k = min(total_pos, 32)

        if k == 0 or random.random() > 0.75:
            x0, y0 = random_crop_coords()
        else:
            for _ in range(100):
                x0, y0 = random_crop_coords()
                patch = mask_array[y0 : y0 + base_height, x0 : x0 + base_width]
                if int((patch > 0).sum()) >= k:
                    break

        x1 = x0 + base_width
        y1 = y0 + base_height

        image = image.crop((x0, y0, x1, y1))
        mask  = mask.crop((x0, y0, x1, y1))
 


    return image, mask



def _blockwise_cv2_dct(y: np.ndarray) -> np.ndarray:
    """8x8 block DCT without quantization."""
    Hp, Wp = y.shape
    # reshape into (n_blocks, 8, 8)
    blocks = y.reshape(Hp // 8, 8, Wp // 8, 8).transpose(0, 2, 1, 3)  # (bh, bw, 8, 8)
    bh, bw, _, _ = blocks.shape

    # apply DCT block by block
    for i in range(bh):
        for j in range(bw):
            blocks[i, j] = cv2.dct(blocks[i, j])

    # back to (Hp, Wp)
    dct_hw = blocks.transpose(0, 2, 1, 3).reshape(Hp, Wp)
    return dct_hw


def doctamper_dct_opencv(pil_img: Image.Image, clip_to_20: bool = False, use_abs: bool = False) -> torch.Tensor:
    """DocTamper-style DCT using cv2.dct."""
    L = np.array(pil_img.convert("L"), dtype=np.float32)
    L = L - 128.0 

    dct = _blockwise_cv2_dct(L)
    return dct



def _get_dct_vector_coords(r=8):
    """Get the coordinates with zigzag order (from RTM)."""
    dct_index = []
    for i in range(r):
        if i % 2 == 0:  # start with even number
            index = [(i - j, j) for j in range(i + 1)]
            dct_index.extend(index)
        else:
            index = [(j, i - j) for j in range(i + 1)]
            dct_index.extend(index)
    for i in range(r, 2 * r - 1):
        if i % 2 == 0:
            index = [(i - j, j) for j in range(i - r + 1, r)]
            dct_index.extend(index)
        else:
            index = [(j, i - j) for j in range(i - r + 1, r)]
            dct_index.extend(index)
    dct_idxs = np.asarray(dct_index)
    return dct_idxs
    

def block_dct_transform(img, block_size=8, zigzag=True, channel='Y', shift=False):
    """Block DCT transformation (from RTM/ASCFormer)."""
    if isinstance(img, Image.Image):
        # PIL -> RGB -> BGR
        img = img.convert("RGB")
        img = np.array(img)              # RGB
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    elif isinstance(img, np.ndarray):
        if img.ndim == 2:
            # grayscale -> BGR
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif img.ndim == 3 and img.shape[2] == 3:
            # assume already BGR (cv2 style), just copy to be safe
            img = img.copy()
        else:
            raise ValueError(f"Unsupported ndarray shape: {img.shape}")
    else:
        raise TypeError("img must be a PIL.Image or np.ndarray")

    # Convert to YCrCb and extract channel
    c_table = {'Y': 0, 'U': 1, 'V': 2}
    idx_c = c_table[channel]
    
    img_Y = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    img_Y = img_Y[:, :, idx_c]
    
    H, W = img_Y.shape[:2]
    
    # Pad if needed to make divisible by block_size
    if (H % block_size != 0) or (W % block_size != 0):
        pad_h = (block_size - H % block_size) % block_size
        pad_w = (block_size - W % block_size) % block_size
        img_Y = cv2.copyMakeBorder(img_Y, 0, pad_h, 0, pad_w, 
                                   cv2.BORDER_CONSTANT, value=0)
        H, W = img_Y.shape[:2]
    
    # Reshape to blocks
    img_Y = img_Y.reshape(H // block_size, block_size, 
                         W // block_size, block_size).transpose(0, 2, 1, 3)
    img_Y = img_Y.reshape(-1, block_size, block_size)
    
    img_Y = img_Y.astype(np.float32)
    if shift:
        img_Y = img_Y - 128
    
    # Get zigzag indices
    if zigzag:
        dct_idxs = _get_dct_vector_coords(block_size)
        dct_encode_idx = dct_idxs.transpose(1, 0)
        zigzag_encode_idx = (dct_encode_idx[0], dct_encode_idx[1])
    
    # Apply DCT to each block
    dct = []
    for i in range(img_Y.shape[0]):
        dct_block = cv2.dct(img_Y[i])
        if zigzag:
            dct_block = dct_block[zigzag_encode_idx]
        dct.append(dct_block)
    
    # Reshape back to image
    dct = np.array(dct)
    dct = dct.reshape(H // block_size, W // block_size, 
                     block_size, block_size).transpose(0, 2, 1, 3)
    dct = dct.reshape(H, W)
    
    return dct

def compute_qt_from_quality(quality):
    """Compute JPEG quantization table from quality (libjpeg formula)."""
    # Standard luminance quantization table from JPEG spec (IJG)
    STD_LUMINANCE_QT = np.array([
        [16, 11, 10, 16, 24, 40, 51, 61],
        [12, 12, 14, 19, 26, 58, 60, 55],
        [14, 13, 16, 24, 40, 57, 69, 56],
        [14, 17, 22, 29, 51, 87, 80, 62],
        [18, 22, 37, 56, 68, 109, 103, 77],
        [24, 35, 55, 64, 81, 104, 113, 92],
        [49, 64, 78, 87, 103, 121, 120, 101],
        [72, 92, 95, 98, 112, 100, 103, 99]
    ], dtype=np.float64)
    
    # jpeg_quality_scaling from jcparam.c
    if quality <= 0:
        quality = 1
    if quality > 100:
        quality = 100
    
    if quality < 50:
        scale_factor = 5000 // quality
    else:
        scale_factor = 200 - quality * 2
    
    # jpeg_add_quant_table formula
    qt = np.zeros((8, 8), dtype=np.int32)
    for i in range(8):
        for j in range(8):
            temp = (STD_LUMINANCE_QT[i, j] * scale_factor + 50) // 100
            qt[i, j] = max(1, int(temp))
    
    return qt

def compress_image(img,max_compressions,min_max_compression_rate,image_path=False,qualitys=False,do_compress=True, model_name=None) :
    """Apply JPEG compression and compute DCT/QT based on model type."""
    assert model_name is not None, "model_name must be provided"
    if qualitys==False :
        if random.random() <0.33:
            num1=0
        elif  random.random() <0.33:
            num1=1
        else:
            num1 = max(random.randint(1, max_compressions),1)

        nums2 = [random.randint(min_max_compression_rate[0], min_max_compression_rate[1]) for _ in range(num1)]
    else :
        nums2 = list(qualitys)
        num1=len(nums2)
    im=img
    image=False
    qt=None
    last_quality = nums2[-1] if (num1!=0 and do_compress) else None  # Track the last compression quality for DTD
    shared_buff = io.BytesIO()
    if num1!=0  and do_compress: 
   
        for index,el in enumerate(nums2) :
            shared_buff.seek(0)
            shared_buff.truncate(0)
            im.save(shared_buff, format="JPEG",quality=el)

            shared_buff.seek(0) 
            with Image.open(shared_buff) as _im:
                _im.load()
                im = _im.copy()

        image = im if im.mode == 'RGB' else im.convert('RGB')
    else:
        image = img if isinstance(img, Image.Image) and img.mode == 'RGB' else img.convert('RGB')

    shared_buff.close()


    if model_name=="DTD" or model_name=="ffdn" or model_name=="catnet":
        dct = doctamper_dct_opencv(image)
        
        # Determine quantization table
        if last_quality is not None:
            # Compression was done, use the last quality
            qt = compute_qt_from_quality(last_quality)
        elif image_path and str(image_path).lower().endswith((".jpg", ".jpeg")):
            try:
                if isinstance(image_path, str):
                    if image_path in qt_per_path_dict.keys():
                        qt=qt_per_path_dict[image_path]
                    else:
                        if not use_pillow_for_qt:
                            jpg = jpegio.read(image_path)
                            qt = copy.deepcopy(np.array(jpg.quant_tables[0], copy=True))
                            qt_per_path_dict[image_path]=qt
                            del jpg
                        else:
                            qt = qt_from_pillow(image_path)
                            if qt is None:
                                print("qt extracted with pillow is None setting qt to ones")
                                qt = np.ones((8, 8), dtype=np.int32)
                            qt_per_path_dict[image_path] = qt
                        if len(qt_per_path_dict) > 5:
                            del qt_per_path_dict[next(iter(qt_per_path_dict))]
                else:
                    qt = np.ones((8, 8), dtype=np.int32)
            except Exception as e:
                print(f"Failed to extract QT from {image_path}: {e}, using ones")
                qt = np.ones((8, 8), dtype=np.int32)
        else:
            # No compression and no JPEG path, use all ones
            qt = np.ones((8, 8), dtype=np.int32)
        
        # Quantize DCT coefficients
        # Expand qt to match dct shape: (H, W) -> tile 8x8 blocks
        H, W = dct.shape
        H_blocks, W_blocks = H // 8, W // 8
        qt_tiled = np.tile(qt, (H_blocks, W_blocks))
        
        # Quantize: divide DCT by QT, round
        dct_quantized = np.round(dct / qt_tiled).astype(np.int32)
        
        # Clip absolute value to [0, 20] as before
        dct = np.clip(np.abs(dct_quantized), 0, 20).astype(np.int32)
        
        # Add batch dimension to qt
        qt = np.expand_dims(qt, axis=0).astype(np.int32)
        
    elif model_name=="ascformer":
        dct = block_dct_transform(image, zigzag=True)
        dct = np.expand_dims(dct, axis=0)  # shape: (1, H, W)
    else:
        dct = np.zeros((8, 8), dtype=float) #dummy


    if qt is None:
        qt=np.zeros((8, 8), dtype=float) #dummy

        
    return image,dct,qt

def augment(img,mask,probabilities):
    """Apply spatial, visual and flip augmentations to image and mask."""
    blur = lambda img : (TF.gaussian_blur(img, kernel_size=random.choice([3, 5, 7])))


    angle = random.randint(-5, 5)
    rotate = lambda img, mask: (TF.rotate(img, angle=angle, fill=(255,255,255)), TF.rotate(mask, angle=angle, fill=0))
    
    hflip = lambda img, mask: (TF.hflip(img), TF.hflip(mask))

    vflip = lambda img, mask: (TF.vflip(img), TF.vflip(mask))


    rot90   = lambda img, mask: (TF.rotate(img,  90, fill=(255,255,255)), TF.rotate(mask,  90, fill=0))
    rotm90  = lambda img, mask: (TF.rotate(img, -90, fill=(255,255,255)), TF.rotate(mask, -90, fill=0))
    rot180  = lambda img, mask: (TF.rotate(img, 180, fill=(255,255,255)), TF.rotate(mask, 180, fill=0))
    rotate_ninety_ops = [rot90, rotm90, rot180]
    index=np.random.randint(0, len(rotate_ninety_ops))
    rotate_harsh=rotate_ninety_ops[index]

    flips=[hflip,vflip]
    index=np.random.randint(0, len(flips))
    flip_chosen=flips[index]
    
    process=[brightness,contrast,saturation,blur]    
    
    flip=[flip_chosen,rotate_harsh]

    move=[random_shrink_rescale,rotate]
    
    data_aug=[(move,probabilities["spatial"]),(process,probabilities["visual"]),(flip,probabilities["flip"])]

    
    for index,aug_operation in enumerate(data_aug) :
        if index==0 or index==2:
            for operation in aug_operation[0]:
                if random.random()<aug_operation[1] :
                    img, mask = operation(img, mask)
        else :
            for operation in aug_operation[0]:
                if random.random()<aug_operation[1] :
                    img=operation(img)
            

    
    return img,mask

def random_shrink_rescale(image,mask):
        original_size = image.size
        scale = 0.5+np.random.rand()*0.5 
        if random.random() >0.5 :   
            scale=1/scale

        new_size = (int(original_size[0] * scale), int(original_size[1] * scale))


        
        FILTERS = [
            Image.BILINEAR,
            Image.BICUBIC,
            Image.LANCZOS,
        ]

        chosen_filter = random.choice(FILTERS)

        image = image.resize(new_size, chosen_filter)

        mask = mask.resize(new_size, Image.NEAREST) 
        

        return image,mask
    
def ela_transform_ascformer(img):
    """Error Level Analysis (from RTM/ASCFormer). Quality=80."""
    # normalize to OpenCV-style BGR uint8
    if isinstance(img, Image.Image):
        # PIL -> RGB -> BGR
        img = img.convert("RGB")
        img = np.array(img)              # RGB
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    elif isinstance(img, np.ndarray):
        if img.ndim == 2:
            # grayscale -> BGR
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif img.ndim == 3 and img.shape[2] == 3:
            # assume already BGR (cv2 style), just copy to be safe
            img = img.copy()
        else:
            raise ValueError(f"Unsupported ndarray shape: {img.shape}")
    else:
        raise TypeError("img must be a PIL.Image or np.ndarray")

    if img.dtype != np.uint8:
        img = img.astype(np.uint8)

    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 80]
    _, encimg = cv2.imencode('.jpg', img, encode_param)
    decimg = cv2.imdecode(encimg, 1)
    ela = cv2.absdiff(img, decimg)
    return ela

class Encoder(nn.Module):
    """Encoder wrapper for document manipulation detection models."""

    def __init__(
        self,
        input_size: List[int],
        pretrained_model_name_or_path: Union[str, bytes, os.PathLike] = None,
        data_aug_config = None,
        model_name= None,
    ):
        super().__init__()

        self.model_name=model_name
        self.input_size = input_size
        self.data_aug_config = data_aug_config
     

        self.to_tensor = transforms.Compose(
            [
                transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=(0.485, 0.455, 0.406), std=(0.229, 0.224, 0.225)),
            ]
        )
        
        self.to_tensor_mask = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )
        
        if self.model_name=="DTD":
            from models.DTD.dtd import seg_dtd
            self.model = seg_dtd(n_class=2)
        elif self.model_name=="psccnet":
            from models.psccnet.model import PSCCNetCombined
            self.model = PSCCNetCombined()
        elif self.model_name=="catnet":
            from models.catnet.model import CATWrapper
            self.model = CATWrapper()
        elif self.model_name=="ascformer":
            from models.ascformer.ascformer_model import ASCFormerModel
            self.model = ASCFormerModel(num_classes=2)
        elif model_name=="ffdn":
            from models.ffdn import build_ffdn_model
            self.model = build_ffdn_model(
                    num_classes=2
                )
        else:
            raise ValueError(
                f"Unsupported model_name '{self.model_name}'. "
                "Expected one of: DTD, psccnet, catnet, ascformer, ffdn."
            )
      
        if pretrained_model_name_or_path:
            print("Loading weights from", pretrained_model_name_or_path)
            old_state_dict = torch.load(pretrained_model_name_or_path,map_location='cpu')
            if 'state_dict' in old_state_dict:
                print(f"state_dict key detected in {pretrained_model_name_or_path}, fetching it!")
                old_state_dict=old_state_dict['state_dict']
            new_state_dict = self.model.state_dict()
            

            for x in new_state_dict:
                ignore_key= lambda k: bool(re.search(r"swin\.\d+\.blocks\.\d+\.attn_mask", k))
                if ignore_key(x) :
                    print(f"skipping {x}")
                else :
                    if x in old_state_dict.keys():
                        key=x
                    elif "module."+x in old_state_dict.keys():
                        key="module."+x
                    elif "module.model.encoder.model."+x in old_state_dict.keys():
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
                        if old_dims != new_dims: print(f"[slice copy] {x}: {old_dims} -> {new_dims}")
                        
            self.model.load_state_dict(new_state_dict)
            print("Weights loaded")

    def forward(self, x, dcts, dwt, ela, qt):

        if self.model_name=="DTD":
            output = self.model(x,dcts,qt)
        elif self.model_name=="catnet":
            output = self.model(x,dcts,qt)
        elif self.model_name=="ascformer":
            extras = {'dct': dcts, 'ela': ela, 'img': x}
            output = self.model(x,extras)
        elif self.model_name=="ffdn":
            extras = {'x': x, 'dct': dcts, 'qtb': qt}
            output = self.model(extras)
        else:    
            output = self.model(x)
        return output

    def prepare_input(self, img ,mask, aug=True,qualitys=False,crop_pad=True,image_path_to_compute_qt=None) -> torch.Tensor:
        
        try:
            prob_disable=self.data_aug_config["prob_disable"]
        except:
            prob_disable=0

        if  random.random()<prob_disable: 
            aug=False
            qualitys=[]

        aug= aug and self.data_aug_config["enabled"]

        if type(img)==str :
            image_path=img
            img=cv2.imread(img)
        else:
            image_path=None
            
        if not isinstance(img, PIL.Image.Image) :
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)

        base_width=self.input_size[1]
        base_height=self.input_size[0]
            
        if type(mask)==str :
            mask=cv2.imread(mask,cv2.IMREAD_GRAYSCALE)
        if not isinstance(mask, PIL.Image.Image) :
            mask = mask / 255
            mask = Image.fromarray(mask)
        
        if aug:
            img,mask=augment(img,mask,probabilities=self.data_aug_config["probs"] )
            
        if crop_pad :
            img,mask=random_crop_pad_white(img, mask, base_width, base_height)
        
        try :
        
            if aug and random.random()>0.9:
                img,mask=generate_homography(img,mask)
        
        except :
            print("homography failed")

        if aug and self.data_aug_config is not None:
            img,dct,qt=compress_image(img,self.data_aug_config["compression"]["max_num"],self.data_aug_config["compression"]["quality_range"],qualitys=qualitys,do_compress=aug,model_name=self.model_name)

        else :
            assert image_path_to_compute_qt is not None
            img,dct,qt=compress_image(img,None,None,image_path=image_path_to_compute_qt,qualitys=qualitys,do_compress=aug,model_name=self.model_name)

        dct=torch.from_numpy(dct)
        qt=torch.from_numpy(qt)
        
        image_tensor=self.to_tensor(img)   
        
        dwt = image_tensor.new_zeros(1, 8, 8)


        
        if self.model_name=="ascformer":
            ela_np=ela_transform_ascformer(img)
            ela_tensor = torch.from_numpy(ela_np.transpose(2, 0, 1)).float()
        else:
            ela_tensor = image_tensor.new_zeros(1, 8, 8)

        return image_tensor, (self.to_tensor_mask(mask)>0).long(), dct, dwt, ela_tensor, qt


class DocMDetectorConfig(PretrainedConfig):
    """Configuration for document manipulation detection model."""

    model_type = "DocMDetector"

    def __init__(
        self,
        input_size: List[int] = [2560, 2560],
        pretrained_model_name_or_path: Union[str, bytes, os.PathLike] = "",
        data_aug_config =None,
        model_name=None,
        **kwargs,
    ):
        super().__init__()
        self.input_size = input_size
        self.pretrained_model_name_or_path = pretrained_model_name_or_path
        self.data_aug_config =data_aug_config
        self.model_name=model_name



class DocMDetector(PreTrainedModel):
    """High level wrapper for document manipulation detection model."""

    config_class = DocMDetectorConfig
    base_model_prefix = "DocMDetector"

    def __init__(self, config: DocMDetectorConfig):
        super().__init__(config)
        self.config = config
        self.encoder = Encoder(
            model_name=self.config.model_name,
            input_size=self.config.input_size,
            pretrained_model_name_or_path=self.config.pretrained_model_name_or_path,
            data_aug_config = self.config.data_aug_config,
        )

        self.encoder = nn.SyncBatchNorm.convert_sync_batchnorm(self.encoder)
        
