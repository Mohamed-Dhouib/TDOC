import os
import math
import random
from typing import Tuple
import torch
import numpy as np
import pandas as pd
import cv2
import PIL
import albumentations as A
from torch.utils.data import Dataset
from pdf2image import convert_from_path
from torchvision import transforms
import torchvision 
from tqdm import tqdm
from torch.utils.data import DataLoader
from PIL import ImageOps
import argparse
import hashlib
import yaml
from PIL import Image

gt_stats={0:0,1:0}
ill_defined_stats={"normal":0,"manipulated":0}

with open("prepare_data.yaml", "r") as f:
    config = yaml.safe_load(f)

vert_thresh = config.get("vert_thresh", 1)
max_w_g = config.get("max_w_g", 1024)
max_h_g = config.get("max_h_g", 64)
target_n = config.get("target_n", 32)
do_augment = config.get("do_augment", True)
save_some_crops = config.get("save_some_crops", False)
maximum_per_bucket = config.get("maximum_per_bucket", 3)

TOL_param = config.get("TOL_param", 1)
consider_edges = config.get("consider_edges", True)
consider_edges_cropped = config.get("consider_edges_cropped", True)
target_folder_name = config.get("target_folder_name", 'no_name_was_given')
min_component_area = config.get("min_component_area", 4)
num_workers = config.get("num_workers", 10)
datasets_upsample_factor = config.get("datasets_upsample_factor", {})
datasets_main_path = config.get("datasets_main_path", "")
total_jobs = config.get("total_jobs", "")

params = {
    "vert_thresh": vert_thresh,
    "max_w_g": max_w_g,
    "max_h_g": max_h_g,
    "target_n": target_n,
    "do_augment": do_augment,
    "save_some_crops": save_some_crops,
    "maximum_per_bucket": maximum_per_bucket,
    "TOL_param": TOL_param,
    "consider_edges": consider_edges,
    "consider_edges_cropped": consider_edges_cropped,
    "target_folder_name": target_folder_name,
}
print("=== data preprocessing config ===")
for k, v in params.items():
    print(f"{k}: {v}")

class RandomTextColor(A.ImageOnlyTransform):
    """Augmentation that shifts foreground text color."""

    def __init__(self, threshold: int = 10, always_apply=False, p=1.0):
        super().__init__(always_apply, p=p)
        self.threshold = threshold

    def apply(self, img: np.ndarray, **params) -> np.ndarray:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        fg_mask = gray <= self.threshold
    
        shift = np.random.randint(50, 140, size=(3), dtype=int)
    
        out = img.astype(int)
        out[fg_mask] += shift
        out = np.clip(out, 0, 255).astype(np.uint8)

        return out


class InvertColors(A.ImageOnlyTransform):
    """Augmentation that inverts image colors."""

    def __init__(self, always_apply=False, p=0.5):
        super().__init__(always_apply, p=p)

    def apply(self, img: np.ndarray, **params) -> np.ndarray:
        return 255 - img


photometric_list = [
    A.RandomBrightnessContrast(brightness_limit=0.4, contrast_limit=0.4, p=0.25),
    A.HueSaturationValue(hue_shift_limit=40, sat_shift_limit=40, val_shift_limit=40, p=0.25),
    A.RGBShift(r_shift_limit=40, g_shift_limit=40, b_shift_limit=40, p=0.25),
    A.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2, p=0.25),
    A.ChannelShuffle(p=0.2),
    RandomTextColor(threshold=40, p=0.2),
    InvertColors(p=0.2),


    A.Blur(blur_limit=(3, 7), p=0.15),
    A.MotionBlur(blur_limit=(3,7), p=0.15),
    A.GaussianBlur(blur_limit=(3, 7), p=0.15),
    A.MedianBlur(blur_limit=7, p=0.15),
    A.Defocus(radius=(1, 2), alias_blur=(0.2, 0.5), p=0.15),
    
    A.ImageCompression(quality_range=[10,85], p=0.2),  
    A.Sharpen(
        alpha=(0.2, 0.5),
        lightness=(0.5, 1.0),
        method="kernel",
        p=0.1
    ),    
    A.GaussNoise(std_range=(0.05, 0.1), p=0.1),   
    A.ISONoise(color_shift=(0.01, 0.02), intensity=(0.01, 0.05), p=0.1),
    A.MultiplicativeNoise(multiplier=(0.975, 1.025), per_channel=True, p=0.1),]


photometric_list_positive = [
    A.RandomBrightnessContrast(brightness_limit=0.4, contrast_limit=0.4, p=0.25),
    A.HueSaturationValue(hue_shift_limit=40, sat_shift_limit=40, val_shift_limit=40, p=0.25),
    A.RGBShift(r_shift_limit=40, g_shift_limit=40, b_shift_limit=40, p=0.25),
    A.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2, p=0.25),
    A.ChannelShuffle(p=0.2),
    RandomTextColor(threshold=40, p=0.2),
    InvertColors(p=0.2),

    A.Blur(blur_limit=(3, 31), p=0.15),
    A.MotionBlur(blur_limit=(3,31), p=0.15),
    A.GaussianBlur(blur_limit=(3, 31), p=0.15),
    A.MedianBlur(blur_limit=31, p=0.15),
    A.Defocus(radius=(1, 11), alias_blur=(0.2, 0.5), p=0.15),
    
    A.ImageCompression(quality_range=[10,85], p=0.2),  
    A.Sharpen(
        alpha=(0.2, 0.5),
        lightness=(0.5, 1.0),
        method="kernel",
        p=0.1
    ),    
    A.GaussNoise(std_range=(0.05, 0.1), p=0.1),   
    A.ISONoise(color_shift=(0.01, 0.02), intensity=(0.01, 0.05), p=0.1),
    A.MultiplicativeNoise(multiplier=(0.975, 1.025), per_channel=True, p=0.1)]


def augment_crop_and_strips(crop, top_bottom, left_right, positive=False):
    """Apply photometric augmentations to crop and edge strips."""
    if positive:
        lst = photometric_list_positive[:]
    else:
        lst = photometric_list[:]   
    random.shuffle(lst)     
    lst.append(A.ToGray(p=0.01))     
    photometric = A.Compose(lst)
    aug_crop = photometric(image=crop)["image"]
    aug_tb = photometric(image=top_bottom)["image"]
    aug_lr = photometric(image=left_right)["image"]

    return aug_crop, aug_tb, aug_lr

def is_bad_noexpand_from_crop(
    gray: np.ndarray,
    TOL: int = 1,
    edges: bool = True,
    invert_mask: bool = True,
    min_component_area_var: int = 4,
) -> Tuple[bool, np.ndarray]:
    """
    Decide 'cut' using only the crop region.

    Returns:
      is_cut (bool): True if any foreground CC (by Otsu) touches crop border (within TOL) and is not fully inside.
      mask (uint8): 255 where foreground (text) is, 0 where background is.
    """

    # Foreground=255 mask:
    # - invert_mask=True  => dark text becomes 255 (THRESH_BINARY_INV)
    # - invert_mask=False => bright text becomes 255 (THRESH_BINARY)
    thresh_flag = cv2.THRESH_BINARY_INV if invert_mask else cv2.THRESH_BINARY
    _, mask = cv2.threshold(gray, 0, 255, thresh_flag + cv2.THRESH_OTSU)

    Hc, Wc = mask.shape[:2]

    # Connected components on the foreground mask (255=FG)
    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)

    is_cut = False
    for i in range(1, num_labels):
        x, y, w, h, area = stats[i]
        if area < min_component_area_var:
            continue

        x2 = x + w
        y2 = y + h

        if edges:
            touches = (x <= TOL) or (y <= TOL) or (x2 >= Wc - TOL) or (y2 >= Hc - TOL)
        else:
            touches = (x <  TOL) or (y <  TOL) or (x2 >  Wc - TOL) or (y2 >  Hc - TOL)

        fully_inside = (x >= TOL and y >= TOL and x2 <= Wc - TOL and y2 <= Hc - TOL)

        if touches and not fully_inside:
            is_cut = True
            break

    return is_cut
    
def is_bad(image,orig_x1,orig_x2,orig_y1,orig_y2,TOL=1,edges=True, invert_mask=True):
    """Check if crop cuts through foreground using expanded region."""
    H_img, W_img = image.shape[:2]

    h = orig_y2 - orig_y1

    x1_pad = max(0, orig_x1 -  max(int(h/2),8))
    x2_pad = min(W_img, orig_x2 + max(int(h/2),8))
    y1_pad = max(0, orig_y1 - max(int(h/2),8))
    y2_pad = min(H_img, orig_y2 + max(int(h/2),8))

    expanded_crop = image[y1_pad:y2_pad, x1_pad:x2_pad].copy()
    gray = cv2.cvtColor(expanded_crop, cv2.COLOR_BGR2GRAY)
    if invert_mask:
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU) #black is foreground
    else:
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU) #white is foreground
    

    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)

    crop=cv2.cvtColor(image[orig_y1:orig_y2, orig_x1:orig_x2].copy(), cv2.COLOR_BGR2GRAY)
    is_cut=is_bad_noexpand_from_crop(crop,TOL=TOL,edges=edges,min_component_area_var=min_component_area,invert_mask=invert_mask)

    for i in range(1, num_labels):
        x_cc, y_cc, w_c, h_c, area = stats[i]
        if area < min_component_area:
            continue

        x_abs = x_cc + x1_pad
        y_abs = y_cc + y1_pad
        x_end = x_abs + w_c
        y_end = y_abs + h_c
        if edges:
            intersects_x = (x_abs <= orig_x2) and (x_end >= orig_x1)
            intersects_y = (y_abs <= orig_y2) and (y_end >= orig_y1)
        else:
            intersects_x = (x_abs < orig_x2) and (x_end > orig_x1)
            intersects_y = (y_abs < orig_y2) and (y_end > orig_y1)

        fully_inside = (
            x_abs >= orig_x1 + TOL and
            x_end <= orig_x2 - TOL and
            y_abs >= orig_y1 + TOL and
            y_end <= orig_y2 - TOL
        )

        if intersects_x and intersects_y and not fully_inside:
            is_cut = True
            return is_cut
    return is_cut

            
def get_image_from_pdf(pdf_path,page_number) :
    """Convert a PDF page to a BGR numpy image."""
    page_number=int(float(page_number))
   
    image = convert_from_path(pdf_path, first_page=page_number, last_page=page_number)
    numpy_image = np.array(image[0])
    image = cv2.cvtColor(numpy_image, cv2.COLOR_RGB2BGR)
    return image

def identify_lines(df,build_using_characters=False,threshold=1):
    """Group text elements into lines based on vertical position."""

    if build_using_characters :
        df = df[df['len_text']==1]
    else :
        df = df[df['len_text'] > 1]
    df = df.sort_values(by='H_b')
    
    lines = []
    current_line = []

    

    current_line_id = 0

    for _, row in df.iterrows():
        if len(current_line)==0 :
            row['line_id']=current_line_id
            current_line = [row]
            current_y = row['H_b']
            current_h = row['H_h']
            
        elif abs(row['H_b'] - current_y) <= threshold and abs(row['H_h'] - current_h) <= threshold:
            row['line_id']=current_line_id
            current_line.append(row)
            current_y = row['H_b']
            current_h = row['H_h']
        else:
            lines.append(current_line)
            current_line_id+=1
            row['line_id']=current_line_id
            current_line = [row]
            current_y = row['H_b']
            current_h = row['H_h']
            
    
    if current_line:
        lines.append(current_line)
    
    return lines

def group_chars(line):
    """Generate all contiguous subgroups of characters in a line."""
    
    line = sorted(line, key=lambda x: x['W_d'])

    groups = [
    line[i:j]
    for i in range(len(line))
    for j in range(i + 1, len(line) + 1)
]

    return groups


def numerise_df(df_image) :
    """Convert coordinate columns to numeric types."""
    df_image["ratio"] = pd.to_numeric(df_image["ratio"], errors='coerce')
    df_image['Width'] = df_image['Width'].astype(int) 
    df_image['Height'] = df_image['Height'].astype(int)
    df_image['W_d'] = df_image['W_d'].astype(int) 
    df_image['W_g'] = df_image['W_g'].astype(int)
    df_image['H_h'] = df_image['H_h'].astype(int) 
    df_image['H_b'] = df_image['H_b'].astype(int) 


    cols_to_drop = ["bg_color", "fg_color"]
    
    df_image.drop(columns=cols_to_drop, errors="ignore", inplace=True)

    return df_image

def ExtractLineSegments(df, build_using_characters=False):
    """ExtractLineSegments: Build text segments from characters (Appendix A)."""
    df = numerise_df(df)

    lines = identify_lines(df, build_using_characters=build_using_characters, threshold=vert_thresh)

    all_groups = [
        group
        for line in lines
        for group in group_chars(line)
    ]

    if len(all_groups) > 20000:
        all_groups = random.sample(all_groups, 20000)

    new_rows = [
        {
            **{
                "text": (
                    "".join(w["text"] for w in group)
                    if build_using_characters
                    else " ".join(w["text"] for w in group)
                ),
                "filename": group[0]["filename"],
                "W_d": min(w["W_d"] for w in group),
                "W_g": max(w["W_g"] for w in group),
                "H_b": min(w["H_b"] for w in group),
                "H_h": max(w["H_h"] for w in group),
            },
            **{
                "Width": max(w["W_g"] for w in group) - min(w["W_d"] for w in group),
                "Height": max(w["H_h"] for w in group) - min(w["H_b"] for w in group),
                "ratio": round(
                    (max(w["H_h"] for w in group) - min(w["H_b"] for w in group)) /
                    (max(w["W_g"] for w in group) - min(w["W_d"] for w in group)),
                    2
                ),
            },
            "len_text": (
                len("".join(w["text"] for w in group))
                if build_using_characters
                else len(" ".join(w["text"] for w in group))
            ),
            "line_id": group[0]["line_id"]
        }
        for group in all_groups
    ]

    df = pd.DataFrame(new_rows)


    return df


def get_df_and_image_from_csv(csv_path) :
    """Load image and extract line segments from CSV metadata."""
    df_image = pd.read_csv(csv_path)
    try :
        page_num=df_image.iloc[0]["page"]
    except :
        page_num=False
    pdf_path=df_image.iloc[0]["filename"]


    if pdf_path.endswith(".pdf") :
        img_destination=get_image_from_pdf(pdf_path,page_num)
    else :
        img_destination=cv2.imread(pdf_path)

    h, w = img_destination.shape[:2]
    df_image = df_image[(df_image['W_d'] >= 0) & (df_image['H_b'] >= 0) &(df_image['W_g'] <= w) & (df_image['H_h'] <= h)]

    df_image["line"] = False

    df_image['text'] = df_image['text'].astype(str)

    h, w = img_destination.shape[:2]
    df_image=ExtractLineSegments(df_image,build_using_characters=True)
    h, w = img_destination.shape[:2]

    df_image=df_image[((df_image['W_g']-df_image['W_d'])>=1) & ((df_image['H_h']-df_image['H_b'])>=1)]

    df_image = df_image[(df_image['W_d'] >= 0) & (df_image['H_b'] >= 0)]

    return img_destination,df_image

def get_data(csv_path, preprocess_func):
    """Generate training samples for G_theta bounding box quality (Section 3.2)."""

    img_destination, df_image = get_df_and_image_from_csv(csv_path)

    img_h, img_w = img_destination.shape[:2]
    all_crops = []


    def sample_width_bins(df,max_w_g,max_h_g,target_n,seed=42):
    
        scale = max_h_g / df["Height"]

        widths = (df["Width"] * scale).astype(int).to_numpy()
        
        max_w_g=max_w_g-64
        bin_max = max_w_g // 64
        bins   = np.minimum(widths // 64, bin_max)
    
        bin_indices = [
        np.where(bins == b)[0].tolist()
        for b in range(bin_max + 1)
    ]
        small_bins = list(range(bin_max))
        small_count = sum(len(bin_indices[b]) for b in small_bins)
        big_count   = len(bin_indices[bin_max])
    
        big_cap = min(big_count, small_count // 60, target_n // 60)
    
        rng = random.Random(seed)
        selected = []
    
        if small_count + big_cap <= target_n:
            for b in small_bins:
                selected.extend(bin_indices[b])
            if big_cap:
                selected.extend(rng.sample(bin_indices[bin_max], big_cap))
        else:
            need_small = target_n - big_cap
    
            nb = len(small_bins)
            base = need_small // nb
            rem  = need_small % nb
            desired = [base + (1 if i < rem else 0) for i in range(nb)]
    
            actual = [0]*nb
            leftover = 0
            for i, b in enumerate(small_bins):
                have = len(bin_indices[b])
                want = desired[i]
                take = want if want <= have else have
                actual[i] = take
                leftover += max(0, want - have)
    
            capacity = [len(bin_indices[b]) - actual[i] for i,b in enumerate(small_bins)]
            i = 0
            while leftover > 0 and any(c > 0 for c in capacity):
                if capacity[i] > 0:
                    actual[i]   += 1
                    capacity[i] -= 1
                    leftover    -= 1
                i = (i + 1) % nb
    
            for i, b in enumerate(small_bins):
                take = actual[i]
                if take:
                    selected.extend(rng.sample(bin_indices[b], take))
            if big_cap:
                selected.extend(rng.sample(bin_indices[bin_max], big_cap))
    
        rng.shuffle(selected)
        selected = selected[:target_n]
        
    
        return df.iloc[selected].sample(frac=1, random_state=seed).reset_index(drop=True)

    def transform_and_collect(df: pd.DataFrame, image: np.ndarray):

        df=sample_width_bins(df,max_w_g,max_h_g,target_n*200)
        H_img, W_img = image.shape[:2]

        well_defined_crops = {
            str(index):0 for index in range(1,26)
        }
        
        ill_defined_crops = {
            str(index):0 for index in range(1,26)
        }

        ill_defined_crops_base = {
            str(index):0 for index in range(1,26)
        }
        
        for _, r in df.iterrows():
            num_characters = len("".join(r["text"]))
            if num_characters>25:
                num_characters=25
            bucket=str(num_characters)

            if ill_defined_crops[bucket]>=maximum_per_bucket and well_defined_crops[bucket]>=maximum_per_bucket:
                print(f"skipping condition 1 {bucket} {ill_defined_crops[bucket],well_defined_crops[bucket]}")
                continue 

            orig_y1, orig_y2 = int(r["H_b"]), int(r["H_h"])
            orig_x1, orig_x2 = int(r["W_d"]), int(r["W_g"])
            if orig_y2 <= orig_y1 or orig_x2 <= orig_x1:
                continue

            
  
            is_cut=is_bad(image,orig_x1,orig_x2,orig_y1,orig_y2,TOL=TOL_param,edges=consider_edges)
            y1, y2 = orig_y1, orig_y2
            x1, x2 = orig_x1, orig_x2
   
            if is_cut:
                is_cut_2=is_bad(image,orig_x1,orig_x2,orig_y1,orig_y2,TOL=TOL_param,edges=consider_edges, invert_mask=False)
                if is_cut_2:
                    if ill_defined_crops[bucket]<maximum_per_bucket and ill_defined_crops_base[bucket]<round(maximum_per_bucket/2) and ill_defined_stats["normal"]<=(ill_defined_stats["manipulated"]+3):
                        gt =1
                        ill_defined_stats["normal"]+=1
                        ill_defined_crops_base[bucket]+=1
                    else:
                        continue
                else:
                    gt = 0 #here generally means background is darker and the crop is well defined
                    fg_darker=False
            else:
                gt = 0
                fg_darker=True


            if gt!=1 and ill_defined_crops[bucket]<maximum_per_bucket and (random.random() < 0.9 or well_defined_crops[bucket]>=maximum_per_bucket):

                sides = ["top", "bottom", "left", "right"]
                k = random.randint(1, 4)
                chosen_sides = random.sample(sides, k)
                ops = {s: None for s in sides}
                for s in chosen_sides:
                    ops[s] = random.choice(["pad", "crop"])

                def rand_off():
                    return 1 + round(random.random()) + int(round(random.random() * 18) * round(random.random() * 0.51))

                off = {s: rand_off() for s in sides}

                if random.random() < 0.5:
                    first_op = list(ops.values())[0]
                    first_off = list(off.values())[0]
                    ops = {s: first_op for s in sides}
                    off = {s: first_off for s in sides}



                new_x1, new_x2 = x1, x2
                new_y1, new_y2 = y1, y2

                # ---- PAD per side ----
                if ops["left"] == "pad":
                    new_x1 = max(0, new_x1 - off["left"])
                if ops["right"] == "pad":
                    new_x2 = min(img_w, new_x2 + off["right"])
                if ops["top"] == "pad":
                    new_y1 = max(0, new_y1 - off["top"])
                if ops["bottom"] == "pad":
                    new_y2 = min(img_h, new_y2 + off["bottom"])

                # ---- CROP per side ----
                if ops["left"] == "crop" and (new_x2 - new_x1) > off["left"] * 1.1:
                    new_x1 += off["left"]
                if ops["right"] == "crop" and (new_x2 - new_x1) > off["right"] * 1.1:
                    new_x2 -= off["right"]
                if ops["top"] == "crop" and (new_y2 - new_y1) > off["top"] * 1.1:
                    new_y1 += off["top"]
                if ops["bottom"] == "crop" and (new_y2 - new_y1) > off["bottom"] * 1.1:
                    new_y2 -= off["bottom"]

                if new_y2 <= new_y1 or new_x2 <= new_x1:
                    new_y1, new_y2 = orig_y1, orig_y2
                    new_x1, new_x2 = orig_x1, orig_x2


                is_bad_bool=is_bad(image,new_x1,new_x2,new_y1,new_y2,TOL=TOL_param,edges=consider_edges_cropped, invert_mask=fg_darker)
                if is_bad_bool:   
                    y1, y2 = new_y1, new_y2
                    x1, x2 = new_x1, new_x2
                    gt = 1
                    ill_defined_stats["manipulated"]+=1
            
            if gt==0 and well_defined_crops[bucket]>=maximum_per_bucket:
                continue 
            if gt==1:
                crop_dbg = image[y1:y2, x1:x2].copy()
                gray_dbg = cv2.cvtColor(crop_dbg, cv2.COLOR_BGR2GRAY)

                #one pixel test, check if it is true
                if not (is_bad_noexpand_from_crop(gray_dbg,TOL=TOL_param,edges=consider_edges_cropped,min_component_area_var=1,invert_mask=True) and is_bad_noexpand_from_crop(gray_dbg,TOL=TOL_param,edges=consider_edges_cropped,min_component_area_var=1,invert_mask=False)):
                    continue

            gt_stats[int(gt)]+=1
            crop = image[y1:y2, x1:x2].copy()
            
            h, w = crop.shape[:2]
            
            scale = max_h_g / h
            new_w = int(w * scale)
            
            if new_w > max_w_g:
                scale = max_w_g / w

            if scale!=1 :    
                new_h = int(h * scale)
                new_w = int(w * scale)
                
                interp = cv2.INTER_CUBIC if scale > 1 else cv2.INTER_AREA
                
                crop = cv2.resize(crop, (new_w, new_h), interpolation=interp)

            target_pixels =9
            
            orig_border = max(1, math.ceil(target_pixels / scale))
            
            H_img, W_img = image.shape[:2]
            top_b    = min(orig_border, y1)
            bottom_b = min(orig_border, H_img - y2)
            left_b   = min(orig_border, x1)
            right_b  = min(orig_border, W_img - x2)
            
            strips = {
                "top":    image[y1 - top_b : y1,       x1:x2].copy(),
                "bottom": image[y2         : y2+bottom_b, x1:x2].copy(),
                "left":   image[y1:y2,      x1 - left_b : x1].copy(),
                "right":  image[y1:y2,      x2         : x2+right_b].copy(),
            }
            
            for name, patch in strips.items():
                ph, pw = patch.shape[:2]
                th, tw = int(ph * scale), int(pw * scale)
                if scale!=1 : 
                    resized = cv2.resize(patch, (tw, th), interpolation=interp)
                else:
                    resized = patch
            
                if name in ("top", "bottom"):
                    actual = th
                else:
                    actual = tw
            
                if actual > target_pixels:
                    if name == "top":
                        resized = resized[-target_pixels:, :]
                    elif name == "bottom":
                        resized = resized[:target_pixels, :]
                    elif name == "left":
                        resized = resized[:, -target_pixels:]
                    elif name == "right":
                        resized = resized[:, :target_pixels]
           
            
                elif actual < target_pixels:
                    pad_amt = target_pixels - actual
                    if name == "top":
                        resized = cv2.copyMakeBorder(resized, pad_amt, 0, 0, 0,
                                                     cv2.BORDER_CONSTANT, value=[255,255,255])
                    elif name == "bottom":
                        resized = cv2.copyMakeBorder(resized, 0, pad_amt, 0, 0,
                                                     cv2.BORDER_CONSTANT, value=[255,255,255])
                    elif name == "left":
                        resized = cv2.copyMakeBorder(resized, 0, 0, pad_amt, 0,
                                                     cv2.BORDER_CONSTANT, value=[255,255,255])
                    else:
                        resized = cv2.copyMakeBorder(resized, 0, 0, 0, pad_amt,
                                                     cv2.BORDER_CONSTANT, value=[255,255,255])
            
                strips[name] = resized


   
            if save_some_crops and random.random()<0.01:
                os.makedirs(str(gt),exist_ok=True)
                cv2.imwrite(os.path.join(str(gt),f"new_{int(random.random()*2000)}.png"),crop)
                
            top_bottom = np.concatenate([np.flipud(strips["top"]),    strips["bottom"]], axis=1)
            left_right = np.concatenate([np.fliplr(strips["left"]),   strips["right"] ], axis=0)
            
            try:
                if do_augment and random.random() < 0.5:
                    crop, top_bottom, left_right = augment_crop_and_strips(crop, top_bottom, left_right, positive=gt==1)
                    if save_some_crops and random.random() < 0.01:
                        os.makedirs(str(gt), exist_ok=True)
                        cv2.imwrite(os.path.join(str(gt), f"new_aug_{int(random.random() * 2000)}.png"), crop)
            except Exception as e:
                print(f"[ERROR] During aug: {e}")


  
            all_crops.append((crop, gt,top_bottom,left_right))

            if gt==0:
                well_defined_crops[bucket]+=1
            else :
                ill_defined_crops[bucket]+=1

            if len(all_crops) >= target_n:
                print(f"ill_defined_crops {ill_defined_crops}")
                print(f"well_defined_crops {well_defined_crops}")
                return True
        return False

    df_image = df_image.sample(frac=1, random_state=42).reset_index(drop=True)
    transform_and_collect(df_image, img_destination)




    results = []
    for crop_img, gt,top_bottom, left_right in all_crops:
        
        tensor = preprocess_func(crop_img,pad=False) 
        
        top_bottom=preprocess_func(top_bottom,pad=False)
        left_right=preprocess_func(left_right,pad=False)
        
        results.append([tensor, top_bottom, left_right, gt])

    print(f"gt_stats is {gt_stats}")

    print(f"ill defined stats are {ill_defined_stats}")

    print(f"got {len(results)} crops")
    return results


def prepare_input(img,pad=True ):
    """Convert image to normalized tensor with optional padding."""
    to_tensor=transforms.Compose(
            [
                transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=(0.485, 0.455, 0.406), std=(0.229, 0.224, 0.225)),
            ]
        )
    
    if type(img)==str :
        img=cv2.imread(img)
     
    if not isinstance(img, PIL.Image.Image) :
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
    if pad:
        w, h = img.size
        pad_w = max(0, 16 - w)
        pad_h = max(0, 16 - h)
        if pad_w or pad_h:
            img = ImageOps.expand(img, border=(0, 0, pad_w, pad_h), fill=0)

    return to_tensor(img)
    
class DocMDetectorDataset(Dataset):
    """Dataset for generating G_theta training samples."""

    def __init__(
        self,
        mod=None,
    ):
        super().__init__()

        self.succeded=0

        self.failed=0

        self.mod=mod

        print(f"mod is {self.mod}")
        self.preprocess = prepare_input
      
        self.dataset_length = 0
        self.csvs=[]                              
            
        prefix = datasets_main_path
        suffix = "/merged_jsons"

        prev_length = 0
        prev_length = len(self.csvs)
     
        
        for dataset, range_val in datasets_upsample_factor.items():
            dirpath = prefix + dataset + suffix
            print(dirpath)
            for filename in os.listdir(dirpath):
                filepath = os.path.join(dirpath, filename)
                if filename.endswith(".csv"):
                    if range_val>1 :
                        for _ in range(range_val):
                            self.csvs.append(filepath)
                            self.dataset_length += 1
                    else :
                        if random.random() < range_val:
                            self.csvs.append(filepath)
                            self.dataset_length += 1

            print(dataset.split('/')[-1])
            print("Length: ", len(self.csvs) - prev_length)
            print(f"Final length {len(self.csvs)}")
            prev_length = len(self.csvs)

    
  

    def __len__(self) -> int:
        return self.dataset_length

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        try :
            csv=self.csvs[idx]
            stem = os.path.splitext(os.path.basename(csv))[0]
            number_rx_output=number_rx(stem)
            assert number_rx_output>=0
            if (number_rx_output)% total_jobs == int(self.mod):
        
                output=get_data(csv,preprocess_func=self.preprocess)
                if len(output)> 1 :
                    fn = os.path.join(target_folder_name, f"{idx}_{stem}.pt")
                    print(f"Saved result from {csv} in {fn}")
                    torch.save(output, fn)
                    self.succeded+=1
                else :
                    self.failed+=1
                    print(f"The resulting crops lenght is zero for {csv}")
                        
        except :
            self.failed+=1
            print(f"Current preprocessing setp has failed has failed for {csv}")
            print(f"stats are: succeded {self.succeded}, failed {self.failed}")



def number_rx(filename: str) -> int:
    """Hash filename to integer for job partitioning."""
    return int(hashlib.sha1(filename.encode()).hexdigest(), 16)
    
def main():
    """Run parallel data preparation for G_theta training."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--mod", type=int, required=True,
                        help="Remainder (0–31) for filename index mod 32")
    args = parser.parse_args()
    

    mod=args.mod
    
    cv2.setNumThreads(0)
    cv2.ocl.setUseOpenCL(False)
    os.makedirs(target_folder_name,exist_ok=True)
    ds = DocMDetectorDataset(mod=mod)

    def identity_collate(batch):
        return batch
    
    loader = DataLoader(
        ds,
        batch_size=1,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=identity_collate,
        shuffle=True,          
    )

    for _ in tqdm(loader):  
        pass
        

if __name__ == "__main__":
    main()
