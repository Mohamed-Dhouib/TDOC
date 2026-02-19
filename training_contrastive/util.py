import os
import random
from collections import defaultdict
import torch
from torch.utils.data import Dataset
from transformers.modeling_utils import PreTrainedModel
from pdf2image import convert_from_path
import pandas as pd
import numpy as np
import cv2
import traceback
import sys
from copy import deepcopy
import albumentations as A
import yaml


yaml_path = "data.yaml"
with open(yaml_path, "r") as f:
    config = yaml.safe_load(f)
vert_thresh = config.get("vert_thresh", 1)
vtol = config.get("vtol", 256)
neg_count = config.get("neg_count", 256)
max_positive = config.get("max_positive", 32)
max_w_g = config.get("max_w_g", 1024)
max_h_g = config.get("max_h_g", 64)
pad_to_max_shape = config.get("pad_to_max_shape", False)
number_of_augmented_pos = config.get("number_of_augmented_pos", 20)
augment_rest = config.get("augment_rest", False)
debug = config.get("debug", False)
num_negative_source_csvs = config.get("num_negative_source_csvs", 10)
AUGMENTATION_PIPELINE = A.Compose([
    A.GaussNoise(std_range=(0.03, 0.06), p=1.0),
    A.MotionBlur(blur_limit=(3, 5), p=1.0)
])



class ChangeTextOrBackgroundColor(A.ImageOnlyTransform):
    """Augmentation that shifts foreground or background color."""

    def __init__(self, threshold: int = 10, always_apply=True, p=1.0):
        super().__init__(always_apply, p)
        self.threshold = threshold

    def apply(self, img: np.ndarray, **params) -> np.ndarray:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if random.random()<0.5:
            fg_or_bg_mask = gray <= self.threshold
            shift = np.random.randint(50, 255 - self.threshold - 50, size=3, dtype=np.int16)
        else:
            fg_or_bg_mask = gray > 255-2*self.threshold
            shift = -np.random.randint(50, 255 - 2*self.threshold - 50, size=3, dtype=np.int16)
        out = img.astype(np.int16)
        out[fg_or_bg_mask] += shift
        out = np.clip(out, 0, 255).astype(np.uint8)

        return out

    def get_transform_init_args_names(self):
        return ("threshold",)



class InvertColors(A.ImageOnlyTransform):
    """Augmentation that inverts image colors."""

    def __init__(self, always_apply=False, p=0.5):
        super().__init__(always_apply, p=p)

    def apply(self, img: np.ndarray, **params) -> np.ndarray:
        return 255 - img


def get_altered_crop(
    img: np.ndarray,
    coord_dest: list[int],
    is_blank,
    threshold=15,
) -> np.ndarray:
    """GetAlteredCrop: Generate hard negatives from anchor (Appendix C.1)."""
    try:
        W_d, W_g, H_b, H_h = coord_dest
        orig_crop = img[H_b:H_h, W_d:W_g]

        # small geometric shift
        if random.random() > 0.85 and not is_blank:
            offset = max(1, int((H_h - H_b) / 24))
            offset_x = offset + round(random.random() * offset * 5) * round(random.random())

            orig_w = W_g - W_d
            orig_h = H_h - H_b

            if random.random() < 0.5:
                if random.random() < 0.5:
                    new_crop = img[H_b - offset_x : H_h - offset_x, W_d : W_g]
                else:
                    new_crop = img[H_b - offset_x : H_h, W_d : W_g]
                    new_crop = cv2.resize(new_crop, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
            else:
                if random.random() < 0.5:
                    new_crop = img[H_b + offset_x : H_h + offset_x, W_d : W_g]
                else:
                    new_crop = img[H_b : H_h + offset_x, W_d : W_g]
                    new_crop = cv2.resize(new_crop, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)

            return new_crop

        if random.random() > 0.1:
            augmentations = [
                InvertColors(p=1.0),
                A.RandomBrightnessContrast(brightness_limit=0.02, contrast_limit=0.02, p=1.0),
                A.HueSaturationValue(hue_shift_limit=15, sat_shift_limit=15, val_shift_limit=0, p=1.0),
                A.OneOf([
    A.Blur(blur_limit=(3, 5), p=1.0),
    A.MotionBlur(blur_limit=(3, 5), p=1.0),
    A.GaussianBlur(blur_limit=(3, 5), p=1.0),
    A.MedianBlur(blur_limit=5, p=1.0),
    A.Defocus(radius=(1, 2), alias_blur=(0.2, 0.5), p=1.0),
], p=1.0),
                A.RGBShift(r_shift_limit=2, g_shift_limit=2, b_shift_limit=2, p=1.0),
                A.ChannelShuffle(p=1.0),
                A.ColorJitter(brightness=0.02, contrast=0.02, saturation=0.02, hue=0.05, p=1.0),
                ChangeTextOrBackgroundColor(threshold=40, p=1.0),
            ]
        else:
            augmentations = [
                InvertColors(p=1.0),
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=1.0),
                A.HueSaturationValue(hue_shift_limit=40, sat_shift_limit=40, val_shift_limit=40, p=1.0),
                                A.OneOf([
    A.Blur(blur_limit=(3, 13), p=1.0),
    A.MotionBlur(blur_limit=(3, 13), p=1.0),
    A.GaussianBlur(blur_limit=(3, 13), p=1.0),
    A.MedianBlur(blur_limit=13, p=1.0),
    A.Defocus(radius=(1, 5), alias_blur=(0.2, 0.5), p=1.0),
], p=1.0),
                A.RGBShift(r_shift_limit=20, g_shift_limit=20, b_shift_limit=20, p=1.0),
                A.ChannelShuffle(p=1.0),
                A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.4, p=1.0),
                ChangeTextOrBackgroundColor(threshold=75, p=1.0),
            ]

        n = len(augmentations)

        weights = [2] + [1.0 / k for k in range(2, n + 1)]
        total_w = sum(weights)
        probs = [w / total_w for w in weights]

        k = random.choices(range(1, n+1), weights=probs, k=1)[0]
        selected_augs = random.sample(augmentations, k)
        aug_names = [aug.__class__.__name__ for aug in selected_augs]
        if "HueSaturationValue" in aug_names:
            threshold=round(threshold*1.33)
        elif "OneOf"  in aug_names:
            threshold=round(threshold*1.25)
        transformed = A.Compose(selected_augs)(image=orig_crop)["image"]

        def calculate_l2_distance(image1, image2):
            if image1.shape != image2.shape:
                raise ValueError("Images must have the same dimensions and number of channels")

            image1 = image1.astype(np.float32)
            image2 = image2.astype(np.float32)

            distance = np.sqrt(np.mean((image1 - image2) ** 2))
            return distance

        one_aug=len(aug_names)==1
        total_size=np.ones_like(orig_crop, dtype=bool).sum()
        selected_augs_compose=A.Compose(selected_augs)
        for i in range(1000) :
            if (transformed != orig_crop).sum() / total_size < 0.05 or calculate_l2_distance(transformed, orig_crop) < threshold:
                transformed = selected_augs_compose(image=transformed)["image"]
            else :
                break
        for _ in range(100) :
            if (transformed != orig_crop).sum() / total_size < 0.05 or calculate_l2_distance(transformed, orig_crop) < threshold:
                one_aug=False
                selected_augs = random.sample(augmentations, 1)
                transformed = A.Compose(selected_augs)(image=transformed)["image"]
            else :
                break

        if debug:
            if not ((transformed != orig_crop).sum() / total_size < 0.05 or calculate_l2_distance(transformed, orig_crop) < threshold):
                if random.random() <  0.0001:
                    rand_idx = random.randint(0, 1_000_000)
                    name_prefix = f"{aug_names[0]}_" if one_aug else ""

                    os.makedirs("augmented_crops", exist_ok=True)
                    cv2.imwrite(f"augmented_crops/{name_prefix}{rand_idx}_orig_crop.png", orig_crop)
                    cv2.imwrite(f"augmented_crops/{name_prefix}{rand_idx}_debug_transformed.png", transformed)

        if not ((transformed != orig_crop).sum() / total_size < 0.05 or calculate_l2_distance(transformed, orig_crop) < threshold):
            return transformed
        else :
            return False
    except:
        return False


    
                    
def get_image_from_pdf(pdf_path,page_number) :
    """Convert a PDF page to a BGR numpy image."""
    page_number=int(float(page_number))
   
    image = convert_from_path(pdf_path, first_page=page_number, last_page=page_number)
    numpy_image = np.array(image[0])
    image = cv2.cvtColor(numpy_image, cv2.COLOR_RGB2BGR)
    return image

    


def identify_lines(df,build_using_characters=False,threshold=None):
    """Group text elements into lines based on vertical position."""

    if build_using_characters :
        df = df[df['len_text']==1]
    else :
        df = df[df['len_text'] > 1]
    df = df.sort_values(by='H_b')
    
    lines = []
    current_line = []

    

    current_line_id = 0

    for index, row in df.iterrows():
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
        
def ExtractLineSegments(df,image_width,image_height, build_using_characters=False, add_blank=True):
    """ExtractLineSegments: Extract text/blank segments with line indices (Appendix A)."""

    df = numerise_df(df)
    lines = identify_lines(df, build_using_characters=build_using_characters, threshold=vert_thresh
)
    
    all_groups = [
        group
        for line in lines
        for group in group_chars(line)
    ]

    if len(all_groups) > 10000:
        all_groups = random.sample(all_groups, 10000)

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
            "line_id": group[0]["line_id"],
            "is_blank": False,
        }
        for group in all_groups
    ]
    

    ## add blank regions
    df = pd.DataFrame(new_rows)
    if add_blank:
        single_w = df.loc[df["len_text"] == 1, "Width"]
        avg_w = single_w.mean() if not single_w.empty else 8
        df.attrs['average_width'] = avg_w
    
        single_h = df.loc[df["len_text"] == 1, "Height"]
        avg_h = single_h.mean() if not single_h.empty else 16
        df.attrs['average_height'] = avg_h
        
        candidate_rows = df.sample(frac=1.0, random_state=42)
        synthetic_rows = []
        added_blank=0
        max_blank=max(min(200,len(df)/50),5)
        for _, row in candidate_rows.iterrows():
            box_w = row["W_g"] - row["W_d"]
            box_h = row["H_h"] - row["H_b"]
            len_text = row["len_text"]
            horiz_thresh=6*df.attrs['average_width']
            
            horiz_thresh=min(int(horiz_thresh),int(image_width/3))
    
            horiz_thresh=min(horiz_thresh,int(row["Width"])*3)
    
            if random.random()<0.3 and len_text >3:
                horiz_thresh=max(horiz_thresh,min(row["Width"],int(df.attrs['average_width']*len_text))+5*df.attrs['average_width'])
                horiz_thresh=min(horiz_thresh,15*df.attrs['average_width'])
                horiz_thresh=min(int(horiz_thresh),int(image_width/3))
                            
            if added_blank>=max_blank :
                break
        
            for direction in [-1, 1]:
                break_all=False
                for step in range(1, int(horiz_thresh // box_w)):
                    offset = direction * step * box_w
                    candidate_x1 = row["W_d"] + offset +1
                    candidate_x2 = candidate_x1 + box_w
                
                    if ((candidate_x1 < 0.33*image_width) and offset<0) or ((candidate_x2 > 0.66*image_width) and offset>0):
                        continue
    
                    overlap = df[
                        (df["W_d"] < candidate_x2) & (df["W_g"] > candidate_x1) &
                        (df["H_b"] < row["H_h"]) & (df["H_h"] > row["H_b"])
                    ]
        
                    if overlap.empty:
                        synthetic_rows.append({
                            "text": "+" * len_text,
                            "filename": row["filename"],
                            "W_d": candidate_x1,
                            "W_g": candidate_x2,
                            "H_h": row["H_h"],   
                            "H_b": row["H_b"],   
                            "Width": box_w,
                            "Height": box_h,
                            "ratio": row["ratio"],
                            "len_text": len_text,
                            "line_id": row["line_id"],
                            "is_blank": True,
                        })
                        break_all=True
                        added_blank+=1
                        break
        
                if break_all:
                    break
            if break_all:
                min_vert_gap = max(max(vtol,df.attrs['average_height']*10),int(image_height/4))    
                
                box_w    = row["W_g"] - row["W_d"]
                box_h    = row["H_h"] - row["H_b"]
                y_top    = row["H_b"]
                y_bottom = row["H_h"]
                
                for v_dir in (-1, 1):
                    found_negative = False
                    max_vert_steps = int((image_height - min_vert_gap - box_h) // box_h) + 1
                
                    for v_step in range(1, max_vert_steps):
                        dist = min_vert_gap + (v_step - 1) * box_h
                
                        if v_dir == 1:
                            y1 = y_bottom + dist
                        else:
                            y1 = y_top - dist - box_h
                
                        y2 = y1 + box_h
                
                        if y1 < 0 or y2 > image_height:
                            break
                
                        for h_dir in (-1, 1):
                            if h_dir == -1:
                                max_border_steps = row["W_d"] // box_w
                            else:
                                max_border_steps = (image_width - row["W_g"]) // box_w
                
                            max_thresh_steps = int(horiz_thresh // box_w)
                            max_steps = min(max_border_steps, max_thresh_steps)
                
                            for h_step in range(max_steps, 0, -1):
                                offset = h_dir * h_step * box_w
                                if h_dir == -1:
                                    x2 = row["W_d"] + offset
                                    x1 = x2 - box_w
                                else:
                                    x1 = row["W_g"] + offset
                                    x2 = x1 + box_w
                
                                if x1 < 0 or x2 > image_width:
                                    continue
                
                                overlap = df[
                                    (df["W_d"] < x2) & (df["W_g"] > x1) &
                                    (df["H_b"] < y2) & (df["H_h"] > y1)
                                ]
                                if overlap.empty:
                                    synthetic_rows.append({
                                        "text": "-" * len_text,
                                        "filename": row["filename"],
                                        "W_d": x1, "W_g": x2,
                                        "H_h": y2, "H_b": y1,
                                        "Width": box_w, "Height": box_h,
                                        "ratio": row["ratio"],
                                        "len_text": len_text,
                                        "line_id": row["line_id"],
                                        "is_blank": True,
                                    })
                                    found_negative = True
                                    break
                            if found_negative:
                                break
                        if found_negative:
                            break
                    if found_negative:
                        break
        if synthetic_rows:
            _old_attrs = df.attrs.copy()
    
            df = pd.concat([df, pd.DataFrame(synthetic_rows)], ignore_index=True)
    
            df.attrs = _old_attrs


    return df


def get_positive_pairs(df: pd.DataFrame, image_width: int):
    """Find positive pairs: similar segments on the same line."""

    df = df.rename(columns={"index": "orig_idx"})
    out = defaultdict(list)

    groups = list(df.groupby("line_id", sort=False))

    random.shuffle(groups)

    for line_id, line_df in groups:
        if len(line_df) < 2:
            continue

        grp_cols = ["len_text", "Height", "Width"]
        subgroups = list(line_df.groupby(grp_cols, sort=False))
        random.shuffle(subgroups)
        for (len_text, height, group_width), grp in subgroups:
            if len(grp) < 2:
                continue
            horiz_thresh=df.attrs['average_width'] *6
        
            horiz_thresh=min(int(horiz_thresh),int(image_width/3))

            horiz_thresh=min(horiz_thresh,int(group_width)*3)

            if random.random()<0.3 and len_text >3:
                horiz_thresh=max(horiz_thresh,min(group_width,int(df.attrs['average_width']*len_text))+5*df.attrs['average_width'])
                horiz_thresh=min(horiz_thresh,15*df.attrs['average_width'])
                horiz_thresh=min(int(horiz_thresh),int(image_width/3))

            if horiz_thresh > group_width :


                arr = grp.sort_values("H_b")
                ids = arr["orig_idx"].to_numpy()
                H_bs = arr["H_b"].to_numpy()
                W_ds = arr["W_d"].to_numpy()
                W_gs = arr["W_g"].to_numpy()
    
                n = len(ids)
                for i in range(n - 1):
                    id_i, hb_i, wd_i, wg_i = ids[i], H_bs[i], W_ds[i], W_gs[i]
                    j = i+1
                    while j < n and (H_bs[j] - hb_i) <= vert_thresh:
                        if abs(wd_i - W_ds[j]) < horiz_thresh:
                            if wg_i <= W_ds[j] or W_gs[j] <= wd_i:
                                out[id_i].append(ids[j])
                                if len(out)>=max_positive :
                                    return dict(out)
                                out[ids[j]].append(id_i)
                        j += 1
                        if len(out)>=max_positive :
                            return dict(out)

            
    return dict(out)


def get_df_and_image_from_csv(csv_path,add_blank=True) :
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
    df_image=ExtractLineSegments(df_image,image_width=w,image_height=h,add_blank=add_blank,build_using_characters=True)
    h, w = img_destination.shape[:2]

    df_image=df_image[((df_image['W_g']-df_image['W_d'])>=1) & ((df_image['H_h']-df_image['H_b'])>=1)]

    df_image = df_image[(df_image['W_d'] >= 0) & (df_image['H_b'] >= 0)]

    return img_destination,df_image


def get_contrastive_data(csv_path,cvs_path_2,other_csvs,preprocess_func) :
    """Mine positive/negative pairs for training F_theta (Algorithm 1, Section 3.1)."""
        
    img_destination,df_image=get_df_and_image_from_csv(csv_path)

    img_2,df_2=get_df_and_image_from_csv(cvs_path_2)

    calculated_list=[False for el in other_csvs]

    failed_list=[False for el in other_csvs]
    
    csvs_dict=dict()
    
    h, w = img_destination.shape[:2]
    h_img, w_img = img_destination.shape[:2]
    df_image = df_image.reset_index()
    groups=get_positive_pairs(df_image,w)



    results = []

    df_image = df_image.reset_index(drop=True)

    for center_idx, pos_idxs in groups.items():
        ref = df_image.iloc[center_idx]
        
        coords = [int(ref[k]) for k in ["W_d", "W_g", "H_b", "H_h"]]
        
        is_blank_center=ref["is_blank"] 
        augmented_so_negatives=[get_altered_crop(img_destination,coords,is_blank_center) for i in range(number_of_augmented_pos)]
        augmented_so_negatives = [el for el in augmented_so_negatives if el is not False]
        is_blank_neg_crops=[is_blank_center for el in augmented_so_negatives]

        cy1, cy2 = int(ref["H_b"]), int(ref["H_h"])
        cx1, cx2 = int(ref["W_d"]), int(ref["W_g"])
        center_crop = img_destination[cy1:cy2, cx1:cx2].copy()


     
        chosen_pi = random.choice(pos_idxs)
        o       = df_image.iloc[chosen_pi]
        oy1, oy2 = int(o["H_b"]), int(o["H_h"])
        ox1, ox2 = int(o["W_d"]), int(o["W_g"])
        pos_crop = img_destination[oy1:oy2, ox1:ox2].copy()
        is_blank_pos = o["is_blank"]
        


        ref_ratio = ref["ratio"]  

        h, w = center_crop.shape[:2]
        
        scale = max_h_g / h
        new_w = int(w * scale)
        
        if new_w > max_w_g:
            scale = max_w_g / w

        new_h = int(h * scale)
        new_w = int(w * scale)

        interp = cv2.INTER_CUBIC if scale > 1 else cv2.INTER_AREA
        if scale!=1 :    
            
            
            center_crop = cv2.resize(center_crop, (new_w, new_h), interpolation=interp)
            pos_crop = cv2.resize(pos_crop, (new_w, new_h), interpolation=interp)


        c1 = df_image[(df_image.len_text==ref.len_text) &
                    (~df_image.index.isin([center_idx]+pos_idxs)) &
                    df_image.ratio.between(ref_ratio*0.9, ref_ratio*1.1)]

        c2 = df_2[(df_2.len_text==ref.len_text) &
          df_2.ratio.between(ref_ratio*0.9, ref_ratio*1.1)]



        neg_crops = [
    cv2.resize(el, (new_w, new_h), interpolation=interp)
        
    for el in augmented_so_negatives
]
        
        min_vert_gap = max(max(vtol,df_image.attrs['average_height']*10),int(h_img/4))     
        for src, dfc, img in [("img1", c1, img_destination), ("img2", c2, img_2)]:
            curent_len=0
            per_source_cap = int(neg_count * 0.4)
            for idx, row in dfc.iterrows():
                if len(neg_crops) >= neg_count:
                    break
                if curent_len > per_source_cap :
                    break
                if src=="img1" and (abs(row.H_h - ref.H_b)<min_vert_gap or abs(row.H_b - ref.H_h)<min_vert_gap or (abs(row.H_h - ref.H_h)<min_vert_gap or abs(row.H_b - ref.H_b)<min_vert_gap)):
                    continue
                if src=="img2" and img_destination.shape == img_2.shape and (abs(row.H_h - ref.H_b)<min_vert_gap or abs(row.H_b - ref.H_h)<min_vert_gap or (abs(row.H_h - ref.H_h)<min_vert_gap or abs(row.H_b - ref.H_b)<min_vert_gap)):
                    continue
                y1,y2,x1,x2 = map(int, (row.H_b, row.H_h, row.W_d, row.W_g))
                crop = img[y1:y2, x1:x2]
                crop = cv2.resize(crop, (new_w, new_h), interpolation=interp)
                neg_crops.append(crop)
                is_blank_neg_crops.append(row["is_blank"])
                curent_len+=1

        
        for index_other,el in enumerate(other_csvs):
            if failed_list[index_other]:
                continue
            if len(neg_crops) < neg_count:
                if not calculated_list[index_other]:
                    try :
                        img_4,df_4=get_df_and_image_from_csv(other_csvs[index_other],add_blank=index_other==0)
                        calculated_list[index_other]=True
                        csvs_dict[index_other]=(img_4,df_4)
                    except :
                        print(f"error for other csv number {index_other}")
                        failed_list[index_other]=True
                        continue
                else :
                    img_4,df_4=csvs_dict[index_other]
    
                if index_other==0 :
                    c4 = df_4[(df_4.len_text==ref.len_text) &
                    df_4.ratio.between(ref_ratio*0.9, ref_ratio*1.1)]
                    
                elif index_other==(len(other_csvs)-1):
                    c4 = df_4
                    
                elif index_other==(len(other_csvs)-2):
                    c4 = df_4[(df_4.len_text==ref.len_text)]
                else:
                    c4 = df_4[(df_4.len_text==ref.len_text) &
                    df_4.ratio.between(ref_ratio*0.8, ref_ratio*1.25)]
                
                src, dfc, img = "img4", c4, img_4
                dfc_sorted = dfc.copy()
                eps = 1e-5
                ratios     = dfc_sorted['ratio']
                safe_ratios = ratios.clip(lower=eps)
                
                dfc_sorted['diff'] = np.maximum(
                    safe_ratios / (ref_ratio + eps),
                    (ref_ratio + eps) / safe_ratios
                )
                dfc_sorted = dfc_sorted.sort_values('diff')
                
                for idx, row in dfc_sorted.iterrows():
                    if len(neg_crops) >= neg_count:
                        break
                 
                    y1,y2,x1,x2 = map(int, (row.H_b, row.H_h, row.W_d, row.W_g))
                    crop = img[y1:y2, x1:x2]
                    crop = cv2.resize(crop, (new_w, new_h), interpolation=interp)
                    neg_crops.append(crop)
                    is_blank_neg_crops.append(row["is_blank"])
  

        if len(neg_crops) > neg_count:
            raise RuntimeError("len(neg_crops) > neg_count should not happen at this stage")
 
        elif len(neg_crops) < neg_count:
            needed = neg_count - len(neg_crops)
            n_aug = len(augmented_so_negatives)
            chosen_indexes=[n_aug+random.choice(range(len(neg_crops[n_aug:]))) for _ in range(needed)]

            to_aug = [neg_crops[chosen_indexes[idx]] for idx in range(needed)]
            is_blank_neg_crops.extend([is_blank_neg_crops[chosen_indexes[idx]] for idx in range(needed)])

            if augment_rest:
                augmented = [AUGMENTATION_PIPELINE(image=img)["image"] for img in to_aug]
            else :
                augmented=to_aug

            neg_crops.extend(augmented)

        is_blank_neg_crops=[el or is_blank_center for el in is_blank_neg_crops]
        is_blank_pos = is_blank_pos or is_blank_center 

        assert len(neg_crops) == len(is_blank_neg_crops)


        results.append({
            "center_img":  center_crop,
            "pos_imgs":   pos_crop,
            "neg_imgs":    neg_crops,
            "is_blank_neg": torch.tensor(is_blank_neg_crops, dtype=torch.bool),
            "is_blank_pos" : torch.tensor(is_blank_pos, dtype=torch.bool)
        })

    def pad_to_max_shape_and_transform_to_tensor(img, target_h, target_w,preprocess_func=preprocess_func,pad_to_max_shape=True):
        if not pad_to_max_shape :
            return preprocess_func(img)
        h, w = img.shape[:2]
        pad_h = target_h - h
        pad_w = target_w - w
        top = pad_h // 2
        bottom = pad_h - top
        left = pad_w // 2
        right = pad_w - left
        img=cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])
        return preprocess_func(img)

    del df_image
    del df_2
    
    all_imgs = []
    for entry in results:
        all_imgs += [entry["center_img"]] + [entry["pos_imgs"]] + entry["neg_imgs"]
    max_h = max(im.shape[0] for im in all_imgs)
    max_w = max(im.shape[1] for im in all_imgs)

    for entry in results:
        entry["anchor"] = pad_to_max_shape_and_transform_to_tensor(entry.pop("center_img"), max_h, max_w,preprocess_func=preprocess_func,pad_to_max_shape=pad_to_max_shape)
        
        entry["positive"] = pad_to_max_shape_and_transform_to_tensor(entry.pop("pos_imgs"), max_h, max_w,preprocess_func=preprocess_func,pad_to_max_shape=pad_to_max_shape)
        
        neg_imgs = entry.pop("neg_imgs")
        entry["negatives"] = torch.stack([pad_to_max_shape_and_transform_to_tensor(img, max_h, max_w,preprocess_func=preprocess_func,pad_to_max_shape=pad_to_max_shape) for img in neg_imgs])

    return results


class DocMDetectorDataset(Dataset):
    """Dataset for F_theta contrastive training (Algorithm 1)."""

    def __init__(
        self,
        seg_model: PreTrainedModel,
        split = "train",
        config = None,
    ):
        super().__init__()

        
        seg_model = deepcopy(seg_model)
        seg_model.encoder.model=None

        self.preprocess = seg_model.encoder.prepare_input
        self.split = split

        self.dataset_length = 0
        self.csvs=[]        
        self.datasets_grouped=dict()
        self.group=[]
        current_group=0
        self.datasets_grouped[current_group]=[]
        self.small_datasets=[]
            
        prefix = config.datasets_main_path
        suffix = "/merged_jsons"

        small_datasets=config.small_datasets

        prev_length = 0
        prev_length = len(self.csvs)
        if split == "train":
            
            for dataset, range_val in config.datasets_upsample_factor.items():
                current_group+=1
                self.datasets_grouped[current_group]=[]
                dirpath = prefix + dataset + suffix
                print(dirpath)
                for filename in os.listdir(dirpath):
                    filepath = os.path.join(dirpath, filename)
                    if filename.endswith(".csv"):
                        if range_val>1 :
                            for _ in range(range_val):
                                self.csvs.append(filepath)
                                if dataset in small_datasets:
                                    self.small_datasets.append(self.dataset_length)
                                self.datasets_grouped[current_group].append(self.dataset_length)
                                self.group.append(current_group)
                                self.dataset_length += 1
                        else :
                                if random.random() < range_val:
                                    self.csvs.append(filepath)
                                    if dataset in small_datasets:
                                        self.small_datasets.append(self.dataset_length)
                                    self.datasets_grouped[current_group].append(self.dataset_length)
                                    self.group.append(current_group)
                                    self.dataset_length += 1
                                
                print(dataset.split('/')[-1])
                print("Length: ", len(self.csvs) - prev_length)
                print(f"Final length {len(self.csvs)}")
                prev_length = len(self.csvs)

                
            for key,value in self.datasets_grouped.items():
                assert len(value)>2
        else:
            self.dataset_length = 1
            self.csvs.append("")
            print(len(self.csvs))

        
        print("correctly initialized")

    
  

    def __len__(self) -> int:
        return self.dataset_length

    def __getitem__(self, idx: int):

        retries=0
        while True:
            try :
                csv = self.csvs[idx]
                base_key = csv.replace("black", "")
                
                grp = self.group[idx]
                same_grp = [
                    i for i in self.datasets_grouped[grp]
                    if i != idx and self.csvs[i].replace("black", "") != base_key
                ]
                negative_source_csvs = set()

                if same_grp:
                    negative_source_csvs.add(random.choice(same_grp))
                
                while len(negative_source_csvs) < num_negative_source_csvs:
                    if self.small_datasets and random.random()<0.2:
                        i=random.choice(self.small_datasets)
                    else :
                        i = random.randrange(len(self.csvs))
                    if i == idx or i in negative_source_csvs:
                        continue
                    if self.csvs[i].replace("black", "") == base_key:
                        continue
                    negative_source_csvs.add(i)
                
                negative_source_csvs = [self.csvs[i] for i in negative_source_csvs]
                output = get_contrastive_data(
                    csv, negative_source_csvs[0],negative_source_csvs[1:], preprocess_func=self.preprocess
                )

                return output
            except Exception as e:
                retries+=1
                print(f"[ERROR] Failed to process item at index {idx} of csv {csv} with error: {e}", flush=True)
                traceback.print_exc(file=sys.stdout)
                

                idx = random.randint(0, len(self.csvs) - 1)
                print(f"[RETRY] Trying with a new random index: {idx} for the {retries} time", flush=True) 
