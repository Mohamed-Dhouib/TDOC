import ast
import math
import json
import os
import torch.multiprocessing as mp
import random
from typing import List
import torch
from torch.utils.data import Dataset, DataLoader
from pdf2image import convert_from_path
import pandas as pd
import numpy as np
import cv2
from skimage.filters import threshold_sauvola
import string
import sys
import torch.nn.functional as F
sys.path.append("../training_contrastive")
from model import DocMDetector, DocMDetectorConfig
from sconf import Config
import uuid
sys.path.append("../training_crop_quality")
from model_cropped import DocMDetector as MCrop 
from model_cropped import DocMDetectorConfig as Mconfig
from tqdm import tqdm
import yaml
from contextlib import nullcontext
import traceback
import argparse
from collections import defaultdict
import time
import hashlib
import gc

def parse_args():
    parser = argparse.ArgumentParser(
        description='Run on full dataset or just a single index')
    parser.add_argument(
        '--job_index',
        type=int,
        default=None,
        help='Shard index used to attribute files across parallel jobs')
    parser.add_argument(
        '--config',
        type=str,
        default="config.yaml",
        help='Config path')
    parser.add_argument(
        '--max_jobs',
        type=int,
        default=1,
        help='Total number of shards for hash-based file assignment')
    parser.add_argument(
        '--resume_map',
        type=str,
        default=None,
        help='Path to JSON mapping {csv_path: left_count}; if set, we only generate these and repeat each CSV left_count times'
    )
    return parser.parse_args()

total=[0,0]

args = parse_args()
config_name=args.config
with open(config_name, "r") as f:
    config = yaml.safe_load(f)

cv2.setLogLevel(3)

max_w_g = config.get("max_w_g", 1024)
max_h_g = config.get("max_h_g", 64)
write_crops = config.get("write_crops", False)
threshold_crop_quality = config.get("threshold_crop_quality", 0.5)
threshold_crop_similarity = config.get("threshold_crop_similarity", 0.8)
mixed_prec = config.get("mixed_prec", False)
use_only_image = config.get("use_only_image", False)
datasets_main_path = config.get("datasets_main_path", "")
output_folder = config.get("output_folder", "")
num_workers = config.get("num_workers", 4)
pixel_budget=  config.get("pixel_budget", 1024*64)
probability_deactivate_crop_quality_network            = config.get("probability_deactivate_crop_quality_network", 0.25)
probability_deactivate_crop_similarity_network = config.get("probability_deactivate_crop_similarity_network", 0.)
probability_only_cm_sp            = config.get("probability_only_cm_sp", 0.75)
probability_insertion            = config.get("probability_insertion", 0.01)
probability_target_blank_area    = config.get("probability_target_blank_area", 0.025)
probability_inpainting           = config.get("probability_inpainting", 0.0025) 
coverage_probability_target_not_character = config.get("coverage_probability_target_not_character", 0.15)
coverage_probability_target_character     = config.get("coverage_probability_target_character", 0.05)
probability_splicing             = config.get("probability_splicing", 0.5)
ratio_diff_tolerance_copy_move   = config.get("ratio_diff_tolerance_copy_move", 0.05)
bbox_expansion_probability = config.get("bbox_expansion_probability", 0.05)
datasets_upsample_factor = config.get("datasets_upsample_factor", {})  
datasets_to_expand = config.get("datasets_to_expand", [])  
grouped_csvs_folder= config.get("grouped_csvs_folder", "")
max_number_of_sequences= config.get("max_number_of_sequences", 10000)
max_splicing_tries= config.get("max_splicing_tries", 10)
splicing_color_threshold = config.get("splicing_color_threshold", 5)
max_candidate_crops_splicing = config.get("max_candidate_crops_splicing", 5)
force_using_stripes_proba = config.get("force_using_stripes_proba", 0.1)
probability_no_manipulation= config.get("probability_no_manipulation", 0.1)
proba_limit_segments_length= config.get("proba_limit_segments_length", 0.25)
max_manipulations_per_doc= config.get("max_manipulations_per_doc",5)
save_in_a_single_folder= config.get("save_in_a_single_folder",False)
get_segments_proba= config.get("get_segments_proba",0)

assert all(config.get(k) for k in ["config_path_crop_embed", "config_path_crop_quality"]), "Missing required config key"

config_path_crop_embed, config_path_crop_quality = config["config_path_crop_embed"], config["config_path_crop_quality"]


cropped_stats={}



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if mixed_prec and device.type == "cuda":
    amp_ctx = lambda: torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
else:
    amp_ctx = nullcontext

_HERSHEY_CHARS = set(
    string.ascii_letters +
    string.digits +
    string.punctuation +
    " "
)

hershey_fonts = (
    cv2.FONT_HERSHEY_SIMPLEX,
    cv2.FONT_HERSHEY_DUPLEX,
    cv2.FONT_HERSHEY_COMPLEX,
)

def color_distance(color_a, color_b):
    """Calculate average channel-wise distance between two colors."""
    if color_a is None or color_b is None:
        return float("inf")
    try:
        diffs = [abs(int(a) - int(b)) for a, b in zip(color_a, color_b)]
    except (TypeError, ValueError):
        return float("inf")
    return sum(diffs) / max(len(diffs), 1)

def is_hershey_ascii_only(text: str) -> bool:
    """Check if text contains only Hershey-renderable ASCII chars."""
    return all(ch in _HERSHEY_CHARS for ch in text.lower())


def prepare_crop_data(self, img_np, global_image, r, verify=True, high_quality=True, disable_crop_quality_network=False, df_image=None, df_image_save=None):
    """Prepare crop data for G_theta inference (Section 3.2)."""
    if not high_quality or disable_crop_quality_network:
        return None  # Skip preparation
    
    if verify:
        idx = r.name
        
        assert idx in df_image_save.index, f"Missing row {idx!r} in df_image_save"
        
        if idx in df_image.index:
            a = df_image.at[idx, 'cropped']
            b = df_image_save.at[idx, 'cropped']
            assert a == b, (
                f"Mismatched 'cropped' at index {idx!r}: "
                f"df_image={a!r} vs df_image_save={b!r}"
            )
            
        if df_image_save.at[idx, 'cropped'] == 0:
            return {'early_return': False}
        if df_image_save.at[idx, 'cropped'] == 1:
            return {'early_return': True}
    
    h, w = img_np.shape[:2]
    
    scale = max_h_g / h
    new_w = int(w * scale)
    
    if new_w > max_w_g:
        scale = max_w_g / w

    if scale != 1:
        new_h = int(h * scale)
        new_w = int(w * scale)
        
        interp = cv2.INTER_CUBIC if scale > 1 else cv2.INTER_AREA
        
        img_np_resized = cv2.resize(img_np, (new_w, new_h), interpolation=interp)
    else:
        img_np_resized = img_np

    img_tensor, image_mask = self.crop_model.encoder.prepare_input(img_np_resized, pad=True)
    
    if not globals().get("curr_use_only_image", False):
        y1, y2 = map(int, (r["H_b"], r["H_h"]))
        x1, x2 = map(int, (r["W_d"], r["W_g"]))

        orig_h_val = y2-y1
        resized_h = img_np_resized.shape[0]
        stripe_scale = resized_h / orig_h_val

        target_pixels = 9
        
        orig_border = max(1, math.ceil(target_pixels / stripe_scale))
        
        H_img, W_img = global_image.shape[:2]
        top_b    = min(orig_border, y1)
        bottom_b = min(orig_border, H_img - y2)
        left_b   = min(orig_border, x1)
        right_b  = min(orig_border, W_img - x2)
        
        strips = {
            "top":    global_image[y1 - top_b : y1,       x1:x2].copy(),
            "bottom": global_image[y2         : y2+bottom_b, x1:x2].copy(),
            "left":   global_image[y1:y2,      x1 - left_b : x1].copy(),
            "right":  global_image[y1:y2,      x2         : x2+right_b].copy(),
        }
        
        interp = cv2.INTER_CUBIC if stripe_scale > 1 else cv2.INTER_AREA
        
        for name, patch in strips.items():
            ph, pw = patch.shape[:2]
            th, tw = int(ph * stripe_scale), int(pw * stripe_scale)
            if stripe_scale != 1:
                try:
                    resized = cv2.resize(patch, (tw, th), interpolation=interp)
                except:
                    print("resizing failed considering corpped")
                    return {'early_return': True, 'error': True}
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
                else:
                    assert False
        
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

        top_bottom = np.concatenate([np.flipud(strips["top"]), strips["bottom"]], axis=1)
        left_right = np.concatenate([np.fliplr(strips["left"]), strips["right"]], axis=0)

        top_bottom_tensor, tb_mask = self.crop_model.encoder.prepare_input(top_bottom, pad=False)
        left_right_tensor, lr_mask = self.crop_model.encoder.prepare_input(left_right, pad=False)
  
    else:
        top_bottom_tensor, left_right_tensor, tb_mask, lr_mask = None, None, None, None
    
    # Return all tensors on CPU and metadata
    return {
        'img_tensor': img_tensor,
        'image_mask': image_mask,
        'top_bottom_tensor': top_bottom_tensor,
        'left_right_tensor': left_right_tensor,
        'tb_mask': tb_mask,
        'lr_mask': lr_mask,
        'img_np': img_np_resized,  # Store the resized version for write_crops
        'verify': verify,
        'idx': r.name if verify else None
    }


def batch_infer_crop_quality(self, prepared_batch):
    """Run batched G_theta bounding box quality inference (Section 3.2)."""
    # Filter out early returns and None values
    valid_items = []
    valid_indices = []
    results = []
    
    for idx, item in enumerate(prepared_batch):
        if item is None or 'early_return' in item:
            results.append(item.get('early_return', False) if item else False)
        else:
            valid_items.append(item)
            valid_indices.append(idx)
            results.append(None)  # Placeholder
    
    del prepared_batch
    
    if not valid_items:
        return results, []

    # Stack tensors - add batch dimension
    img_tensors = torch.stack([item['img_tensor'] for item in valid_items]).to(device, non_blocking=True)
    image_masks = torch.stack([item['image_mask'] for item in valid_items]).to(device, non_blocking=True)
    
    # Clean up CPU tensors from valid_items
    for item in valid_items:
        del item['img_tensor']
        del item['image_mask']
    
    if not globals().get("curr_use_only_image", False):
        def pad_feat(tensor, target_h, target_w):
            if tensor.shape[-2] == target_h and tensor.shape[-1] == target_w:
                return tensor
            pad_h = target_h - tensor.shape[-2]
            pad_w = target_w - tensor.shape[-1]
            return F.pad(tensor, (0, pad_w, 0, pad_h))

        def pad_feat_mask(mask_tensor, target_h, target_w):
            if mask_tensor.shape[-2] == target_h and mask_tensor.shape[-1] == target_w:
                return mask_tensor
            padded = mask_tensor.new_full((target_h, target_w), False)
            padded[: mask_tensor.shape[-2], : mask_tensor.shape[-1]] = mask_tensor
            return padded

        tb_feats = [item['top_bottom_tensor'] for item in valid_items]
        lr_feats = [item['left_right_tensor'] for item in valid_items]
        tb_masks_list = [item['tb_mask'] for item in valid_items]
        lr_masks_list = [item['lr_mask'] for item in valid_items]

        max_tb_h = max(t.shape[-2] for t in tb_feats) #will be 9
        max_tb_w = max(t.shape[-1] for t in tb_feats)
        max_lr_h = max(t.shape[-2] for t in lr_feats)
        max_lr_w = max(t.shape[-1] for t in lr_feats) #will be 9

        tb_feats = [pad_feat(t, max_tb_h, max_tb_w) for t in tb_feats] 
        lr_feats = [pad_feat(t, max_lr_h, max_lr_w) for t in lr_feats]
        tb_masks = [pad_feat_mask(m, max_tb_h, max_tb_w) for m in tb_masks_list]
        lr_masks = [pad_feat_mask(m, max_lr_h, max_lr_w) for m in lr_masks_list]

        del tb_masks_list
        del lr_masks_list

        top_bottom_tensors = torch.stack(tb_feats).to(device, non_blocking=True)
        left_right_tensors = torch.stack(lr_feats).to(device, non_blocking=True)
        tb_masks = torch.stack(tb_masks).to(device, non_blocking=True)
        lr_masks = torch.stack(lr_masks).to(device, non_blocking=True)

        del tb_feats
        del lr_feats

        tb_masks = tb_masks.any(dim=1, keepdim=True)
        lr_masks = lr_masks.any(dim=2, keepdim=True)
        
        # Clean up CPU tensors from valid_items
        for item in valid_items:
            if item.get('top_bottom_tensor') is not None:
                del item['top_bottom_tensor']
                del item['left_right_tensor']
                del item['tb_mask']
                del item['lr_mask']
    else:
        top_bottom_tensors = None
        left_right_tensors = None
        tb_masks = None
        lr_masks = None
    
    # Run inference
    with torch.no_grad():
        with amp_ctx():
            logits = self.crop_model.encoder(
                img_tensors,
                top_bottom_tensors,
                left_right_tensors,
                image_masks,
                tb_masks,
                lr_masks,
                use_only_image=globals().get("curr_use_only_image", False)
            )
    
    # Get probabilities
    all_probs = torch.softmax(logits, dim=-1)[:, 1]
    
    del logits
    del img_tensors
    del image_masks
    if top_bottom_tensors is not None:
        del top_bottom_tensors
        del left_right_tensors
        del tb_masks
        del lr_masks
    
    # Map back to original positions
    valid_idx = 0
    for i in range(len(results)):
        if results[i] is None:
            results[i] = all_probs[valid_idx]
            valid_idx += 1
    
    del all_probs
    
    return results, valid_items


def is_cropped_fn(self, img_np, global_image, r, df_image=None, df_image_save=None, TOL=1, edges=True, verify=True, high_quality=True, disable_crop_quality_network=False):
    """Use G_theta to check if crop is ill-defined (Section 3.2)."""
    if not high_quality or disable_crop_quality_network:
        return False
    
    # Prepare data
    prepared = prepare_crop_data(self, img_np, global_image, r, verify, high_quality, disable_crop_quality_network, df_image, df_image_save)
    
    if prepared is None:
        return False
    
    if 'early_return' in prepared:
        return prepared['early_return']
    
    # Run inference
    probs, valid_items = batch_infer_crop_quality(self, [prepared])
    prob = probs[0]

    prob_float = float(prob.detach().cpu().item())
    is_cropped = prob_float > threshold_crop_quality

    if is_cropped:
        cropped_stats[dataset_name][1] += 1
    else:
        cropped_stats[dataset_name][0] += 1

    if write_crops and random.random() < 0.01:
        base_folder = "crops_predic"
        bucket = min(20, max(1, math.ceil(prob_float * 20)))
    
        target_dir = os.path.join(base_folder, str(bucket))
        os.makedirs(target_dir, exist_ok=True)

        
        if mixed_prec:
            prob_float = prob_float + 0.01 * random.random()
    
        filename = f"prob_{prob_float:.4f}.png"
    
        cv2.imwrite(os.path.join(target_dir, filename), prepared['img_np'])

    if verify:
        idx = prepared['idx']
        df_image_save.at[idx, "cropped"] = int(is_cropped)
        
        if idx in df_image.index:
            df_image.at[idx, "cropped"] = int(is_cropped)
    
    return is_cropped


def is_cropped_fn_batch(self, batch_data, high_quality=True, disable_crop_quality_network=False):
    """Batch G_theta bounding box quality check (Section 3.2)."""
    if not high_quality or disable_crop_quality_network:
        return [False] * len(batch_data)
    
    results = [False] * len(batch_data)
    max_pixels = max(1, max_h_g * max_w_g)
    chunk_size = max(1, int(0.2 * pixel_budget) // max_pixels)

    for chunk_start in range(0, len(batch_data), chunk_size):
        chunk_end = min(len(batch_data), chunk_start + chunk_size)
        prepared_batch = []
        index_map = []

        for idx in range(chunk_start, chunk_end):
            img_np, global_image, r, df_image, df_image_save, verify = batch_data[idx]
            prepared = prepare_crop_data(
                self,
                img_np,
                global_image,
                r,
                verify,
                high_quality,
                disable_crop_quality_network,
                df_image,
                df_image_save,
            )
            prepared_batch.append(prepared)
            index_map.append(idx)

        if not prepared_batch:
            continue

        probs, _ = batch_infer_crop_quality(self, prepared_batch)

        for local_idx, (prepared, prob) in enumerate(zip(prepared_batch, probs)):
            batch_idx = index_map[local_idx]

            if prepared is None:
                results[batch_idx] = False
                continue

            if 'early_return' in prepared:
                results[batch_idx] = prepared['early_return']
                continue

            if isinstance(prob, torch.Tensor):
                prob_float = float(prob.detach().cpu().item())
            else:
                prob_float = float(prob)

            is_cropped = prob_float > threshold_crop_quality

            if is_cropped:
                cropped_stats[dataset_name][1] += 1
            else:
                cropped_stats[dataset_name][0] += 1

            if write_crops and random.random() < 0.01:
                base_folder = "crops_predic"
                bucket = min(20, max(1, math.ceil(prob_float * 20)))
                target_dir = os.path.join(base_folder, str(bucket))
                os.makedirs(target_dir, exist_ok=True)

                dump_prob = prob_float
                if mixed_prec:
                    dump_prob = dump_prob + 0.01 * random.random()

                filename = f"prob_{dump_prob:.4f}.png"
                cv2.imwrite(os.path.join(target_dir, filename), prepared['img_np'])

            if prepared['verify']:
                idx = prepared['idx']
                img_np, global_image, r, df_image, df_image_save, verify = batch_data[batch_idx]
                df_image_save.at[idx, "cropped"] = int(is_cropped)

                if idx in df_image.index:
                    df_image.at[idx, "cropped"] = int(is_cropped)

            results[batch_idx] = is_cropped

    return results

    


def choose_crop(
    self,
    source_patches: List[np.ndarray],
    crop_dest: np.ndarray,
    apply_softmax: bool = False,
    high_quality=True,
    is_blank=False,
    disable_crop_similarity_network=False,
    target_fg_color=None,
    target_bg_color=None,
    source_colors= None,
    sample=True
) -> np.ndarray:
    """Select best source patch using F_theta similarity S(x,u) (Section 3.1)."""
    if disable_crop_similarity_network:
        if not source_patches:
            raise ValueError("source_patches must contain at least one patch when similarity network is disabled")
        if source_colors is not None:
            try:
                fallback_color = np.array([-100, -100, -100], dtype=np.int32)

                def parse_color_or_fallback(raw_value, label):#remove the label part
                    try:
                        parsed = parse_color(raw_value)
                        if parsed is None or len(parsed) < 3:
                            raise ValueError("parsed color must have at least three components")
                        return np.array([int(parsed[0]), int(parsed[1]), int(parsed[2])], dtype=np.int32)
                    except Exception as exc:
                        # print(f"[WARN] Failed to parse {label} {raw_value!r}: {exc}; using {fallback_color.tolist()}")
                        return fallback_color.copy()

                target_fg = parse_color_or_fallback(target_fg_color, "target fg color")
                target_bg = parse_color_or_fallback(target_bg_color, "target bg color")

                best_idx = None
                best_distance = float("inf")

                if source_colors:
                    assert len(source_colors) == len(source_patches),"source_patches and source_colors should have the same length"
                    for idx, meta in enumerate(source_colors):
                        if not isinstance(meta, dict): #to remove shouldn't be the case
                            print(f"[WARN] Source color metadata at index {idx} is not a dict: {meta}")
                            continue
                        cand_fg_raw = meta.get("fg_color")
                        cand_bg_raw = meta.get("bg_color")

                        cand_fg = parse_color_or_fallback(cand_fg_raw, "candidate fg color")
                        cand_bg = parse_color_or_fallback(cand_bg_raw, "candidate bg color")

                        distance_components = []

                        distance_components.append(float(np.mean(np.abs(cand_fg - target_fg))))
                        distance_components.append(float(np.mean(np.abs(cand_bg - target_bg))))

                        avg_distance = sum(distance_components) / len(distance_components)
                        if avg_distance < best_distance:
                            best_distance = avg_distance
                            best_idx = idx

                if best_idx is not None:
                    # print(f"[INFO] Chose patch index {best_idx} with color distance {best_distance:.2f}")
                    return source_patches[best_idx], True
            except Exception as e:
                print(f"Best color sampling failed, choosing randomly, error is {e}")
        return random.choice(source_patches), True

    all_images = source_patches + [crop_dest]

    resized_images = []
    for img_np in all_images:
        h, w = img_np.shape[:2]
        
        scale = max_h_g / h
        new_w = int(w * scale)
        
        if new_w > max_w_g:
            scale = max_w_g / w

        if scale!=1 :    
            new_h = int(h * scale)
            new_w = int(w * scale)
            
            interp = cv2.INTER_CUBIC if scale > 1 else cv2.INTER_AREA
            
            img_np = cv2.resize(img_np, (new_w, new_h), interpolation=interp)
        resized_images.append(img_np)
    
    del all_images
    all_images = resized_images
    del resized_images

    cpu_tensors = [
        self.meta_model.encoder.prepare_input(img_np).unsqueeze(0)
        for img_np in all_images
    ]

    del all_images

    batch_cpu = torch.cat(cpu_tensors, dim=0)

    del cpu_tensors

    batch_gpu = batch_cpu.to(device, non_blocking=True)

    N = batch_gpu.shape[0]
    H, W = int(batch_gpu.shape[-2]), int(batch_gpu.shape[-1])


    if N*H*W<pixel_budget:
        del batch_cpu
        with torch.no_grad():
            with amp_ctx():
                embeddings = self.meta_model.encoder(batch_gpu)
    else:
        print("chunking forward pass")
        pixels_per_sample = H * W
        chunk_bs = max(1, (pixel_budget // pixels_per_sample))
        emb_chunks = []
        with torch.no_grad():
            for start in range(0, N, chunk_bs):
                end = min(N, start + chunk_bs)
                chunk_cpu = batch_cpu[start:end]
                chunk_gpu = chunk_cpu.to(device, non_blocking=True)
                with amp_ctx():
                    chunk_emb = self.meta_model.encoder(chunk_gpu)
                emb_chunks.append(chunk_emb.detach().cpu())
                del chunk_gpu, chunk_emb, chunk_cpu
                if device.type == "cuda":
                    torch.cuda.empty_cache()

        del batch_cpu
        embeddings = torch.cat(emb_chunks, dim=0).to(device, non_blocking=True)
        del emb_chunks

    del batch_gpu

    source_embs = embeddings[:-1]
    dest_emb    = embeddings[-1:].detach()

    del embeddings

    if is_blank:
        half = source_embs.shape[1] // 2
        scores = torch.matmul(source_embs[:, :half], dest_emb[:, :half].T).squeeze(1)
    else:
        scores = torch.matmul(source_embs, dest_emb.T).squeeze(1)/2

    del source_embs
    del dest_emb

    if apply_softmax:
        scores = F.softmax(scores, dim=0)

    #sampling
    effective_threshold = threshold_crop_similarity if high_quality else (threshold_crop_similarity * 0.5)
    qualified_indices = (scores > effective_threshold).nonzero(as_tuple=False).flatten().tolist()

    if sample and qualified_indices:
        qualified_scores = scores[qualified_indices]
        weights = (10*(qualified_scores - effective_threshold) + 0.8)**2.5
        weight_list = weights.cpu().tolist()
        sampled_idx = random.choices(qualified_indices, weights=weight_list, k=1)[0]
        best_idx = sampled_idx
    else:
        best_idx = torch.argmax(scores).item()

    best_score = scores[best_idx]


    if write_crops and len(source_patches) > 0 and ((best_score>threshold_crop_similarity) or random.random()<0.1):
        scores_cpu = scores.cpu().numpy()

        N = len(source_patches)
        idx_sorted = np.argsort(-scores_cpu)
        
        if N <= 6:
            chosen_indices = idx_sorted.tolist()
        else:
            top4    = idx_sorted[:2].tolist()
            bot4    = idx_sorted[-2:].tolist()
            mid_center = N // 2
            start_mid = max(0, mid_center - 1)
            end_mid   = start_mid + 1
            if end_mid > N:
                end_mid   = N
                start_mid = N - 2
            mid4 = idx_sorted[start_mid:end_mid].tolist()
        
            chosen_indices = top4 + mid4 + bot4

        h_dest, w_dest = crop_dest.shape[:2]

        font         = cv2.FONT_HERSHEY_SIMPLEX
        font_scale   = 0.3
        thickness    = 1
        text_margin  = 4

        sample_text = "0.0000"
        (text_w, text_h), _ = cv2.getTextSize(sample_text, font, font_scale, thickness)

        block_h = h_dest + text_h + text_margin

        dest_block = np.ones((block_h, w_dest, 3), dtype=np.uint8) * 255
        dest_block[0:h_dest, 0:w_dest] = crop_dest


        blocks = [dest_block]
        for i in chosen_indices:
            src_patch = source_patches[i]
            resized = cv2.resize(src_patch, (w_dest, h_dest), interpolation=cv2.INTER_LINEAR)

            block = np.ones((block_h, w_dest, 3), dtype=np.uint8) * 255
            block[0:h_dest, 0:w_dest] = resized

            score_val = scores_cpu[i]
            score_str = f"{score_val*10:.1f}"

            text_size, _ = cv2.getTextSize(score_str, font, font_scale, thickness)
            tx, ty = text_size
            x_text = (w_dest - tx) // 2
            y_text = h_dest + text_h
            cv2.putText(
                block,
                score_str,
                (x_text, y_text),
                font,
                font_scale,
                (0, 0, 0),
                thickness,
                lineType=cv2.LINE_AA
            )

            blocks.append(block)

        margin = 10
        margin_block = np.ones((block_h, margin, 3), dtype=np.uint8) * 255

        strip = blocks[0]
        for blk in blocks[1:]:
            strip = np.hstack([strip, margin_block, blk])

        out_dir = "candidate_crop_meta_scores"
        os.makedirs(out_dir, exist_ok=True)

        rnd_name = f"{uuid.uuid4().hex}.png"
        out_path = os.path.join(out_dir, rnd_name)
        cv2.imwrite(out_path, strip)

    if not high_quality:
        found_adequate_crop=best_score>(threshold_crop_similarity*0.5)
    else:
        found_adequate_crop=best_score>threshold_crop_similarity

    del scores
    del best_score

    return source_patches[best_idx],found_adequate_crop


def find_blank_regions_helper(img_width, img_height, width, height, df_image_save):
    """
    Return all (x1, x2, y1, y2) regions of size (height×width) that do NOT overlap
    any box in df_image_save. The image is conceptually divided into a grid of
    cells each of size (width × height); any cell that intersects an existing
    (W_d, W_g, H_b, H_h) box is removed from valid_boxes.

    Args:
        img_width (int): full image width
        img_height (int): full image height
        width (int): desired crop width
        height (int): desired crop height
        df_image_save (DataFrame): each row must have 'W_d','W_g','H_b','H_h'
    Returns:
        List[Tuple[int,int,int,int]]: list of (x1, x2, y1, y2) for each valid cell
    """
    num_cols = max(1, (img_width + width - 1) // width)
    num_rows = max(1, (img_height + height - 1) // height)

    total_cells = num_cols * num_rows
    valid_boxes = set(range(total_cells))

    for _, row in df_image_save.iterrows():
        x1, x2 = int(row['W_d']), int(row['W_g'])
        y1, y2 = int(row['H_b']), int(row['H_h'])

        col_start = x1 // width
        col_end   = (x2 - 1) // width
        row_start = y1 // height
        row_end   = (y2 - 1) // height

        col_start = max(0, min(col_start, num_cols - 1))
        col_end   = max(0, min(col_end,   num_cols - 1))
        row_start = max(0, min(row_start, num_rows - 1))
        row_end   = max(0, min(row_end,   num_rows - 1))

        for r in range(row_start, row_end + 1):
            base = r * num_cols
            for c in range(col_start, col_end + 1):
                valid_boxes.discard(base + c)

    valid_regions = []
    for cell in sorted(valid_boxes):
        r = cell // num_cols
        c = cell % num_cols
        x1 = c * width
        y1 = r * height
        x2 = min(x1 + width, img_width)
        y2 = min(y1 + height, img_height)
        valid_regions.append((x1, x2, y1, y2))

    del valid_boxes

    return valid_regions



def find_blank_regions(img, width, height, df_image_save, max_regions=2000):
    """
    Returns up to `max_regions` blank-region crops of size (height×width) from `img`.
    """
    img_height, img_width, _ = img.shape
    valid_regions = find_blank_regions_helper(img_width, img_height, width, height, df_image_save)

    if len(valid_regions) > max_regions:
        valid_regions = random.sample(valid_regions, max_regions)

    crops = []
    for (x1, x2, y1, y2) in valid_regions:
        crops.append(img[y1:y2, x1:x2])

    del valid_regions

    return crops


def collide(box1, box2):
    """Check if two bounding boxes overlap."""
    wd1, wg1, hb1, hh1 = box1
    wd2, wg2, hb2, hh2 = box2
    return not (wd1 > wg2 or wd2 > wg1 or hb1 > hh2 or hb2 > hh1)


def get_blank_area(data, max_w, max_h,img_destination, max_attempts=200,get_closest_textbox=False):
    """
    Tries up to `max_attempts` times to place a box of the same size as a randomly
    chosen existing box into a blank region (no overlap). Returns either:
      (new_coords, text, original_coords)
    or (False, False, False) if all attempts collide.
    """
    boxes = [
        (row['W_d'], row['W_g'], row['H_b'], row['H_h'], row['text'], row['fg_color'])
        for _, row in data.iterrows()
    ]
    for _ in range(max_attempts):
        chosen_box = random.choice(boxes)
        box_w = chosen_box[1] - chosen_box[0]
        box_h = chosen_box[3] - chosen_box[2]
        if is_hershey_ascii_only(chosen_box[4]):
            for _ in range(max_attempts):
                new_wd = random.randint(0, max_w - box_w)
                new_wg = new_wd + box_w
                new_hb = random.randint(0, max_h - box_h)
                new_hh = new_hb + box_h
                new_box = (new_wd, new_wg, new_hb, new_hh)
        
                if not any(collide(new_box, b[:4]) for b in boxes):
                    window_size = 13
                    k = 0.5
                    r = 128        
                    crop= img_destination[new_box[2]:new_box[3], new_box[0]:new_box[1]]
                    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                    thresh_map = threshold_sauvola(gray, window_size=window_size, k=k, r=r)
                    fg_mask = gray < thresh_map
                    if not fg_mask.any():
                                            
                        if get_closest_textbox:
                            cx, cy = (new_wd + new_wg) / 2, (new_hb + new_hh) / 2
                            same_size = [
                                b for b in boxes
                                if (b[1] - b[0] == box_w) and (b[3] - b[2] == box_h)
                            ]
                            def dist2(b):
                                bx, by = (b[0] + b[1]) / 2, (b[2] + b[3]) / 2
                                return (bx - cx)**2 + (by - cy)**2
                            closest = min(same_size, key=dist2)
                            if is_hershey_ascii_only(closest[4]):
                                return new_box, closest[4], list(closest[:4]), closest[5]
                            else :
                                return new_box, chosen_box[4], list(closest[:4]), closest[5]
                        else:
                            return new_box, chosen_box[4], list(chosen_box)[:4], None

    return False, False, False, False


def fit_text_to_image_opencv(
    image,
    text,
    font,
    initial_font_scale=2.0,
    ratio=0.8,
    margin_w=0.0,
    margin_h=0.0,
    margin_map={0.0:0},
    min_scale=0.05,
    max_iter=30,
    max_expand=16,
):
    """Find font scale to fit text in image with margins."""
    h, w = image.shape[:2]
    if not isinstance(text, str) or not text.strip() or w <= 1 or h <= 1:
        scale = max(min_scale, 1e-3)
        thickness = max(1, int(round(scale * max(float(ratio), 1e-6))))
        return float(scale), int(thickness), 0, 0

    margin_w = max(0.0, min(0.45, float(margin_w)))
    margin_h = max(0.0, min(0.45, float(margin_h)))
    margin_w=max(round(h*margin_w), margin_map[margin_w]) #margin calculated always according to h
    margin_h=max(round(h*margin_h), margin_map[margin_h])


    ratio  = float(max(ratio, 1e-6))
    target_w = max(1, int(w -margin_w)) 
    target_h = max(1, int(h - margin_h))

    def fits(scale: float):
        t = max(1, int(round(scale * ratio)))
        (tw, th), baseline = cv2.getTextSize(text, font, scale, t)        
        total_h = th + baseline                                           
        return (tw <= target_w and total_h <= target_h), t                


    low  = max(min_scale, 1e-3)
    high = max(float(initial_font_scale), low)
    ok, _ = fits(high)
    if ok:
        upper = high
        for _ in range(max_expand):
            nxt = high * 2.0
            ok2, _ = fits(nxt)
            if not ok2:
                upper = nxt
                break
            high = nxt
        else:
            upper = high * 2.0
        low = high
    else:
        upper = high

    best_scale = low
    best_thickness = max(1, int(round(low * ratio)))
    for _ in range(max_iter):
        mid = 0.5 * (low + upper)
        ok, t = fits(mid)
        if ok:
            best_scale, best_thickness = mid, t
            low = mid
        else:
            upper = mid
        if upper - low < 1e-3:
            break

    best_scale *= 0.98
    best_thickness = max(1, int(round(best_scale * ratio)))
    return float(best_scale), int(best_thickness), margin_w, margin_h


def _clip_rgb(v):
    """Clamp value to valid RGB range [0, 255]."""
    return 0 if v < 0 else (255 if v > 255 else int(v))


def generate_fg_variants(fg_rgb):
    """Generate color variants for text rendering."""
    r, g, b = (int(fg_rgb[0]), int(fg_rgb[1]), int(fg_rgb[2]))
    if r == g == b:
        deltas = (-4,-3, -2, -1, 0, 1, 2, 3, 4)
        return [(_clip_rgb(r+d), _clip_rgb(g+d), _clip_rgb(b+d)) for d in deltas]
    vals = (-4, -2, 0,2 ,4)
    variants = []
    for dr in vals:
        for dg in vals:
            for db in vals:
                variants.append((_clip_rgb(r+dr), _clip_rgb(g+dg), _clip_rgb(b+db)))
    seen, out = set(), []
    random.shuffle(variants)
    for c in variants:
        if c not in seen:
            seen.add(c)
            out.append(c)
            if len(out)>8:
                break

    if (0, 0, 0) not in seen:
        out.append((0, 0, 0))
    return out

def randomize_text_like(s):
    """Replace each alphanumeric char with a random char of the same type."""
    out = []
    for ch in s:
        if 'a' <= ch <= 'z':                    
            out.append(random.choice(string.ascii_lowercase))
        elif 'A' <= ch <= 'Z':
            out.append(random.choice(string.ascii_uppercase))
        elif ch.isdigit():
            out.append(random.choice(string.digits))
        else:
            out.append(ch)                       
    return ''.join(out)


def write_on_image(
    patch_unpainted,
    text,
    fg_color_list,
    ratio_candidates=(0.7, 0.8, 0.9),
    margin_candidates=[0, 0.05, 0.1,0.2],
    vary_color=True,
    random_text_probability=0.5
):
    """Render text on image patch for insertion."""
    try:
        fg = [int(el) for el in fg_color_list]
        if vary_color:
            fg_variants = generate_fg_variants(fg)
        else :
            fg_variants = [fg]
    except Exception:
        gray = int(random.random() * 50)
        fg = [gray, gray, gray]
        fg_variants= [fg]

    margin_map={el:2**index-1 for index,el in enumerate(margin_candidates)}

    if random.random()<random_text_probability:
        text=randomize_text_like(text)
    
    out_images = []
    for font in hershey_fonts:
        for ratio in ratio_candidates:
            for margin_w in margin_candidates[:3]:
                for margin_h in margin_candidates:
                    img = patch_unpainted.copy()
                    font_scale, thickness, margin_w_pixels, margin_h_pixels = fit_text_to_image_opencv(img, text, font, ratio=float(ratio), margin_w=margin_w, margin_h=margin_h, margin_map=margin_map)
                    (sz_w, sz_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)  
                    total_h = sz_h + baseline                                                    
                    x = (img.shape[1] - sz_w) // 2                                               
                    y = (img.shape[0] - total_h) // 2 + sz_h  

                    x_y=[(x,y),(x,y-margin_h_pixels),(x,y+margin_h_pixels)]    
                    for x,y in x_y:         
                        x=int(x)
                        y=int(y)                       
                        for col in fg_variants:
                            img = patch_unpainted.copy()
                            cv2.putText(img, text, (x, y), font, font_scale, col, thickness, cv2.LINE_AA)
                            out_images.append(img)
    return out_images, text


def get_image_from_pdf(pdf_path,page_number) :
    """Convert a PDF page to a BGR numpy image."""
    page_number=int(float(page_number))
   
    image = convert_from_path(pdf_path, first_page=page_number, last_page=page_number)
    numpy_image = np.array(image[0])
    image = cv2.cvtColor(numpy_image, cv2.COLOR_RGB2BGR)
    return image



def identify_lines(df, threshold=2,build_using_characters=False):
    """Group OCR boxes into text lines (used by ExtractLineSegments, Appendix A)."""
    if build_using_characters :
        df = df[df['len_text']==1]
    else :
        df = df[df['len_text'] > 1]
    df = df.sort_values(by='H_b')
    
    lines = []
    current_line = []
    
    for index, row in df.iterrows():
        if len(current_line)==0 :
            current_line = [row]
            current_y = row['H_b']
            current_h = row['H_h']
            
        elif abs(row['H_b'] - current_y) <= threshold and abs(row['H_h'] - current_h) <= threshold:
            current_line.append(row)
        else:
            lines.append(current_line)
            current_line = [row]
            current_y = row['H_b']
            current_h = row['H_h']
            
    
    if current_line:
        lines.append(current_line)
    
    return lines

def group_words_and_eliminate_chars(line, max_len=None):
    """Group characters into contiguous word segments."""
    line = sorted(line, key=lambda x: x['W_d'])

    groups_by_len = defaultdict(list)
    n = len(line)
    max_window = max_len or n

    for i in range(n):
        upper = min(n + 1, i + max_window + 1)
        for j in range(i + 2, upper):
            group = line[i:j]
            groups_by_len[j - i].append(group)

    return dict(groups_by_len)

def parse_color(x):
    """Parse color from string or return as list."""
    if isinstance(x, str):
        return ast.literal_eval(x)
    elif isinstance(x, (list, tuple, np.ndarray)):
        return list(x)
    else:
        raise ValueError(f"Unsupported color type: {type(x)}")

def numerise_df(df_image) :
    """Convert coordinate and color columns to numeric types."""
    df_image["ratio"] = pd.to_numeric(df_image["ratio"], errors='coerce')
    df_image['Width'] = df_image['Width'].astype(int)
    df_image['Height'] = df_image['Height'].astype(int)
    df_image['W_d'] = df_image['W_d'].astype(int)
    df_image['W_g'] = df_image['W_g'].astype(int)
    df_image['H_h'] = df_image['H_h'].astype(int)
    df_image['H_b'] = df_image['H_b'].astype(int)
    df_image["bg_color"] = df_image["bg_color"].apply(parse_color)
    df_image["fg_color"] = df_image["fg_color"].apply(parse_color)
    return df_image

def process_groups(all_groups,build_using_characters):
    """Convert character/word groups into segment rows."""
    new_rows = []
    append_row = new_rows.append
    for group in all_groups:
        first = group[0]
        texts = [w["text"] for w in group]
        if build_using_characters:
            text_joined = "".join(texts)
        else:
            text_joined = " ".join(texts)

        len_text = len(text_joined)

        w_ds = [w["W_d"] for w in group]
        w_gs = [w["W_g"] for w in group]
        h_bs = [w["H_b"] for w in group]
        h_hs = [w["H_h"] for w in group]

        min_w_d = min(w_ds)
        max_w_g = max(w_gs)
        min_h_b = min(h_bs)
        max_h_h = max(h_hs)

        width = max_w_g - min_w_d
        height = max_h_h - min_h_b

        ratio = round(height / width, 2)

        len_group = len(group)
        fg_color = [round(sum(channel) / len_group) for channel in zip(*(w["fg_color"] for w in group))]
        bg_color = [round(sum(channel) / len_group) for channel in zip(*(w["bg_color"] for w in group))]

        append_row({
            "text": text_joined,
            "filename": first["filename"],
            "W_d": min_w_d,
            "W_g": max_w_g,
            "H_b": min_h_b,
            "H_h": max_h_h,
            "Width": width,
            "Height": height,
            "ratio": ratio,
            "fg_color": fg_color,
            "bg_color": bg_color,
            "len_text": len_text,
            "page": 0,
            "line": True,
        })
    return new_rows

def ExtractLineSegments(df, build_using_characters=False):
    """ExtractLineSegments: Group OCR boxes into text segments (Appendix A)."""
    lines = identify_lines(df, build_using_characters=build_using_characters)

    prune_long_lines = random.random() < proba_limit_segments_length
    max_group_len = 5 if prune_long_lines else None

    groups_by_length = defaultdict(list)
    for line in lines:
        line_groups = group_words_and_eliminate_chars(line, max_group_len)
        for length, groups in line_groups.items():
            groups_by_length[length].extend(groups)

    del lines

    all_groups = [group for groups in groups_by_length.values() for group in groups]

    del groups_by_length

    if len(all_groups) <= max_number_of_sequences:
        selected_groups = all_groups

    else:
        shuffled = all_groups[:]
        random.shuffle(shuffled)
        selected_groups = shuffled[:max_number_of_sequences]

    new_rows = process_groups(selected_groups, build_using_characters)    

    del selected_groups
    del all_groups

    if new_rows:
        df = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)
    
    del new_rows
    
    return df

def get_frauded_image_from_csv(self,csv_path,disable_crop_similarity_network=False,disable_crop_quality_network=False,disable_get_lines=False, disable_all=False) :
    """Generate manipulated document using F_theta and G_theta (Algorithm 2, Section 3.3)."""
    if random.random()< probability_deactivate_crop_quality_network:
        disable_crop_quality_network=True

    if random.random()< probability_deactivate_crop_similarity_network:
        disable_crop_similarity_network=True
    
    if disable_all:
        disable_crop_similarity_network=True
        disable_crop_quality_network=True
        disable_get_lines=True

    
    
    df_image = pd.read_csv(csv_path)

    try :
        page_num=df_image.iloc[0]["page"]
    except :
        page_num=False
        
    image_path=df_image.iloc[0]["filename"]


    if image_path.endswith(".pdf") :
        img_destination=get_image_from_pdf(image_path,page_num)
    else :
        img_destination=cv2.imread(image_path)
        if not isinstance(img_destination, np.ndarray):
            print(f"[WARN] Failed to read image at {image_path} line 1354, type is {type(img_destination)}")
    
 


    parent_dir = os.path.dirname(os.path.dirname(image_path))

    grouped_csv_path = os.path.join(parent_dir, grouped_csvs_folder)

    if not os.path.exists(grouped_csv_path):
        print(f"grouped_csvs folder not found at: {grouped_csv_path}")


    df_image["line"] = False

    df_image['text'] = df_image['text'].astype(str)

    #used later to cap number of manipulations
    number_of_words = len(df_image[df_image['len_text'] > 1])
    initial_len= len(df_image)
    
    df_image = numerise_df(df_image)
    if random.random()<probability_no_manipulation:
        n=0
    else:
        n=1+int(random.random()*max_manipulations_per_doc)

    if n>0 and not disable_get_lines and random.random()<get_segments_proba:
        #add variables in config for these
        build_using_characters=len(df_image[df_image['len_text'] == 1]) > len(df_image[df_image['len_text'] > 1])
        if not build_using_characters:
            print(f"got {len(df_image[df_image['len_text'] == 1])} characters and {len(df_image[df_image['len_text'] > 1])} words")
        else:
            df_image=ExtractLineSegments(df_image,build_using_characters=build_using_characters)
            print(f"length is {len(df_image)}")

    if random.random()<0.25:
        mask_len = df_image['len_text'] > 1
        if mask_len.sum() > 500:
            df_image = df_image.loc[mask_len].copy()


    
    #putback
    if dataset_name in datasets_to_expand:
        df_image["W_d"] -= 1
        df_image["W_g"] += 1
        df_image["H_b"] -= 1
        df_image["H_h"] += 1

    if dataset_name not in cropped_stats.keys():
        cropped_stats[dataset_name]=[0,0]

    if random.random() < bbox_expansion_probability :
        df_image["W_d"] -= random.choice([1,0,-1])
        df_image["W_g"] += random.choice([1,0,-1])
        df_image["H_b"] -= random.choice([1,0,-1])
        df_image["H_h"] += random.choice([1,0,-1])


    # clamp boxes to image bounds (prevents empty slices later)
    H_img, W_img = img_destination.shape[:2]
    df_image["W_d"] = df_image["W_d"].clip(lower=0, upper=W_img - 1)
    df_image["W_g"] = df_image["W_g"].clip(lower=0, upper=W_img)
    df_image["H_b"] = df_image["H_b"].clip(lower=0, upper=H_img - 1)
    df_image["H_h"] = df_image["H_h"].clip(lower=0, upper=H_img)
    #recompute
    df_image["Width"]  = (df_image["W_g"] - df_image["W_d"]).clip(lower=0)
    df_image["Height"] = (df_image["H_h"] - df_image["H_b"]).clip(lower=0)

    df_image=df_image[((df_image['W_g']-df_image['W_d'])>=2) & ((df_image['H_h']-df_image['H_b'])>=2)]
    df_image["ratio"] = (df_image["Height"] / df_image["Width"])
    
    assert (df_image['Width'] >= 0).all(), "Some Width values are negative"
    assert (df_image['Height'] >= 0).all(), "Some Height values are negative"
    
    df_image_save = df_image.copy()

    df_valid = df_image[df_image['line'] == False]

    number_of_words = len(df_valid[df_valid['len_text'] > 1])

    max_fraud = max(
        int(number_of_words / 5),
        int(initial_len / 20)
    )

    n=min(n,max_fraud)
    
    mask = np.zeros_like(img_destination, dtype=np.uint8)[:,:,0]

    img_destination_save = img_destination.copy()

    attemps=0
    
    force_using_stripes=random.random()<force_using_stripes_proba
    global curr_use_only_image
    curr_use_only_image=use_only_image and not force_using_stripes

    
    only_cm_sp =random.random() < probability_only_cm_sp
    high_quality=True
    #insertion
    if not only_cm_sp and (n!=0 and n!=1 and random.random()<probability_insertion):
        while attemps<50:
            try :
                attemps+=1
                max_w,max_h=img_destination.shape[1],img_destination.shape[0]
                coord_dest,text,coord_source,fg_color=get_blank_area(df_image,max_w,max_h,img_destination,get_closest_textbox=True)
                if coord_dest!=False :
                    fg_color=[max(el,0) for el in fg_color]
                    coord_dest=list(coord_dest)

                    if coord_dest[3]>coord_dest[2]+5 and coord_dest[1] > coord_dest[0]+5 :

                            patch_unpainted=img_destination[coord_dest[2]:coord_dest[3], coord_dest[0]:coord_dest[1]]

                            patch_to_compared_to=img_destination[coord_source[2]:coord_source[3], coord_source[0]:coord_source[1]]

                            patch_unpainted_copy=patch_unpainted.copy()

                            list_patch_unpainted, text=write_on_image(patch_unpainted,text,fg_color,  vary_color=False, ratio_candidates=[0.8], margin_candidates=[0])
                            chosen_crop,more_than_threshold = choose_crop(self, list_patch_unpainted, patch_to_compared_to, is_blank=False, disable_crop_similarity_network=disable_crop_similarity_network, high_quality=(high_quality and attemps<45))

                            if more_than_threshold:
                                list_patch_unpainted, _=write_on_image(patch_unpainted,text,fg_color)
                                chosen_crop,more_than_threshold = choose_crop(self, list_patch_unpainted, patch_to_compared_to, is_blank=False, disable_crop_similarity_network=disable_crop_similarity_network, high_quality=(high_quality and attemps<45))
                                
                                if more_than_threshold:
                                    img_destination[coord_dest[2]:coord_dest[3], coord_dest[0]:coord_dest[1]]=chosen_crop
            
                                    mask_added = (chosen_crop!= patch_unpainted_copy).sum(-1)>=1

                                    region = mask[coord_dest[2]:coord_dest[3], coord_dest[0]:coord_dest[1]]
                                    mask[coord_dest[2]:coord_dest[3], coord_dest[0]:coord_dest[1]] = np.maximum(region, mask_added.astype(np.uint8)*255)
                                    
                                    break
            except Exception as e:
                exc_type, exc_value, exc_tb = sys.exc_info()
                fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                print(f"[ERROR] {exc_type.__name__} in {fname}:{exc_tb.tb_lineno} -> {exc_value}")
                traceback.print_exc()
            
    succeded=0   
    attemps=0
    curr_splicing_tries=0

    df_image["cropped"] = -1
    df_image_save["cropped"] = -1

    while succeded<n and attemps<= 50000:
            attemps+=1

            if high_quality and succeded==0 and (len(df_image)<=2 or attemps==49000):
                img_destination=img_destination_save.copy()
                df_image= df_image_save.copy()
                mask = np.zeros_like(img_destination, dtype=np.uint8)[:,:,0]
                high_quality=False
                attemps=40000
                print("Generation failed, trying again with a lower quality")

            if len(df_image)==0:
                break
            

            if not only_cm_sp and random.random()< probability_target_blank_area:
                #target is a blank area
                max_w,max_h=img_destination.shape[1],img_destination.shape[0]
                coord_dest,text,_,_=get_blank_area(df_image_save,max_w,max_h,img_destination)
                if coord_dest!=False :
                    x1, x2, y1, y2 = coord_dest
   
                    row = {
                        "text":             text,
                        "filename":         df_image_save.iloc[0]["filename"],
                        "W_d":              x1,
                        "W_g":              x2,
                        "H_b":              y1,
                        "H_h":              y2,
                        "Width":            x2 - x1,
                        "Height":           y2 - y1,
                        "ratio":            round((y2 - y1) / (x2 - x1),  2),
                        "line":             False,
                        "cropped":          0,
                        "len_text": len(text)
                    }
                    is_blank=True

                else :
                    continue
            else :
                #target is a text area
                row=df_image.sample(1).squeeze()             
                df_image = df_image.drop(row.name)
                is_blank=False
            try:
                row["text"]=str(row["text"]).strip()
                coord_dest = [int(row[k]) for k in ["W_d","W_g","H_b","H_h"]]
                crop_dest = img_destination[coord_dest[2]:coord_dest[3], coord_dest[0]:coord_dest[1]]

                h_dest, w_dest = crop_dest.shape[:2]
                source_colors_for_selection = None #used only when ablating f
                
                if not is_blank:
                    #check cropped in not blank
                    cropped_dest =is_cropped_fn(self,crop_dest,img_destination,row,df_image,df_image_save, high_quality=high_quality, disable_crop_quality_network=disable_crop_quality_network)
    
                    if cropped_dest:
                        continue

                if not only_cm_sp and (random.random() < probability_inpainting and not is_blank):
                    #Inpainting
                    x1, x2, y1, y2 = [int(row[k]) for k in ("W_d","W_g","H_b","H_h")]
                    if y2 <= y1+5 or x2 <= x1+5:
                        continue
                    text = row["text"]
                
                    full_mask = np.zeros_like(mask, dtype=np.uint8)
                
                    if random.random() < 0.5:
                        full_mask[y1:y2, x1:x2] = 255
                
                        img_destination = cv2.inpaint(img_destination, full_mask, 3, cv2.INPAINT_NS)
                        mask |= full_mask
                
                        succeded += 1
                        
                    else :
                        try :
                            patch = img_destination[y1:y2, x1:x2]
                            gray  = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
                    
                            thresh = threshold_sauvola(gray, window_size=21, k=0.7, r=50)
                            text_mask = (gray < thresh).astype(np.uint8) * 255
                    
                            full_mask[y1:y2, x1:x2] = text_mask
                    
                            img_destination = cv2.inpaint(img_destination, full_mask, 3, cv2.INPAINT_NS)
                            mask |= full_mask
                    
                            mask[y1:y2, x1:x2] |= 255
                    
                            succeded += 1
                            only_cm_sp= True

                        except Exception as e:
                            exc_type, exc_value, exc_tb = sys.exc_info()
                            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                            print(f"[ERROR] {exc_type.__name__} in {fname}:{exc_tb.tb_lineno} -> {exc_value}")
                            traceback.print_exc()
                            continue

                    #insertion
                    if random.random()<probability_insertion:

                        patch_unpainted = img_destination[y1:y2, x1:x2]
                        ref_patch = img_destination_save[y1:y2, x1:x2]
                        fg_color=row["fg_color"]
                        fg_color=[max(el,0) for el in fg_color]
                        candidates, _ = write_on_image(patch_unpainted, row["text"], fg_color)
                        chosen_crop, more_than_threshold = choose_crop(self, candidates, ref_patch, high_quality=high_quality, is_blank=False, disable_crop_similarity_network=disable_crop_similarity_network)

                        if more_than_threshold:
                            img_destination[y1:y2, x1:x2] = chosen_crop
                            # update mask without overflow
                            diff = (chosen_crop != patch_unpainted).any(axis=-1).astype(np.uint8) * 255
                            mask[y1:y2, x1:x2] = np.maximum(mask[y1:y2, x1:x2], diff)
                 
                    continue

   
                    
    
                elif not only_cm_sp and not is_blank and ((random.random() < coverage_probability_target_not_character and len(row["text"])>1) or (random.random() < coverage_probability_target_character and len(row["text"])==1)):   
                        #source is blank region (coverage)              
    
                        
                        h_,w_=coord_dest[3]-coord_dest[2], coord_dest[1]-coord_dest[0]     
    
                        list_crops = find_blank_regions(img_destination_save, w_, h_, df_image_save)
    
                        source_patches = []
                        window_size = 25
                        k = 0.5
                        r = 128
                        
                        for crop in list_crops:
                            try :
                                gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                                thresh_map = threshold_sauvola(gray, window_size=window_size, k=k, r=r)
                                fg_mask = gray < thresh_map
                                if not fg_mask.any():
                                    crop=cv2.resize(crop, (w_dest, h_dest), interpolation=cv2.INTER_LINEAR)
                                    source_patches.append(crop)
                            except Exception as e:
                                exc_type, exc_value, exc_tb = sys.exc_info()
                                fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                                print(f"[ERROR] {exc_type.__name__} in {fname}:{exc_tb.tb_lineno} -> {exc_value}")
                                traceback.print_exc()
                                #don't need continue here

                                            
    
                        if len(source_patches) < 1:
                            continue
                        else:
                            chosen_crop,more_than_threshold = choose_crop(self, source_patches, crop_dest, high_quality=high_quality, is_blank=True, disable_crop_similarity_network=disable_crop_similarity_network)
                            if more_than_threshold:
                                x1, x2, y1, y2 = coord_dest[0], coord_dest[1], coord_dest[2], coord_dest[3]
                                # apply coverage
                                img_destination[y1:y2, x1:x2] = chosen_crop
                                mask[y1:y2, x1:x2] = 255
                                succeded += 1
                                total[0] += 1
                                only_cm_sp=True

                                # optional insertion on top of coverage
                                if random.random() < probability_insertion:
                                    patch_base = img_destination[y1:y2, x1:x2]        
                                    fg_color=row["fg_color"]
                                    fg_color=[max(el,0) for el in fg_color]
                                    candidates, _ = write_on_image(patch_base, row["text"], fg_color)
                                    chosen_overlay, more_than_threshold = choose_crop(self, candidates, crop_dest, high_quality=high_quality, is_blank=False, disable_crop_similarity_network=disable_crop_similarity_network)
                                    if more_than_threshold:
                                        img_destination[y1:y2, x1:x2] = chosen_overlay
                                        diff = (chosen_overlay != patch_base).any(axis=-1).astype(np.uint8) * 255
                                        mask[y1:y2, x1:x2] = np.maximum(mask[y1:y2, x1:x2], diff)
                            continue

                elif curr_splicing_tries<max_splicing_tries and random.random() < probability_splicing :
                    #Splicing
                    curr_splicing_tries+=1
                    start_time=time.time()
                    group_keys=["len_text","Width","Height"]
                    name=tuple(row[col] for col in group_keys)
                    name_csv='_'.join(str(x) for x in name) + '.csv'
                    dest=grouped_csv_path
                    path_csv=os.path.join(dest, name_csv)
                    
                    df= pd.DataFrame()
                    found_csv=False
                    folder_path = path_csv.replace('.csv', '')

                    if os.path.isdir(folder_path):
                        files_in_folder = [file for file in os.listdir(folder_path) if file.endswith('.csv')]
    
                        if files_in_folder:
                            random_file = random.choice(files_in_folder)
                            path_csv = os.path.join(folder_path, random_file)
                            found_csv=True
                  
                            
                    if found_csv :
                        df=pd.read_csv(path_csv)
                        df =numerise_df(df)

                        df.dropna(subset=['Width', 'Height', 'W_d', 'W_g', 'H_h', 'H_b'], inplace=True)
    
                        df_filtered = df[(df["filename"] != row["filename"])&
        (df["text"].astype(str).str.strip() != row["text"])]


                        if len(df_filtered)>max_candidate_crops_splicing:
                            if len(df_filtered) < 1:
                                continue
                            
                            #filter using color to accelerate the loading step below, threshold should be kept large as color estimation is not precise
                            target_fg = row.get("fg_color")
                            target_bg = row.get("bg_color")
                            if isinstance(target_fg, str):
                                target_fg = parse_color(target_fg)
                            if isinstance(target_bg, str):
                                target_bg = parse_color(target_bg)

                            def within_color_threshold(candidate_row, threshold):
                                candidate_fg = candidate_row.get("fg_color")
                                candidate_bg = candidate_row.get("bg_color")
                                if isinstance(candidate_fg, str):
                                    candidate_fg_converted = parse_color(candidate_fg)
                                else:
                                    candidate_fg_converted = candidate_fg
                                if isinstance(candidate_bg, str):
                                    candidate_bg_converted = parse_color(candidate_bg)
                                else:
                                    candidate_bg_converted = candidate_bg

                                pass_fg = True
                                pass_bg = True
                                if isinstance(target_fg, (list, tuple, np.ndarray)):
                                    pass_fg = color_distance(candidate_fg_converted, target_fg) <= threshold
                                if isinstance(target_bg, (list, tuple, np.ndarray)):
                                    pass_bg = color_distance(candidate_bg_converted, target_bg) <= threshold
                                return pass_fg and pass_bg

                            for filtring_iteration in range(splicing_color_threshold,0,-1):
                                if isinstance(target_fg, (list, tuple, np.ndarray)) or isinstance(target_bg, (list, tuple, np.ndarray)):
                                    length_before_filtring=len(df_filtered)
                                    df_filtered = df_filtered[df_filtered.apply(lambda row: within_color_threshold(row, filtring_iteration), axis=1)]
                                    print(f"length went from {length_before_filtring} to {len(df_filtered)} after color filtring before splicicng after {splicing_color_threshold-filtring_iteration+1} iterations")
                                if len(df_filtered)<=max_candidate_crops_splicing:
                                    break
                                ##finished filtring
                        if len(df_filtered) < 1:
                            continue
                        if len(df_filtered)>max_candidate_crops_splicing:
                            df_filtered = df_filtered.sample(
                                            n=max_candidate_crops_splicing,
                                            replace=False
                                        ).reset_index(drop=True)
                            print(f"sampled {len(df_filtered)} elements")
                            
                        
    
                        
                        # Collect all candidates and their data for batch processing
                        source_color_candidates = []
                        batch_data = []
                        source_patch_candidates = []
                        for _, candidate_row in df_filtered.iterrows():
                            fname = candidate_row["filename"]
                            if fname.lower().endswith(".pdf"):
                                img_source = get_image_from_pdf(fname, int(candidate_row["page"]))
                            else:
                                img_source = cv2.imread(fname)
                                if not isinstance(img_source, np.ndarray):
                                    print(f"[WARN] Failed to read image at {fname} line 1354, type is {type(img_source)}")
    
                            
                            coord_source = [int(candidate_row[k]) for k in ["W_d", "W_g", "H_b", "H_h"]]
                            x1, x2, y1, y2 = coord_source[0], coord_source[1], coord_source[2], coord_source[3]
                            
                            source_patch = img_source[y1:y2, x1:x2]
    
                            source_patch=cv2.resize(source_patch, (w_dest, h_dest), interpolation=cv2.INTER_LINEAR)
                            
                            # Collect data for batch processing
                            batch_data.append((source_patch, img_source, candidate_row, None, None, False))
                            source_patch_candidates.append(source_patch)
                            source_color_candidates.append({
                                "fg_color": candidate_row.get("fg_color"),
                                "bg_color": candidate_row.get("bg_color"),
                            })
                            
                            del img_source
                        
                        del df_filtered
                        del df
                        
                        # Process all candidates in batch
                        if len(batch_data) > 0:
                            cropped_results = is_cropped_fn_batch(self, batch_data, high_quality=high_quality, disable_crop_quality_network=disable_crop_quality_network)
                            
                            del batch_data
                            
                            # Filter out cropped patches
                            source_patches = []
                            selected_colors = []
                            for i, cropped_source in enumerate(cropped_results):
                                if not cropped_source:
                                    source_patches.append(source_patch_candidates[i])
                                    selected_colors.append(source_color_candidates[i])
                            
                            del cropped_results
                            del source_patch_candidates
                            del source_color_candidates
                            
                            source_colors_for_selection = selected_colors if selected_colors else None
                        else:
                            source_patches = []
                            source_colors_for_selection = None #should be None anyway
                            
    
                    else :
                        continue
    
                    splicing_cost=int((time.time()-start_time))-1
                    if splicing_cost>0:
                        curr_splicing_tries+=splicing_cost
                        print(f"splicing is costly, tries are incremented by {splicing_cost}")
    
                else :
                    #Copy-move
                    
                    df = df_image
                    df = df[df["text"].astype(str).str.strip() != row["text"]]
                    df = df[df["len_text"] == row["len_text"]]
                    df = df[
                        (df["ratio"] / row["ratio"]).apply(lambda x: abs(max(x, 1 / x) - 1) <= ratio_diff_tolerance_copy_move)
                    ]
                    #removing overlap
                    if len(df)>0:
                        df = df[~((df["W_d"] < coord_dest[1]) & (df["W_g"] > coord_dest[0]) & (df["H_b"] < coord_dest[3]) & (df["H_h"] > coord_dest[2]))]
                    else :
                        continue
    
                    
                    if len(df) < 1:
                        continue
    
                    
    
                    # Collect all candidates and their data for batch processing
                    batch_data = []
                    source_patch_candidates = []
                    source_color_candidates = []
                    for _, candidate in df.iterrows():
                        
                        x1 = int(candidate["W_d"])
                        x2 = int(candidate["W_g"])
                        y1 = int(candidate["H_b"])
                        y2 = int(candidate["H_h"])
                        source_patch = img_destination_save[y1:y2, x1:x2]
                    
                        resized_patch = cv2.resize(source_patch, (w_dest, h_dest), interpolation=cv2.INTER_LINEAR)
                        
                        # Collect data for batch processing
                        batch_data.append((resized_patch, img_destination_save, candidate, df_image, df_image_save, True))
                        source_patch_candidates.append(resized_patch)
                        source_color_candidates.append({
                            "fg_color": candidate.get("fg_color"),
                            "bg_color": candidate.get("bg_color"),
                        })
                    
                    del df
                    
                    # Process all candidates in batch
                    if len(batch_data) > 0:
                        cropped_results = is_cropped_fn_batch(self, batch_data, high_quality=high_quality, disable_crop_quality_network=disable_crop_quality_network)
                        
                        del batch_data
                        
                        # Filter out cropped patches
                        source_patches = []
                        selected_colors = []
                        for i, cropped_source in enumerate(cropped_results):
                            if not cropped_source:
                                source_patches.append(source_patch_candidates[i])
                                selected_colors.append(source_color_candidates[i])
                        
                        del cropped_results
                        del source_patch_candidates
                        del source_color_candidates
                        
                        source_colors_for_selection = selected_colors if selected_colors else None
                    
                    else:
                        source_patches = []
                        source_colors_for_selection = None #should be None anyway

    
                if len(source_patches)>0:
                    color_kwargs = {}
                    if disable_crop_similarity_network and source_colors_for_selection: ##use colors only if crop_similarity_network is disabled
                        target_fg = row.get("fg_color")
                        target_bg = row.get("bg_color")

                        color_kwargs = {
                            "target_fg_color": target_fg,
                            "target_bg_color": target_bg,
                            "source_colors": source_colors_for_selection,
                        }

                    chosen_crop,more_than_threshold = choose_crop(
                        self,
                        source_patches,
                        crop_dest,
                        high_quality=high_quality,
                        is_blank=is_blank,
                        disable_crop_similarity_network=disable_crop_similarity_network,
                        **color_kwargs,
                    )
                else:
                    more_than_threshold=False
    
                if more_than_threshold:
                    img_destination[coord_dest[2]:coord_dest[3], coord_dest[0]:coord_dest[1]] = chosen_crop
                    mask[coord_dest[2]:coord_dest[3], coord_dest[0]:coord_dest[1]] = 255 
                    succeded+=1
                    if is_blank:
                        only_cm_sp=True
                    total[0]+=1
                    
                if 'source_patches' in locals():
                    del source_patches
                if source_colors_for_selection is not None:
                    del source_colors_for_selection

                if high_quality:
                    df_image = df_image[~(df_image["cropped"] == 1)]
            except Exception as e:
                print(f"{type(e).__name__}: {e}")      # error type + message
                print("args:", e.args)                  # any arguments passed to the exception
                traceback.print_exc()                   # full stack trace
    for key in sorted(cropped_stats.keys()):
        value = cropped_stats[key]
        total_v = value[0] + value[1] + 1e-8
        norm = [value[0] / total_v, value[1] / total_v]
        # print(f"{key}: {norm}")
    
    del df_image
    del df_image_save
    del img_destination_save
    
    return img_destination,mask


class DocMDetectorDataset(Dataset):
    """Dataset for generating manipulated documents using F_theta and G_theta (Algorithm 2)."""

    def __init__(
        self,
        job_index=None,
        max_jobs=1,
        resume_map_path=None 
    ):
        super().__init__()

        self.resume_map_path= resume_map_path
        self.job_index = job_index
        self.max_jobs = max(1, int(max_jobs))

        if self.max_jobs > 1:
            if self.job_index is None:
                raise ValueError("job_index must be provided when max_jobs > 1")
            if not (0 <= self.job_index < self.max_jobs):
                raise ValueError(f"job_index {self.job_index} must satisfy 0 <= job_index < {self.max_jobs}")

        self.dataset_length = 0
        self.csvs=[]
        self.paths=[]

        cfg = Config(config_path_crop_embed)
        
        model_config = DocMDetectorConfig(pretrained_model_name_or_path=cfg.pretrained_model_name_or_path)

        meta_model = DocMDetector(model_config)
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        meta_model = meta_model.to(device)
        meta_model.eval()
        self.meta_model=meta_model

        cfg = Config(config_path_crop_quality)
        model_config = Mconfig(pretrained_model_name_or_path=cfg.pretrained_model_name_or_path)
        crop_model = MCrop(model_config)
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        crop_model = crop_model.to(device)
        crop_model.eval()
        self.crop_model = crop_model

        prefix = datasets_main_path
        suffix = "/merged_jsons"

        for current_index, (dataset, range_val) in enumerate(datasets_upsample_factor.items()):
            prev_length = 0
            prev_length = len(self.csvs)

            dirpath = prefix + dataset + suffix
            print(dirpath)
            for filename in os.listdir(dirpath):
                filepath = os.path.join(dirpath, filename)
                if filename.endswith(".csv"):
                    if range_val>1 :
                        for _ in range(range_val):
                            self.csvs.append((filepath,dataset))
                            self.dataset_length += 1
                    else :
                        if random.random() < range_val:
                            self.csvs.append((filepath,dataset))
                            self.dataset_length += 1
                        
                            
            print(dataset.split('/')[-1])
            print("ratio ", (len(self.csvs) - prev_length)/3000)
            print("Length: ", len(self.csvs) - prev_length)
            print(f"Final length {len(self.csvs)}")
            prev_length = len(self.csvs)
            print(len(self.csvs))
        total_before_shard = len(self.csvs)
        if self.max_jobs > 1:
            filtered = []
            for filepath, dataset_name in self.csvs:
                shard_id = self._assign_shard(filepath)
                if shard_id == self.job_index:
                    filtered.append((filepath, dataset_name))
            print(f"Shard {self.job_index}/{self.max_jobs} retained {len(filtered)} of {total_before_shard} samples")
            self.csvs = filtered

        self.dataset_length = len(self.csvs)

        if self.resume_map_path is not None:
            try:
                with open(self.resume_map_path, "r") as f:
                    resume_map = json.load(f)  # {csv_path: left_count}
            except Exception as e:
                raise RuntimeError(f"[ERROR] resume_map could not be loaded: {e}")

            # Map discovered CSVs to their dataset tag (preserves your dataset_name usage)
            discovered = {p: d for (p, d) in self.csvs}

            new_list = []
            kept, skipped = 0, 0
            for csv_path, left_cnt in resume_map.items():
                try:
                    left_cnt = int(left_cnt)
                except Exception:
                    continue
                if left_cnt <= 0:
                    continue
                dtag = discovered.get(csv_path)
                if dtag is None:
                    # not in this shard or not discovered with current config
                    skipped += 1
                    continue
                new_list.extend([(csv_path, dtag)] * left_cnt)                
                kept += left_cnt

            self.csvs = new_list
            self.dataset_length = len(self.csvs)
            print(f"Resume mode: expanded to {kept} items from {len(resume_map)} entries (skipped {skipped}).")


    def __len__(self) -> int:
        return self.dataset_length

    def _assign_shard(self, filepath: str) -> int:
        normalized = os.path.normcase(os.path.abspath(filepath))
        digest = hashlib.md5(normalized.encode("utf-8")).digest()
        return int.from_bytes(digest, "big") % self.max_jobs

    def __getitem__(self, idx: int):
        try:
            global dataset_name
            csv, dataset_name = self.csvs[idx]
            base = os.path.splitext(os.path.basename(csv))[0]

            # --- ensure directories ---
            if not save_in_a_single_folder:
                main_dir = os.path.join(output_folder, dataset_name)
                os.makedirs(main_dir, exist_ok=True)
            else:
                os.makedirs(os.path.join(output_folder, "images"), exist_ok=True)
                os.makedirs(os.path.join(output_folder, "masks"), exist_ok=True)

            # --- main sample ---
            img_main, mask_main = get_frauded_image_from_csv(self, csv)
            if not save_in_a_single_folder:
                cv2.imwrite(os.path.join(main_dir, f"{base}_image_{idx}.png"), img_main)
                cv2.imwrite(os.path.join(main_dir, f"{base}_mask_{idx}.png"),  mask_main)
            else:
                cv2.imwrite(os.path.join(output_folder,"images", f"{base}_{idx}.png"), img_main)
                cv2.imwrite(os.path.join(output_folder,"masks", f"{base}_{idx}.png"),  mask_main)

            del img_main
            del mask_main


        except Exception as e:
            print(f"[Warning] generation failed for {csv}, this is unusual! Error is {e}")
        return None, None

def no_collate(batch):
    """Identity collate function."""
    return batch

def clean():
    """Free GPU memory and run garbage collection."""
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()

def main():
    """Main entry point for fraud data generation pipeline."""
    args = parse_args()

    if args.resume_map is not None:
        assert save_in_a_single_folder, (
            "When using --resume_map, you must set save_in_a_single_folder: true "
            "to guarantee consistent output paths."
        )

    max_jobs = max(1, int(args.max_jobs))

    if max_jobs > 1:
        job_index = args.job_index if args.job_index is not None else 0
    else:
        job_index = args.job_index

    dataset = DocMDetectorDataset(job_index=job_index, max_jobs=max_jobs, resume_map_path=args.resume_map)
    

    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=1000 if num_workers>0 else None,
        collate_fn=no_collate,

    )

    for iteration,empty_batch in enumerate(tqdm(loader, total=len(dataset))):
        if iteration %100==99:
            clean()
        

if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    main()

