import argparse
import os
from contextlib import nullcontext

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

from model import DocMDetector, DocMDetectorConfig
from sconf import Config


IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp")


def _list_images(path: str) -> list[str]:
    if os.path.isfile(path):
        return [path]
    if not os.path.isdir(path):
        raise FileNotFoundError(f"Input path not found: {path}")
    files = []
    for fn in os.listdir(path):
        if fn.lower().endswith(IMAGE_EXTS):
            files.append(os.path.join(path, fn))
    files.sort()
    return files


def _infer_one(
    img_path: str,
    orig_img: np.ndarray,
    model: DocMDetector,
    input_size: int,
    device: str,
) -> tuple[np.ndarray, float]:
    """Infer a single image using the same tiling/merge logic as the eval script.

    Returns:
        prob_map (uint8 HxW, 0..255), cls_prob (float)
    """
    prepare_input = model.encoder.prepare_input
    encoder_fwd = model.encoder
    border_val = [255, 255, 255]
    use_amp = device == "cuda"
    amp_ctx = torch.cuda.amp.autocast(dtype=torch.bfloat16) if use_amp else nullcontext()

    H, W = orig_img.shape[:2]
    dummy_mask = Image.fromarray(np.zeros((H, W), dtype=np.uint8))

    if H > input_size or W > input_size:
        window_size = input_size
        stride = input_size
        orig_H, orig_W = H, W

        pad_top = pad_bottom = pad_left = pad_right = 0
        if H < window_size:
            pad_total = window_size - H
            pad_top = pad_total // 2
            pad_bottom = pad_total - pad_top
        if W < window_size:
            pad_total = window_size - W
            pad_left = pad_total // 2
            pad_right = pad_total - pad_left

        if pad_top or pad_bottom or pad_left or pad_right:
            orig_img = cv2.copyMakeBorder(
                orig_img, pad_top, pad_bottom, pad_left, pad_right,
                cv2.BORDER_CONSTANT, value=border_val
            )
            mask_np = np.array(dummy_mask)
            mask_padded_np = np.pad(
                mask_np,
                ((pad_top, pad_bottom), (pad_left, pad_right)),
                mode="constant", constant_values=0
            )
            dummy_mask = Image.fromarray(mask_padded_np).convert("L")

        H, W = orig_img.shape[:2]
        merged_prob = torch.zeros((H, W), dtype=torch.float32, device=device)
        cls_p_max = None

        for top0 in range(0, H, stride):
            bottom = min(top0 + window_size, H)
            top = bottom - window_size if (bottom - top0) < window_size else top0

            for left0 in range(0, W, stride):
                right = min(left0 + window_size, W)
                left = right - window_size if (right - left0) < window_size else left0

                img_crop = orig_img[top:bottom, left:right, :]
                mask_crop = dummy_mask.crop((left, top, right, bottom))

                im, _, dct, dwt, ela, qt = prepare_input(
                    img_crop, mask_crop,
                    aug=False, qualitys=[], crop_pad=False,
                    image_path_to_compute_qt=img_path
                )

                im = im.unsqueeze(0).to(device)
                dct = dct.unsqueeze(0).to(device)
                dwt = dwt.unsqueeze(0).to(device)
                ela = ela.unsqueeze(0).to(device)
                qt = qt.unsqueeze(0).to(device)

                with amp_ctx:
                    pred_logits, cls_logits = encoder_fwd(im, dct, dwt, ela, qt)

                prob_crop = F.softmax(pred_logits, dim=1)[0, 1]
                mp = merged_prob[top:bottom, left:right]
                torch.maximum(mp, prob_crop, out=mp)

                cls_prob = F.softmax(cls_logits, dim=1)[:, 1]
                cls_p_max = cls_prob if cls_p_max is None else torch.maximum(cls_p_max, cls_prob)

        if pad_top or pad_bottom or pad_left or pad_right:
            final_prob = merged_prob[pad_top:pad_top + orig_H, pad_left:pad_left + orig_W]
        else:
            final_prob = merged_prob

        cls_p_val = float(cls_p_max.detach().cpu().item()) if cls_p_max is not None else 0.0

    else:
        dh = max(input_size - H, 0)
        dw = max(input_size - W, 0)
        top, bottom = dh // 2, dh - dh // 2
        left, right = dw // 2, dw - dw // 2

        img_pad = cv2.copyMakeBorder(
            orig_img, top, bottom, left, right,
            cv2.BORDER_CONSTANT, value=border_val
        )
        mask_np = np.array(dummy_mask)
        mask_pad_np = np.pad(
            mask_np,
            ((top, bottom), (left, right)),
            mode="constant", constant_values=0
        )
        mask_pad = Image.fromarray(mask_pad_np).convert("L")

        im, _, dct, dwt, ela, qt = prepare_input(
            img_pad, mask_pad,
            aug=False, qualitys=[], crop_pad=False,
            image_path_to_compute_qt=img_path
        )

        im = im.unsqueeze(0).to(device)
        dct = dct.unsqueeze(0).to(device)
        dwt = dwt.unsqueeze(0).to(device)
        ela = ela.unsqueeze(0).to(device)
        qt = qt.unsqueeze(0).to(device)

        with amp_ctx:
            pred_logits, cls_logits = encoder_fwd(im, dct, dwt, ela, qt)

        seg_prob = F.softmax(pred_logits, dim=1)[0, 1]
        final_prob = seg_prob[top:top + H, left:left + W]

        cls_p = F.softmax(cls_logits, dim=1)[:, 1]
        cls_p_val = float(cls_p.detach().cpu().item())

    prob_map = (final_prob.cpu().numpy() * 255.0).round().astype(np.uint8)
    return prob_map, cls_p_val


def main():
    parser = argparse.ArgumentParser(description="Infer DocMDetector on images (same logic as eval).")
    parser.add_argument("--input", required=True, help="Path to image file or directory.")
    parser.add_argument("--output-dir", required=True, help="Directory to save prediction masks.")
    parser.add_argument(
        "--config-path",
        default=os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "configs",
            "ffdn_pretraining.yaml",
        ),
        help="Path to the configuration YAML file.",
    )
    parser.add_argument("--weights-path", default=None, help="Optional checkpoint override.")
    parser.add_argument("--threshold", type=float, default=0.5, help="Segmentation threshold.")
    parser.add_argument("--save-binary", action="store_true", help="Save thresholded mask as 0/255.")
    args = parser.parse_args()

    cfg_file = Config(args.config_path)
    pretrained_path = args.weights_path if args.weights_path else cfg_file.pretrained_model_name_or_path
    cfg = DocMDetectorConfig(
        model_name=cfg_file.model_name,
        pretrained_model_name_or_path=pretrained_path,
        input_size=cfg_file.input_size,
    )
    model = DocMDetector(cfg)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).eval()

    os.makedirs(args.output_dir, exist_ok=True)
    images = _list_images(args.input)

    with torch.inference_mode():
        for img_path in tqdm(images, desc="Inferring"):
            orig_img = cv2.imread(img_path, cv2.IMREAD_COLOR)
            if orig_img is None:
                print(f"Warning: failed to read {img_path}, skipping.")
                continue

            prob_map, cls_p_val = _infer_one(
                img_path=img_path,
                orig_img=orig_img,
                model=model,
                input_size=cfg.input_size[0],
                device=device,
            )

            fn = os.path.basename(img_path)
            out_prob = os.path.join(args.output_dir, fn)
            cv2.imwrite(out_prob, prob_map)

            if args.save_binary:
                bin_map = (prob_map > int(255 * args.threshold)).astype(np.uint8) * 255
                root, ext = os.path.splitext(fn)
                out_bin = os.path.join(args.output_dir, f"{root}_bin{ext or '.png'}")
                cv2.imwrite(out_bin, bin_map)

            print(f"{fn} -> cls_prob={cls_p_val:.6f}")


if __name__ == "__main__":
    main()
