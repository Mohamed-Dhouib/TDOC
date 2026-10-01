"""
Image preprocessing utilities for ASCFormer inference.
Implements DCT and ELA transformations matching RTM/ASCFormer exactly.
"""

import cv2
import numpy as np
import torch


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
    """
    Apply Block DCT transformation (exact copy from RTM/ASCFormer).
    
    Args:
        img: BGR image array (H, W, 3)
        block_size: DCT block size (default: 8)
        zigzag: Apply zigzag ordering (default: True)
        channel: YCrCb channel to use 'Y', 'U', or 'V' (default: 'Y')
        shift: Whether to shift by 128 before DCT (default: False)
    
    Returns:
        dct: DCT coefficients array (H, W)
    """
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


def ela_transform(img, quality=90, backend='cv2'):
    """
    Apply Error Level Analysis (exact copy from RTM/ASCFormer).
    
    Args:
        img: BGR image array (H, W, 3)
        quality: JPEG compression quality (1-100, default: 90)
        backend: 'cv2' or 'pillow' (default: 'cv2')
    
    Returns:
        ela: ELA result array (H, W, 3)
    """
    if backend == 'cv2':
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        _, encimg = cv2.imencode('.jpg', img, encode_param)
        decimg = cv2.imdecode(encimg, 1)
    else:
        import tempfile
        from PIL import Image
        with tempfile.NamedTemporaryFile(delete=True, suffix='.jpg') as tmp:
            im = Image.fromarray(img)
            im.save(tmp.name, "JPEG", quality=quality)
            decimg = Image.open(tmp.name)
            decimg = np.array(decimg)
    
    ela = cv2.absdiff(img, decimg)
    return ela


def preprocess_image(img_path, quality=80, device='cpu'):
    """
    Complete preprocessing pipeline matching RTM config.
    Defaults: quality=80, mean/std from ImageNet
    
    Args:
        img_path: Path to input image file
        quality: JPEG quality for ELA (default: 80 from RTM config)
        device: Target device for tensors ('cpu' or 'cuda')
    
    Returns:
        dict with 'img', 'dct', 'ela', 'ori_shape'
    """
    img = cv2.imread(str(img_path))
    if img is None:
        raise ValueError(f"Failed to load image from {img_path}")
    
    H, W = img.shape[:2]
    
    # Apply transformations (RTM config: quality=80, zigzag=True)
    ela = ela_transform(img, quality=quality)
    dct = block_dct_transform(img, zigzag=True)
    
    # BGR to RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Normalize with ImageNet statistics (from RTM config)
    mean = np.array([123.675, 116.28, 103.53], dtype=np.float32)
    std = np.array([58.395, 57.12, 57.375], dtype=np.float32)
    img_normalized = (img_rgb.astype(np.float32) - mean) / std
    
    # HWC -> CHW
    img_tensor = torch.from_numpy(img_normalized.transpose(2, 0, 1)).unsqueeze(0)
    ela_tensor = torch.from_numpy(ela.transpose(2, 0, 1)).unsqueeze(0).float()
    dct_tensor = torch.from_numpy(dct).unsqueeze(0).unsqueeze(0).float()
    
    # Move to device
    img_tensor = img_tensor.to(device)
    ela_tensor = ela_tensor.to(device)
    dct_tensor = dct_tensor.to(device)
    
    return {
        'img': img_tensor,
        'dct': dct_tensor,
        'ela': ela_tensor,
        'ori_shape': (H, W)
    }


def visualize_results(prediction, confidence, output_path):
    """
    Visualize and save segmentation results.
    
    Args:
        prediction: Binary mask array (H, W)
        confidence: Confidence map array (H, W)
        output_path: Base path for saving (without extension)
    """
    from pathlib import Path
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Binary mask
    mask_img = (prediction * 255).astype(np.uint8)
    cv2.imwrite(f"{output_path}_mask.png", mask_img)
    
    # Confidence map
    conf_img = (confidence * 255).astype(np.uint8)
    cv2.imwrite(f"{output_path}_confidence.png", conf_img)
    
    # Heatmap
    colored = cv2.applyColorMap(conf_img, cv2.COLORMAP_JET)
    cv2.imwrite(f"{output_path}_heatmap.png", colored)
    
    print(f"Results saved: {output_path}_mask.png, {output_path}_confidence.png, {output_path}_heatmap.png")
