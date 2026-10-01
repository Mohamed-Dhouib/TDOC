"""
Preprocessing utilities for FFDN inference.
Handles JPEG compression, DCT coefficient extraction, and quantization table loading.
"""
import io
import os
import tempfile
import warnings

import cv2
import numpy as np
import torch
from PIL import Image

try:
    import jpegio
    JPEGIO_AVAILABLE = True
except ImportError:
    JPEGIO_AVAILABLE = False
    warnings.warn('jpegio not available. Install with: pip install jpegio', RuntimeWarning)


def jpeg_compress_and_load_info(image, quality=75):
    """
    Compress image to JPEG and extract DCT coefficients and quantization table.
    
    Args:
        image: PIL Image or numpy array (H, W, 3) in RGB
        quality: JPEG compression quality (1-100)
    
    Returns:
        dict with keys:
            - 'img': Compressed RGB image as numpy array (H, W, 3)
            - 'dct': DCT coefficients clipped to [0, 20] (H, W)
            - 'qtb': Quantization table (1, 8, 8)
    """
    if not JPEGIO_AVAILABLE:
        raise RuntimeError("jpegio is required for DCT extraction. Install with: pip install jpegio")
    
    # Convert to PIL if numpy
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    
    # Compress to JPEG
    buffer = io.BytesIO()
    image.save(buffer, format='JPEG', quality=quality)
    buffer.seek(0)
    
    # Load compressed image
    compressed_img = Image.open(buffer)
    img_array = np.array(compressed_img)

    # Persist JPEG bytes temporarily for jpegio
    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp_file:
        tmp_file.write(buffer.getvalue())
        tmp_path = tmp_file.name
    try:
        jpg = jpegio.read(tmp_path)
    finally:
        os.remove(tmp_path)
    
    # Get DCT coefficients from Y channel
    dct = np.abs(jpg.coef_arrays[0])  # Y channel DCT
    dct = np.clip(dct, 0, 20)  # Clip to [0, 20]
    
    # Get quantization table
    qtb = jpg.quant_tables[0].astype(np.uint8)
    qtb = np.clip(qtb, 0, 63)  # Clip to [0, 63]
    qtb = np.expand_dims(qtb, 0)  # (1, 8, 8)
    
    return {
        'img': img_array,
        'dct': dct,
        'qtb': qtb
    }


def preprocess_image(image_path, quality=75, size=512, device='cuda'):
    """
    Full preprocessing pipeline for FFDN inference.
    
    Args:
        image_path: Path to input image
        quality: JPEG compression quality (default: 75)
        size: Target size for inference (default: 512)
        device: 'cuda' or 'cpu'
    
    Returns:
        dict with tensors ready for model input:
            - 'img': (1, 3, H, W) normalized RGB
            - 'dct': (1, H, W) DCT coefficients
            - 'qtb': (1, 1, 8, 8) quantization table
            - 'ori_shape': Original image shape (H, W)
    """
    # Load image
    img = Image.open(image_path).convert('RGB')
    ori_shape = img.size[::-1]  # (H, W)

    # Resize to the network input size
    img = img.resize((size, size), Image.LANCZOS)

    # Convert to numpy
    img_array = np.array(img)
    
    # Apply JPEG compression and extract DCT
    data = jpeg_compress_and_load_info(img_array, quality=quality)
    
    # Normalize image (ImageNet stats)
    mean = np.array([123.675, 116.28, 103.53])
    std = np.array([58.395, 57.12, 57.375])
    
    img_normalized = (data['img'].astype(np.float32) - mean) / std
    img_normalized = img_normalized.transpose(2, 0, 1)  # (3, H, W)
    
    # Convert to tensors
    img_tensor = torch.from_numpy(img_normalized).unsqueeze(0).float()  # (1, 3, H, W)
    dct_tensor = torch.from_numpy(data['dct']).unsqueeze(0).long()  # (1, H, W)
    qtb_tensor = torch.from_numpy(data['qtb']).unsqueeze(0).long()  # (1, 1, 8, 8)
    
    # Move to device
    if device == 'cuda' and torch.cuda.is_available():
        img_tensor = img_tensor.cuda()
        dct_tensor = dct_tensor.cuda()
        qtb_tensor = qtb_tensor.cuda()
    
    return {
        'img': img_tensor,
        'dct': dct_tensor,
        'qtb': qtb_tensor,
        'ori_shape': ori_shape
    }


def visualize_results(prediction, confidence, output_path):
    """
    Visualize and save segmentation results.
    
    Args:
        prediction: Binary mask (H, W) with 0=authentic, 1=tampered
        confidence: Confidence map (H, W) with values in [0, 1]
        output_path: Base path for saving (will create _mask.png and _confidence.png)
    """
    import matplotlib.pyplot as plt
    
    # Save binary mask
    mask_img = (prediction * 255).astype(np.uint8)
    cv2.imwrite(f'{output_path}_mask.png', mask_img)
    
    # Save confidence heatmap
    plt.figure(figsize=(10, 10))
    plt.imshow(confidence, cmap='jet', vmin=0, vmax=1)
    plt.colorbar()
    plt.title('Tampering Confidence')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(f'{output_path}_confidence.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved visualization:")
    print(f"  - {output_path}_mask.png")
    print(f"  - {output_path}_confidence.png")
