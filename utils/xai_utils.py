import cv2
import numpy as np
import torch

def generate_attention_rollout(attentions, discard_ratio=0.9, head_fusion="mean"):
    """
    Computes Attention Rollout to highlight morphological regions the ViT focuses on.
    
    Args:
        attentions: List of attention matrices from ViT encoder layers.
        discard_ratio: Top percentage of attention to keep.
        head_fusion: How to fuse attention heads (mean, max, min).
    Returns:
        heatmap: 2D numpy array of the final attention map (14x14 for 224x224 patch16).
    """
    result = torch.eye(attentions[0].size(-1))
    
    with torch.no_grad():
        for attention in attentions:
            if head_fusion == "mean":
                attention_heads_fused = attention.mean(axis=1)
            elif head_fusion == "max":
                attention_heads_fused = attention.max(axis=1)[0]
            elif head_fusion == "min":
                attention_heads_fused = attention.min(axis=1)[0]
            else:
                raise Exception("Attention head fusion type Not supported")

            # Drop the lowest attentions, but don't drop the class token
            flat = attention_heads_fused.view(attention_heads_fused.size(0), -1)
            _, indices = flat.topk(int(flat.size(-1)*discard_ratio), -1, False)
            indices = indices[attention_heads_fused.size(0) != indices]
            flat[0, indices] = 0

            I = torch.eye(attention_heads_fused.size(-1))
            a = (attention_heads_fused + 1.0*I)/2
            a = a / a.sum(dim=-1, keepdims=True)

            result = torch.matmul(a, result)
            
    # Look at the total attention between the class token and the image patches
    mask = result[0, 0, 1:]
    
    # In case of 196 patches (14x14)
    width = int(np.sqrt(mask.size(0)))
    mask = mask.reshape(width, width).numpy()
    mask = mask / np.max(mask)
    
    return mask

def overlay_heatmap(original_img, mask, alpha=0.5):
    """
    Overlays processing mask on top of the original image for XAI visualization.
    
    Args:
        original_img: PIL Image or Numpy array of original image.
        mask: 2D numpy array (from rollout).
        alpha: Transparency factor.
    """
    # Resize mask to fit original image
    if isinstance(original_img, torch.Tensor):
        original_img = original_img.numpy().transpose(1, 2, 0)
        
    original_img = np.array(original_img)
    mask = cv2.resize(mask, (original_img.shape[1], original_img.shape[0]))
    
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    
    # Needs to normalize original image 0-1 if not already
    orig = np.float32(original_img) / 255 if original_img.max() > 1.0 else original_img
    
    overlay = heatmap * alpha + orig * (1 - alpha)
    overlay = overlay / np.max(overlay)
    
    return np.uint8(255 * overlay)
