import torch
import torch.nn as nn
from transformers import ViTModel, ViTConfig

class VisionEncoder(nn.Module):
    """
    Vision Transformer (ViT) encoder that captures local and global
    cellular morphological features from blood smear images.
    """
    def __init__(self, model_name="google/vit-base-patch16-224"):
        super().__init__()
        # Load a pre-trained Vision Transformer
        # We don't need the classification head, just the hidden states (patch embeddings)
        self.vit = ViTModel.from_pretrained(model_name, output_attentions=True)
        self.hidden_size = self.vit.config.hidden_size
        
        # Freeze initial layers for fine-tuning if necessary
        for param in self.vit.parameters():
            param.requires_grad = True # False to freeze

    def forward(self, pixel_values):
        """
        Args:
            pixel_values: Tensor of shape (batch, 3, 224, 224)
        Returns:
            last_hidden_state: Tensor of shape (batch, sequence_length, hidden_size)
                               Sequence length for patch16 224x224 is 197 (196 patches + 1 CLS token)
            attentions: List of attention matrices from each layer, useful for XAI.
        """
        outputs = self.vit(pixel_values=pixel_values)
        return outputs.last_hidden_state, outputs.attentions
