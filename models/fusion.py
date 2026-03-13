import torch
import torch.nn as nn

class MultimodalFusion(nn.Module):
    """
    Fuses visual embeddings from ViT with contextual embeddings (e.g., a clinical prompt).
    Uses Cross-Attention to align visual and text representations.
    """
    def __init__(self, visual_dim=768, text_dim=768, num_heads=8):
        super().__init__()
        self.cross_attention = nn.MultiheadAttention(embed_dim=visual_dim, num_heads=num_heads, batch_first=True)
        
        # Projection layer if text_dim differs from visual_dim
        self.text_proj = nn.Linear(text_dim, visual_dim) if text_dim != visual_dim else nn.Identity()
        
        self.layer_norm1 = nn.LayerNorm(visual_dim)
        self.layer_norm2 = nn.LayerNorm(visual_dim)
        
        self.ffn = nn.Sequential(
            nn.Linear(visual_dim, visual_dim * 4),
            nn.GELU(),
            nn.Linear(visual_dim * 4, visual_dim)
        )

    def forward(self, visual_embeds, text_embeds):
        """
        Args:
            visual_embeds: (batch, seq_len_v, visual_dim) - From ViT
            text_embeds: (batch, seq_len_t, text_dim) - From LLM prompt embedder
        Returns:
            fused_embeds: (batch, seq_len_t, visual_dim)
        """
        text_embeds_proj = self.text_proj(text_embeds)
        
        # Cross Attention: Text queries Visual information
        # Query: Text, Key: Visual, Value: Visual
        attn_out, _ = self.cross_attention(query=text_embeds_proj, 
                                           key=visual_embeds, 
                                           value=visual_embeds)
        
        # Add and Norm
        out1 = self.layer_norm1(text_embeds_proj + attn_out)
        
        # FFN
        ffn_out = self.ffn(out1)
        
        # Add and Norm
        fused_embeds = self.layer_norm2(out1 + ffn_out)
        return fused_embeds
