import torch
import torch.nn as nn
from .vision_encoder import VisionEncoder
from .fusion import MultimodalFusion
from .llm_generator import LLMReportGenerator

class VLMLeukemiaModel(nn.Module):
    """
    Top-level Vision-Language Model integrating ViT, Fusion, and LLM.
    """
    def __init__(self, num_classes=2):
        super().__init__()
        
        # 1. Vision Encoder
        self.vision_encoder = VisionEncoder()
        
        # 2. LLM Generator (and its embedder)
        self.llm_generator = LLMReportGenerator()
        
        # 3. Multimodal Fusion Module
        # ViT-base hidden dim = 768, GPT-2 hidden dim = 768
        self.fusion = MultimodalFusion(visual_dim=self.vision_encoder.hidden_size, 
                                       text_dim=self.llm_generator.hidden_size)
        
        # 4. Classification Head (predicts Leukemia vs Healthy directly from fused output)
        self.clf_head = nn.Sequential(
            nn.Linear(self.vision_encoder.hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )

    def forward(self, images, text_input_ids, text_attention_mask=None, labels=None):
        """
        Args:
            images: Batch of images (B, C, H, W)
            text_input_ids: Prompt tokens "Analyze this blood smear:"
        """
        # A. Visual Feature Extraction
        visual_embeds, attentions = self.vision_encoder(images)
        
        # B. Text Prompt Embedding
        text_embeds = self.llm_generator.llm.transformer.wte(text_input_ids)
        
        # C. Multimodal Fusion
        # Fused representation grounded in the clinical prompt
        fused_embeds = self.fusion(visual_embeds, text_embeds)
        
        # D. Classification
        # Use a pooled representation (e.g., mean of fused tokens) for classification
        pooled_fused = torch.mean(fused_embeds, dim=1) 
        logits = self.clf_head(pooled_fused)
        
        # E. Text Generation / Modeling (Training Phase)
        # We append the actual report to the prompt during training 
        # and calculate standard Causal LM loss.
        # For simplicity here, we pass the fused outputs through the LLM 
        # to calculate language modeling loss.
        llm_outputs = self.llm_generator(inputs_embeds=fused_embeds, labels=labels)
        lm_loss = llm_outputs.loss
        
        return logits, lm_loss, attentions
