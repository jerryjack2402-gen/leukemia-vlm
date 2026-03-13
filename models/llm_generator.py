import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel, GPT2Config, GPT2Tokenizer

class LLMReportGenerator(nn.Module):
    """
    Medical Large Language Model for generating pathology reports.
    Takes fused multimodal embeddings as input prompts to autoregressively
    generate text and classification.
    """
    def __init__(self, model_name="gpt2", max_length=150):
        super().__init__()
        # Using a small GPT-2 model as a scaffold for the LLM
        # In a real deployed medical system, a specialized model like MedLLaMA is preferred
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.llm = GPT2LMHeadModel.from_pretrained(model_name)
        
        # Dimensions for projection if needed
        self.hidden_size = self.llm.config.n_embd
        self.max_length = max_length

    def forward(self, input_ids=None, attention_mask=None, inputs_embeds=None, labels=None):
        """
        Args:
            input_ids: Input token IDs for text.
            attention_mask: Mask for padding tokens.
            inputs_embeds: Fused embeddings directly fed into the LLM 
                           (overrides input_ids for the prompt part).
            labels: Ground truth token IDs for calculating language modeling loss.
        """
        # If inputs_embeds is provided (e.g. from multimodal fusion), we use that 
        # instead of the standard input_ids embedding lookup.
        if inputs_embeds is not None:
            outputs = self.llm(inputs_embeds=inputs_embeds, 
                               attention_mask=attention_mask, 
                               labels=labels)
        else:
            outputs = self.llm(input_ids=input_ids, 
                               attention_mask=attention_mask, 
                               labels=labels)
        
        return outputs

    def generate(self, inputs_embeds, max_new_tokens=100):
        """
        Generates text given fused embeddings.
        """
        # Note: Direct inputs_embeds to huggingface generate() needs a custom loop 
        # or specific wrapper depending on the transformers version.
        # Here we provide a simplified conceptual pass:
        outputs = self.llm.generate(inputs_embeds=inputs_embeds, 
                                    max_new_tokens=max_new_tokens,
                                    pad_token_id=self.tokenizer.eos_token_id)
        return outputs
