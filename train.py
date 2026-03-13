import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import os

from models.vlm_model import VLMLeukemiaModel
from data.dataset import BloodSmearDataset
from transformers import GPT2Tokenizer

def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on: {device}")

    # Hyperparameters
    epochs = 5
    batch_size = 8
    lr = 2e-5

    # Data
    train_dataset = BloodSmearDataset(data_dir="data_path", split="train")
    val_dataset = BloodSmearDataset(data_dir="data_path", split="val")
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Model
    model = VLMLeukemiaModel(num_classes=2).to(device)
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    
    # Loss and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr)

    # Standard clinical prompt
    prompt = "Analyze this microscopic blood smear and provide a structural pathology report detailing early signs of Leukemia."

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for images, labels in pbar:
            images = images.to(device)
            labels = labels.to(device)
            
            # Tokenize prompt to create input_ids
            inputs = tokenizer([prompt]*images.size(0), return_tensors="pt", padding=True).to(device)
            
            # Dummy labels for LLM report generation (in reality, actual reports from dataset)
            report_labels = inputs.input_ids.clone() 
            
            optimizer.zero_grad()
            
            # Forward pass
            logits, lm_loss, _ = model(images, inputs.input_ids, labels=report_labels)
            
            # Combine classification loss and language modeling loss
            clf_loss = criterion(logits, labels)
            
            # Weighting between classification and generation
            loss = clf_loss + (0.5 * lm_loss) if lm_loss is not None else clf_loss
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = logits.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            pbar.set_postfix({'loss': loss.item(), 'acc': 100.*correct/total})
            
        print(f"Epoch {epoch+1} Summary: Avg Loss: {total_loss/len(train_loader):.4f}, Acc: {100.*correct/total:.2f}%")
        
        # Save checkpoint
        os.makedirs("checkpoints", exist_ok=True)
        torch.save(model.state_dict(), f"checkpoints/vlm_model_epoch_{epoch+1}.pth")

if __name__ == "__main__":
    train()
