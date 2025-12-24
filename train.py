import torch
from torch.utils.data import DataLoader
from transformers import DetrImageProcessor
import matplotlib.pyplot as plt
import os
import time
from eval import visualize_results, evaluate_metrics
from tqdm import tqdm  

from dataload import Dataset, customCollate_fn
from model import DETR

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(23)
print("Device:", DEVICE)

def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

def convert_targets_for_detr(batch_targets, orig_sizes):
    detr_targets = []

    for i, tgt in enumerate(batch_targets):
        og_h, og_w = orig_sizes[i]  

        boxes_xyxy = tgt["boxes"].clone().float()

        boxes_xyxy[:, [0, 2]] /= og_w     
        boxes_xyxy[:, [1, 3]] /= og_h     

        x_min = boxes_xyxy[:, 0]
        y_min = boxes_xyxy[:, 1]
        x_max = boxes_xyxy[:, 2]
        y_max = boxes_xyxy[:, 3]

        w = x_max - x_min
        h = y_max - y_min
        
        cx = x_min + (w / 2)
        cy = y_min + (h / 2)

        boxes_cxcywh = torch.stack([cx, cy, w, h], dim=-1)

        labels = tgt["labels"].long() 

        detr_targets.append({
            "class_labels": labels.to(DEVICE), 
            "labels": labels.to(DEVICE), 
            "boxes": boxes_cxcywh.to(DEVICE)
        })

    return detr_targets

ds = Dataset('../data/matched_annotations')

total_len = len(ds)
train_val_size = int(0.8 * total_len)
test_size = total_len - train_val_size

train_val_set, test_set = torch.utils.data.random_split(
    ds, [train_val_size, test_size], generator=torch.Generator().manual_seed(23)
)

train_len = int(0.9 * len(train_val_set))
val_len = len(train_val_set) - train_len

train_set, val_set = torch.utils.data.random_split(
    train_val_set, [train_len, val_len], generator=torch.Generator().manual_seed(23)
)

train_loader = DataLoader(train_set, batch_size=8, shuffle=True, collate_fn=customCollate_fn, pin_memory=True)
val_loader = DataLoader(val_set, batch_size=8, shuffle=False, collate_fn=customCollate_fn, pin_memory=True)
test_loader = DataLoader(test_set, batch_size=8, shuffle=False, collate_fn=customCollate_fn, pin_memory=True)


processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")

detr = DETR()
detr = detr.to(DEVICE)

trainable_params = detr.fine_tune3()   
#for Experiment B, comment the previous line (line 81) and uncomment the next line (line 83)
#trainable_params = detr.fine_tune4()

print(f"Trainable Parameters: {count_parameters(detr):,}")

optimizer = torch.optim.AdamW(
    trainable_params,
    weight_decay=1e-4
)

lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=275, gamma=0.1)

NUM_EPOCHS = 360

print(f"Starting training on device: {DEVICE}")
print(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)} | Test batches: {len(test_loader)}")

val_loss_history = []
train_loss_history = []
total_time = 0

for epoch in range(NUM_EPOCHS):
    start_time = time.time()
    
    detr.train()
    train_loss_accum = 0.0
    
    train_loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Train]")
    
    for i, batch in enumerate(train_loop):
        bf_imgs = batch["images_before"]
        af_imgs = batch["images_after"]
        targets = batch["targets"]
        og_sizes = [img.shape[0:2] for img in af_imgs]

        conv_tgt = convert_targets_for_detr(targets, og_sizes)

        loss = detr(bf_imgs, af_imgs, labels=conv_tgt)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        current_loss = loss.item()
        train_loss_accum += current_loss

        train_loop.set_postfix(loss=current_loss)

    avg_train_loss = train_loss_accum / len(train_loader)

    detr.eval() 
    val_loss_accum = 0.0

    val_loop = tqdm(val_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Val]", leave=False)

    with torch.no_grad():
        for i, batch in enumerate(val_loop):
            bf_imgs = batch["images_before"]
            af_imgs = batch["images_after"]
            targets = batch["targets"]
            og_sizes = [img.shape[0:2] for img in af_imgs]

            conv_tgt = convert_targets_for_detr(targets, og_sizes)

            loss = detr(bf_imgs, af_imgs, labels=conv_tgt)
            
            val_loss_accum += loss.item()
            
            val_loop.set_postfix(loss=loss.item())

    avg_val_loss = val_loss_accum / len(val_loader)
    
    lr_scheduler.step()
    current_lr = optimizer.param_groups[0]['lr'] 
    
    elapsed = time.time() - start_time
    total_time += elapsed
    print(f"Epoch {epoch+1} Completed in {elapsed:.1f}s | LR: {current_lr:.2e}")
    print(f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
    print("-" * 60)

    val_loss_history.append(avg_val_loss)
    train_loss_history.append(avg_train_loss)

print("Training Complete.")
print(f"Total time: {total_time:.1f}s")


print("Starting Final Evaluation on Test Set...")
evaluate_metrics(detr, test_loader, DEVICE)
visualize_results(detr, test_loader, DEVICE, output_dir="debug_pred4")