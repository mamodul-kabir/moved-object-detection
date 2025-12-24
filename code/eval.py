import torch
import torch.nn.functional as F
import cv2
import numpy as np
import torchvision
import os
from torchmetrics.detection.mean_ap import MeanAveragePrecision

def calculate_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)

    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    iou = interArea / float(boxAArea + boxBArea - interArea + 1e-6)
    return iou

def calculate_point_metrics(preds, targets, conf_threshold=0.5, iou_threshold=0.5):
    tp = 0
    fp = 0
    fn = 0

    for p, t in zip(preds, targets):
        pred_boxes = p['boxes']
        pred_scores = p['scores']
        pred_labels = p['labels']
        
        gt_boxes = t['boxes']
        gt_labels = t['labels']

        keep_idxs = pred_scores > conf_threshold
        pred_boxes = pred_boxes[keep_idxs]
        pred_labels = pred_labels[keep_idxs]

        if len(gt_boxes) == 0:
            fp += len(pred_boxes)
            continue
        
        if len(pred_boxes) == 0:
            fn += len(gt_boxes)
            continue

        matched_gt = set()
        
        sorted_idxs = torch.argsort(pred_scores[keep_idxs], descending=True)
        pred_boxes = pred_boxes[sorted_idxs]
        pred_labels = pred_labels[sorted_idxs]

        for i in range(len(pred_boxes)):
            p_box = pred_boxes[i]
            p_label = pred_labels[i]
            
            best_iou = 0.0
            best_gt_idx = -1

            for j in range(len(gt_boxes)):
                if j in matched_gt:
                    continue
                
                if p_label != gt_labels[j]:
                    continue

                iou = torchvision.ops.box_iou(p_box.unsqueeze(0), gt_boxes[j].unsqueeze(0)).item()
                
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = j

            if best_iou >= iou_threshold and best_gt_idx != -1:
                tp += 1
                matched_gt.add(best_gt_idx)
            else:
                fp += 1
        
        fn += (len(gt_boxes) - len(matched_gt))

    epsilon = 1e-6
    precision = tp / (tp + fp + epsilon)
    recall = tp / (tp + fn + epsilon)
    f1 = 2 * (precision * recall) / (precision + recall + epsilon)

    return precision, recall, f1, (tp, fp, fn)


def evaluate_metrics(model, data_loader, device, conf_threshold=0.5, iou_threshold=0.5):
    print("\nStarting Detailed Evaluation...")
    model.eval()
    
    coco_metric = MeanAveragePrecision(box_format='xyxy', iou_type='bbox', class_metrics=True)
    coco_metric.to(device)
    
    all_preds = []
    all_targets = []
    
    global_max_score = 0.0

    with torch.no_grad():
        for i, batch in enumerate(data_loader):
            bf_imgs = batch["images_before"]
            af_imgs = batch["images_after"]
            targets = batch["targets"]
            
            outputs = model(bf_imgs, af_imgs, labels=None) 
            
            pred_logits = outputs["logits"]
            pred_boxes = outputs["boxes"]

            batch_preds = []
            batch_targets = []

            for logits, box in zip(pred_logits, pred_boxes):
                probs = F.softmax(logits, dim=-1)
                scores, labels = probs[..., :-1].max(dim=-1)
                
                batch_max = scores.max().item()
                if batch_max > global_max_score:
                    global_max_score = batch_max

                x_c, y_c, w, h = box.unbind(-1)
                b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
                     (x_c + 0.5 * w), (y_c + 0.5 * h)]
                boxes_xyxy = torch.stack(b, dim=-1) * 800.0
                
                batch_preds.append({
                    "boxes": boxes_xyxy.cpu(),
                    "scores": scores.cpu(),
                    "labels": labels.cpu()
                })

            for t in targets:
                batch_targets.append({
                    "boxes": t["boxes"].cpu(),
                    "labels": t["labels"].cpu()
                })
            

            tm_preds = [{k: v.to(device) for k, v in x.items()} for x in batch_preds]
            tm_targets = [{k: v.to(device) for k, v in x.items()} for x in batch_targets]
            
            coco_metric.update(tm_preds, tm_targets)
            
            all_preds.extend(batch_preds)
            all_targets.extend(batch_targets)
            
            if i % 5 == 0:
                print(f"Eval Batch {i} processed. Max score seen: {global_max_score:.4f}")

    print(f"\nGlobal Max Score across dataset: {global_max_score:.4f}")

    results = coco_metric.compute()
    
    print(f"\n--- Point Metrics (Conf={conf_threshold}, IoU={iou_threshold}) ---")
    print("Calculating strict P/R based on TP/FP/FN counts...")
    
    p_val, r_val, f1_val, counts = calculate_point_metrics(all_preds, all_targets, conf_threshold, iou_threshold)
    
    print(f"  True Positives:  {counts[0]}")
    print(f"  False Positives: {counts[1]}")
    print(f"  False Negatives: {counts[2]}")
    print(f"  Strict Precision: {p_val:.4f}")
    print(f"  Strict Recall:    {r_val:.4f}")
    print(f"  Strict F1-Score:  {f1_val:.4f}")

    print(f"\n--- COCO Standard Metrics (Averaged over IoU 0.5:0.95) ---")
    print(f"  mAP (Mean Average Precision): {results['map']:.4f}")
    print(f"  mAR (Mean Average Recall):    {results['mar_100']:.4f}")
    print(f"  mAP_50:                       {results['map_50']:.4f}")
    
    if 'map_per_class' in results:
        class_map = {0:'Unknown', 1:'Person', 2:'Car', 3:'Other Vehicle', 4:'Other', 5:'Bike'}
        print("\n--- Per Class mAP ---")
        for i, score in enumerate(results['map_per_class']):
            class_id = int(results['classes'][i].item())
            c_name = class_map.get(class_id, f"Class {class_id}")
            print(f"  {c_name}: {score:.4f}")

    final_stats = {
        'precision': p_val,
        'recall': r_val,
        'f1': f1_val,
        'map': results['map'].item(),
        'mar': results['mar_100'].item(),
        'map_50': results['map_50'].item()
    }
    return final_stats

def visualize_results(model, data_loader, device, output_dir="debug_preds", conf_threshold=0.5, iou_threshold=0.2):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    print(f"Visualizing batch to {output_dir}...")
    model.eval()
    
    try:
        batch = next(iter(data_loader))
    except StopIteration:
        print("DataLoader is empty!")
        return

    bf_imgs = batch["images_before"]
    af_imgs = batch["images_after"]
    targets = batch["targets"] 
    
    with torch.no_grad():
        outputs = model(bf_imgs, af_imgs, labels=None)
        pred_logits = outputs["logits"]
        pred_boxes = outputs["boxes"]
    
    batch_size = len(batch["images_before"])
    
    for i in range(batch_size):
        img_bf_np = batch["images_before"][i].copy()
        img_af_np = batch["images_after"][i].copy()
        img_h, img_w, _ = img_af_np.shape
        
        img_bf_np = cv2.cvtColor(img_bf_np, cv2.COLOR_RGB2BGR)
        img_af_np = cv2.cvtColor(img_af_np, cv2.COLOR_RGB2BGR)

        logits = pred_logits[i]
        boxes = pred_boxes[i]
        
        probs = F.softmax(logits, dim=-1)
        scores, labels = probs[..., :-1].max(dim=-1)
        
        keep_indices = scores > conf_threshold
        keep_boxes = boxes[keep_indices]
        keep_scores = scores[keep_indices]
        keep_labels = labels[keep_indices]

        if len(keep_boxes) > 0:
            xc, yc, w, h = keep_boxes.unbind(-1)
            x1 = (xc - 0.5 * w) * img_w
            y1 = (yc - 0.5 * h) * img_h
            x2 = (xc + 0.5 * w) * img_w
            y2 = (yc + 0.5 * h) * img_h
            boxes_xyxy = torch.stack([x1, y1, x2, y2], dim=1)

            nms_indices = torchvision.ops.nms(boxes_xyxy, keep_scores, iou_threshold)
            
            final_boxes = boxes_xyxy[nms_indices]
            final_scores = keep_scores[nms_indices]
            final_labels = keep_labels[nms_indices]

            for box, score, label in zip(final_boxes, final_scores, final_labels):
                bx1, by1, bx2, by2 = box.tolist()
                bx1, by1, bx2, by2 = int(bx1), int(by1), int(bx2), int(by2)
                cv2.rectangle(img_af_np, (bx1, by1), (bx2, by2), (0, 0, 255), 2)
                cv2.putText(img_af_np, f"{label.item()}: {score:.2f}", (bx1, by1 - 5), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        if targets is not None:
            tgt_boxes = targets[i]["boxes"]
            tgt_labels = targets[i]["labels"]
            for box, label in zip(tgt_boxes, tgt_labels):
                t_x1, t_y1, t_x2, t_y2 = box.int().tolist()
                cv2.rectangle(img_af_np, (t_x1, t_y1), (t_x2, t_y2), (0, 255, 0), 2)

        if img_bf_np.shape != img_af_np.shape:
             img_bf_np = cv2.resize(img_bf_np, (img_w, img_h))
             
        combined_img = np.hstack((img_bf_np, img_af_np))
        cv2.imwrite(os.path.join(output_dir, f"pred_batch_{i}.png"), combined_img)

    print(f"Done. Saved visualizations to {output_dir}/")
