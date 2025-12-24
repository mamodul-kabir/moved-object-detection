import os
import cv2
import torch

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(23)

def scan(anno): 
    image_dir = "../data/base/cv_data_hw2/data"
    samples = []
    for f in os.listdir(anno):
        folder,f1,f2 = os.path.basename(f).split('.')[0].split('-')
        f2 = f2.split("_match")[0];
        f1,f2 = map(lambda x: f"{image_dir}/{folder}/{x}.png", [f1,f2])
        full_f = f"{anno}/{f}"
        samples.append((f1, f2, full_f))
    return samples

class Dataset: 
    def __init__(self, dir):
        self.samples = scan(dir)
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        before_img_path, after_img_path, ann_path = self.samples[idx]
        im1, im2 = map(cv2.imread, [before_img_path, after_img_path])
        
        h_orig, w_orig, _ = im1.shape

        target_size = (800, 800)
        im1, im2 = map(lambda x: cv2.resize(x, target_size), [im1, im2])
        im1, im2 = map(lambda x: cv2.cvtColor(x, cv2.COLOR_BGR2RGB), [im1, im2])
        
        boxes = []
        labels = []
        with open(ann_path, "r") as file: 
            for f in file: 
                _,x,y,w,h,type = f.split()
                x,y,w,h,type = map(int, [x,y,w,h,type])
                
                x = x * (target_size[0] / w_orig)
                w = w * (target_size[0] / w_orig)
                y = y * (target_size[1] / h_orig)
                h = h * (target_size[1] / h_orig)
                
                x_max = x + w
                y_max = y + h
                boxes.append((x, y, x_max, y_max))
                labels.append(type)

        boxes = torch.tensor(boxes, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.int64)
        item = {
            "image_before" : im1, 
            "image_after"  : im2, 
            "target" : {
                "boxes" : boxes, 
                "labels" : labels
            }
        }
        return item

def customCollate_fn(batch):
    before_imgs = [sample["image_before"] for sample in batch]
    after_imgs = [sample["image_after"] for sample in batch]
    targets = [sample["target"] for sample in batch]
    output = {
        "images_before" : before_imgs, 
        "images_after" : after_imgs, 
        "targets" : targets
    }
    return output


if __name__=="__main__":
    print(scan('../data/matched_annotations'))
