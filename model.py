import torch
from transformers import DetrForObjectDetection, DetrConfig

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(23)

class DETR(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
        self.config = DetrConfig.from_pretrained("facebook/detr-resnet-50")
        self.config.num_labels = 6
        
        self.model = DetrForObjectDetection.from_pretrained(
            "facebook/detr-resnet-50",
            config=self.config,
            ignore_mismatched_sizes=True
        )
        
        self.backbone = self.model.model.backbone
        self.input_proj = self.model.model.input_projection
        self.query_mbed = self.model.model.query_position_embeddings
        self.class_mbed = self.model.class_labels_classifier
        self.bbox_mbed = self.model.bbox_predictor
        self.encoder = self.model.model.encoder
        self.decoder = self.model.model.decoder
        self.custom_features = None

    def feature_hook(self, module, input, output):
        if self.custom_features is not None:
            return self.custom_features
        return output

    def forward(self, before_images, after_images, labels=None):
        device = next(self.parameters()).device

        b_ten = torch.stack([torch.from_numpy(img) for img in before_images]).permute(0, 3, 1, 2).float().to(device)
        a_ten = torch.stack([torch.from_numpy(img) for img in after_images]).permute(0, 3, 1, 2).float().to(device)

        mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
        std  = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
        b_ten = (b_ten / 255.0 - mean) / std
        a_ten = (a_ten / 255.0 - mean) / std

        batch_size, _, h, w = b_ten.shape
        pixel_mask = torch.ones((batch_size, h, w), device=device)

        with torch.no_grad():
            feat_b_out = self.backbone(b_ten, pixel_mask)
            feat_a_out = self.backbone(a_ten, pixel_mask)

        f_b_list = feat_b_out[0]
        f_a_list = feat_a_out[0]

        class FakeFeat:
            def __init__(self, t, m):
                self.tensors = t
                self.mask = m
            def __iter__(self):
                yield self.tensors
                yield self.mask

        new_feature_list = []
        for fb, fa in zip(f_b_list, f_a_list):
            if hasattr(fb, 'tensors'):
                tens_b = fb.tensors
                tens_a = fa.tensors
                mask = fa.mask
            else:
                tens_b, _ = fb
                tens_a, mask = fa

            delta_feat = torch.abs(tens_a - tens_b)

            new_feature_list.append(FakeFeat(delta_feat, mask))

        if isinstance(feat_a_out, tuple):
            self.custom_features = (new_feature_list,) + feat_a_out[1:]
        else:
            self.custom_features = feat_a_out
            self.custom_features.feature_maps = new_feature_list

        handle = self.backbone.register_forward_hook(self.feature_hook)
        outputs = self.model(pixel_values=a_ten, pixel_mask=pixel_mask, labels=labels, return_dict=True)
        handle.remove()
        self.custom_features = None

        if labels is not None:
            return outputs.loss

        return {
            "logits": outputs.logits,
            "boxes": outputs.pred_boxes
        }

    def fine_tune3(self):
        for p in self.backbone.parameters():
            p.requires_grad = False
        for p in self.input_proj.parameters():
            p.requires_grad = False
        for p in self.encoder.parameters():
            p.requires_grad = False
        for p in self.decoder.parameters():
            p.requires_grad = False
        for p in self.query_mbed.parameters():
            p.requires_grad = False
        for p in self.class_mbed.parameters():
            p.requires_grad = True
        for p in self.bbox_mbed.parameters():
            p.requires_grad = True

        return [
            {
                "params": [p for n, p in self.named_parameters() if p.requires_grad],
                "lr": 1e-4,
            }
        ]

    def fine_tune4(self):
        for p in self.backbone.parameters():
            p.requires_grad = False
        for p in self.input_proj.parameters():
            p.requires_grad = True          
        for p in self.query_mbed.parameters():
            p.requires_grad = True          

        for p in self.class_mbed.parameters():
            p.requires_grad = True
        for p in self.bbox_mbed.parameters():
            p.requires_grad = True
        for p in self.encoder.parameters():
            p.requires_grad = True
        for p in self.decoder.parameters():
            p.requires_grad = True

        return [
            {
                "params": [p for n, p in self.named_parameters()
                           if ("encoder" in n or "decoder" in n or "input_proj" in n or "query_mbed" in n)
                           and p.requires_grad],
                "lr": 1e-5,
            },
            {
                "params": [p for n, p in self.named_parameters()
                           if ("class_mbed" in n or "bbox_mbed" in n)
                           and p.requires_grad],
                "lr": 1e-4,
            },
        ]
