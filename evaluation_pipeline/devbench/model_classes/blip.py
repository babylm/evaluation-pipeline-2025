from devbench.eval_model import EvalModel
from tqdm import tqdm
import torch

class BlipEvalModel(EvalModel):
    def __init__(self, model, processor=None, image_model=None, device="cpu"):
        self.device = torch.device(device)  
        self.model = model.to(self.device)  
        self.processor = processor
        self.image_model = image_model

        self.get_image_features = self.image_model.get_image_features
        self.get_all_text_feats = self.get_all_text_feats
        self.get_similarity_scores = lambda **x: self.model(**x).itm_score

    def get_all_text_feats(self, data_loader):
        all_text_feats = []
        with torch.no_grad():
            for d in tqdm(data_loader, desc="Processing data"):
                inputs = self.processor(text=d["text"], padding=True, return_tensors="pt").to(self.device)
                text_features = self.image_model.get_text_features(**inputs)
                all_text_feats.append(text_features)

        all_text_feats = torch.cat(all_text_feats, dim=0)
        return all_text_feats
