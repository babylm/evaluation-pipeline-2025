from devbench.eval_model import EvalModel
from tqdm import tqdm
import torch
import numpy as np
from PIL import Image

class SiglipEvalModel(EvalModel):
    def __init__(self, model, processor=None, tokenizer=None, model_embed=None, processor_embed = None, device="cpu"):
        self.device = device
        self.model = model.to(device)
        self.processor = processor
        self.tokenizer = tokenizer
        self.model_embed = model_embed.to(device)
        self.processor_embed = processor_embed
        self.get_image_features = self.get_all_image_feats
        self.get_text_features = self.get_all_text_feats
        self.get_similarity_scores = self.get_all_sim_scores

    def get_all_image_feats(self, dataloader):
        all_feats = []
        with torch.no_grad():
            for d in tqdm(dataloader, desc="Processing data"):
                inputs = self.processor_embed(images=d['images'], return_tensors="pt")
                image_features = self.model_embed.get_image_features(**inputs)
                all_feats.append(image_features)
        return np.concatenate(all_feats, axis=0)

    def get_all_text_feats(self, dataloader):
        all_feats = []
        with torch.no_grad():
            for d in tqdm(dataloader, desc="Processing data"):
                inputs = self.tokenizer(d['text'], padding = "max_length", return_tensors = "pt")
                text_features = self.model_embed.get_text_features(**inputs)
        return text_features
    

    def get_all_sim_scores(self, dataloader):
        all_sims = []
        with torch.no_grad():
            for d in tqdm(dataloader, desc="Processing data"):
                # Process the batch for variable number of images and texts
                images_rgb = [img.convert("RGB") if isinstance(img, Image.Image) else img for img in d["images"]]
                inputs = self.processor(images=images_rgb, text=d["text"], 
                                return_tensors="pt", padding=True)
                
                outputs = self.model(**inputs)
                
                logits_per_image = outputs.logits_per_image  # Get logits for each image-text pair
                sims = torch.sigmoid(logits_per_image).detach().cpu().numpy()  # Convert to probabilities
                all_sims.append(sims)

        return all_sims