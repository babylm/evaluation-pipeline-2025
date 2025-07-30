from tqdm import tqdm
import torch
import numpy as np

class EvalModel():
    def __init__(self, model, processor=None, device="cpu"):
        self.device = device
        self.model = model.to(device)
        self.processor = processor
        self.get_image_features = self.model.get_image_features
        self.get_text_features = self.model.get_text_features
        self.get_similarity_scores = lambda **x: self.model(**x).logits_per_image
    
    def get_all_image_feats(self, dataloader):
        """
        Gets image features from a dataloader
        -----
        Inputs:
        - dataloader: a dataloader constructed from a DevBenchDataset
        - processor: the appropriate input processor for the model
        - get_image_features: the model or model attribute used to
        extract image features
        Outputs:
        - a numpy array of shape [num_images, embed_dim]
        """
        all_feats = []
        with torch.no_grad():
            for d in tqdm(dataloader, desc="Processing data"):
                inputs = self.processor(images=d["images"], return_tensors="pt").to(self.device)
                feats = self.get_image_features(**inputs).detach().numpy()
                all_feats.append(feats)
        return np.concatenate(all_feats, axis=0)

    def get_all_text_feats(self, dataloader):
        """
        Gets text features from a dataloader
        -----
        Inputs:
        - dataloader: a dataloader constructed from a DevBenchDataset
        - processor: the appropriate input processor for the model
        - get_text_features: the model or model attribute used to
        extract text features
        Outputs:
        - a numpy array of shape [num_texts, embed_dim]
        """
        all_feats = []
        with torch.no_grad():
            for d in tqdm(dataloader, desc="Processing data"):
                inputs = self.processor(text=d["text"], return_tensors="pt", 
                                padding=True).to(self.device)
                feats = self.get_text_features(**inputs).detach().numpy()
                all_feats.append(feats)
        return np.concatenate(all_feats, axis=0)
    
    def get_all_sim_scores(self, dataloader):
        """
        Gets image--text similarity scores from a dataloader
        -----
        Inputs:
        - dataloader: a dataloader constructed from a DevBenchDataset
        - processor: the appropriate input processor for the model
        - get_similarity_scores: the model or model attribute used to
        obtain similarity scores
        Outputs:
        - a numpy array of shape [num_trials, num_images_per_trial, 
        num_texts_per_trial]
        """
        all_sims = []
        with torch.no_grad():
            for d in tqdm(dataloader, desc="Processing data"):
		# inputs = self.processor(images=d["images"], return_tensors="pt").to(self.device)
                # images_tensor = torch.stack(d["images"]).to(self.device)
                inputs = self.processor(images=d["images"], text=d["text"], 
                                        return_tensors="pt", padding=True)
                sims = self.get_similarity_scores(**inputs).detach().numpy()
                all_sims.append(sims)
        return np.stack(all_sims, axis=0)
