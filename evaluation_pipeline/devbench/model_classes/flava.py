from devbench.eval_model import EvalModel
from tqdm import tqdm
import torch
import numpy as np

class FlavaEvalModel(EvalModel):
    def __init__(self, model, processor=None, image_model=None, feature_extractor=None, device="cpu"):
        self.device = device
        self.model = model.to(device)
        self.processor = processor
        self.image_model = image_model
        self.feature_extractor = feature_extractor

        self.get_image_features = self.get_all_image_feats
        self.get_text_features = self.get_all_text_feats
        self.get_similarity_scores = self.get_all_sim_scores

    def get_all_image_feats(self, dataloader):
        """
        Gets image features from a dataloader and applies mean pooling to each set of image features.
        -----
        Inputs:
        - dataloader: a dataloader constructed from a DevBenchDataset
        - fe: the feature extractor
        - model: the model used to extract image features
        Outputs:
        - a numpy array of shape [num_images, embed_dim] after mean pooling
        """
        all_feats = []
        with torch.no_grad():
            for d in tqdm(dataloader, desc="Processing data"):
                images_rgb = [image.convert("RGB") for image in d["images"]]
                image_input = self.feature_extractor(images_rgb, return_tensors="pt")
                feats = self.image_model.get_image_features(**image_input).detach().numpy()
                mean_feats = np.mean(feats, axis=1)  # Mean pooling across the patches
                all_feats.append(mean_feats)
        return np.concatenate(all_feats, axis=0)

    def get_all_text_feats(self, dataloader):
        """
        Gets text features from a dataloader using the FLAVA model
        -----
        Inputs:
        - dataloader: a dataloader constructed from a DevBenchDataset
        - processor: the appropriate input processor for the model
        - model: the FLAVA model

        Outputs:
        - a list of numpy arrays, where each array represents the text features for a data point
        """
        all_feats = []
        with torch.no_grad():
            for d in tqdm(dataloader, desc="Processing data"):
                inputs = self.processor(text=d["text"], return_tensors="pt", padding=True)
                text_features = self.image_model.get_text_features(**inputs)[:, 0].float()
                all_feats.extend(text_features)
        return all_feats
    

    def get_all_sim_scores(self, dataloader):
        all_sims = []
        with torch.no_grad():
            for d in tqdm(dataloader, desc="Processing data"):
                images_rgb = [image.convert("RGB") for image in d["images"]]
                texts = [d["text"][0]] * len(images_rgb)  
                inputs = self.processor(text=texts, images=images_rgb, 
                                        return_tensors="pt", max_length=77, 
                                        padding=True, return_codebook_pixels=True, 
                                        return_image_mask=True)
                outputs = self.model(**inputs)
                scores = outputs.contrastive_logits_per_image[:, 0].view(-1, 1).numpy()
                all_sims.append(scores)
        return np.array(all_sims)