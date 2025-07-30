from devbench.eval_model import EvalModel
from tqdm import tqdm
import torch
import numpy as np
from PIL import Image

class BridgetowerEvalModel(EvalModel):
    def __init__(self, model, image_model=None, image_processor = None, processor=None, device="cpu"):
        self.device = device
        self.model = model.to(device)
        self.processor = processor
        self.image_processor = image_processor
        self.image_model = image_model
        self.get_image_features = self.get_all_image_feats
        self.get_text_features = self.get_all_text_feats
        self.get_similarity_scores = self.get_all_sim_scores

    def get_all_image_feats(self, dataloader):
        """
        Gets image embeddings from a dataloader using a model that outputs embeddings.
        
        Inputs:
        - dataloader: a dataloader constructed from a DevBenchDataset
        - processor: the appropriate input processor for the model
        - model: the model used to extract image embeddings
        
        Outputs:
        - a numpy array of shape [num_images, embed_dim]
        """
        all_feats = []
        with torch.no_grad():
            for d in tqdm(dataloader, desc="Processing data"):
                # Process each image individually
                for image in d["images"]:
                    # Convert image to RGB if not already in that format
                    if image.mode != 'RGB':
                        image = image.convert('RGB')
                    
                    # Pass a blank text to satisfy model requirements
                    inputs = self.image_processor(images=[image], text=[""], 
                                            return_tensors="pt", padding=True, truncation=True)
                    
                    # Model inference
                    outputs = self.image_model(**inputs)
                    
                    # Extract image features
                    image_features = outputs.image_features  # Shape: (batch_size, image_sequence_length, hidden_size)
                    
                    # Average pooling over the sequence length dimension to get (batch_size, hidden_size)
                    pooled_feats = image_features.mean(dim=1).squeeze().detach().numpy()
                    #print(pooled_feats.shape)
                    # Ensure the feature dimension is as expected and add to the list
                    if len(pooled_feats.shape) == 1:  # When batch_size is 1
                        all_feats.append(pooled_feats)
                    elif len(pooled_feats.shape) == 2:  # General case
                        all_feats.extend(pooled_feats)
                    else:
                        print(f"Unexpected shape of pooled features: {pooled_feats.shape}")
        
        return np.array(all_feats)

    def get_all_text_feats(self, dataloader):
        """
        Gets text features from a dataloader using a model that outputs logits
        -----
        Inputs:
        - dataloader: a dataloader constructed from a DevBenchDataset
        - processor: the appropriate input processor for the model
        - model: the model used to extract text features
        Outputs:
        - a numpy array of shape [num_texts, embed_dim]
        """
        # Create a blank (black) image with RGB channels and size 224x224
        blank_image = Image.new('RGB', (224, 224), (0, 0, 0))

        all_feats = []
        with torch.no_grad():
            for d in tqdm(dataloader, desc="Processing data"):
                # Use a blank image for each text input
                inputs = self.processor(images=blank_image, text=d["text"], 
                                        return_tensors="pt", padding=True, 
                                        truncation=True, max_length=512)
                outputs = self.model(**inputs)
                # Assuming the relevant text features are in the first position of logits
                feats = outputs.logits[:, 0].detach().numpy()  # Adjust indexing if necessary
                all_feats.append(feats)
        return np.concatenate(all_feats, axis=0)

    def get_all_sim_scores(self, dataloader):
        """
        Gets image--text similarity scores from a dataloader using Bridge Tower model
        -----
        Inputs:
        - dataloader: a dataloader constructed from a DevBenchDataset
        - processor: the BridgeTowerProcessor
        - model: the BridgeTowerModel
        Outputs:
        - a numpy array of shape [num_trials, num_images_per_trial, num_texts_per_trial]
        """
        all_sims = []
        with torch.no_grad():
            for d in tqdm(dataloader, desc="Processing data"):
                # Prepare inputs with padding and truncation
                # Assuming each data point in the dataloader has multiple images and texts
                num_images = len(d["images"])
                num_texts = len(d["text"])
                sims = np.zeros((num_images, num_texts))

                for i, image in enumerate(d["images"]):
                    for j, text in enumerate(d["text"]):
                        # Prepare inputs for each image-text pair
                        inputs = self.processor(images=image, text=text, return_tensors="pt", 
                                           padding=True, truncation=True)
                        # Forward pass
                        outputs = self.model(**inputs)
                        sims[i, j] = outputs.logits[0, 1].item()

                # Append the similarity scores for this batch to all_sims
                all_sims.append(sims)
        
        return np.stack(all_sims, axis=0)
