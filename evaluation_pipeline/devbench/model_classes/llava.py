from devbench.eval_model import EvalModel
from tqdm import tqdm
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms

class LlavaEvalModel(EvalModel):
    def __init__(self, model, processor=None, is_tiny=False, device="cpu"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.processor = processor
        self.is_tiny = is_tiny

    def get_all_sim_scores(self, dataloader):
        all_sims = []
        # Define a transform to convert PIL images to tensors
        transform = transforms.Compose([
            transforms.ToTensor()
        ])
        with torch.no_grad():
            for d in tqdm(dataloader, desc="Processing data"):
                trial_sims = []
                for image in d["images"]:
                    # Convert image to RGB if it's not already in that format
                    if image.mode != 'RGB':
                        image = image.convert('RGB')
                    # Convert the PIL Image to a PyTorch tensor
                    image_tensor = transform(image)
                    # Move tensor to the GPU
                    image_tensor = image_tensor.to(self.device)

                    for text in d['text']:
                        if self.is_tiny:
                            prompt = f"USER: <image>\nCaption: {text}. Does the caption match the image? Answer either Yes or No. \nASSISTANT:"
                        else:
                            prompt = f"[INST] <image>\nCaption: {text}. Does the caption match the image? Answer either Yes or No. [/INST]"
                        inputs = self.processor(text=prompt, images=image_tensor, return_tensors='pt').to(self.device)  # Ensure inputs are on GPU

                        logits = self.model(**inputs).logits.squeeze()
                        yes_token_id = self.processor.tokenizer.encode("Yes")[1]
                        no_token_id = self.processor.tokenizer.encode("No")[1]

                        yes_logits = logits[-1, yes_token_id]
                        no_logits = logits[-1, no_token_id]

                        pair_logits = torch.stack((yes_logits, no_logits), dim=0).cpu().numpy()
                        trial_sims.append(pair_logits)

                all_sims.append(np.array(trial_sims))

        return np.array(all_sims)