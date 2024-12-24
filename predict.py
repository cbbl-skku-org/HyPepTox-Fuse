from src.architectures import HyPepToxFuse_Hybrid
from src.utils import set_nested_attr
from argparse import Namespace
import yaml
import torch
from torch.nn.functional import softmax
import os

class HyPepToxFuse_Predictor:
    def __init__(self, model_config, ckpt_dir, nfold, device):
        self.models = [HyPepToxFuse_Hybrid(**model_config) for _ in range(nfold)]
        self.device = device
        
        for ckpt_file, model in zip(os.listdir(ckpt_dir), self.models):
            if not ckpt_file.endswith('.pth'):
                continue
            
            pt_file_path = os.path.join(ckpt_dir, ckpt_file)
            self._load_model_from_checkpoint(pt_file_path, model)
    
    def _load_model_from_checkpoint(self, checkpoint_path, model):
        state_dict = torch.load(checkpoint_path, map_location=self.device)
        model.load_state_dict(state_dict, strict=False)
        model.eval()
        model.to(self.device)
        
    def predict_one(self, f1, f2, f3, fccd, threshold=0.5):
        f1 = f1.to(self.device)
        f2 = f2.to(self.device)
        f3 = f3.to(self.device)
        all_prob = []
        for model in self.models:
            logit, _ = model.forward(f1, f2, f3, fccd)
            prob = softmax(logit, dim=-1)[:, 1]
            all_prob.append(prob)

        # Stack all_prob to tensor of shape(5,)
        all_prob = torch.stack(all_prob).squeeze(1)
        is_toxic = (all_prob >= threshold).sum() >= 3
        return is_toxic.item(), all_prob.tolist()
    

    def __call__(self, f1s, f2s, f3s, fccds, threshold=0.5):
        toxic_predictions = []
        probs = []
        
        for f1, f2, f3, fccd in zip(f1s, f2s, f3s, fccds):
            is_toxic, prob = self.predict_one(f1, f2, f3, fccd, threshold)
            toxic_predictions.append(is_toxic)
            probs.append(prob)
            
        return toxic_predictions, probs

    