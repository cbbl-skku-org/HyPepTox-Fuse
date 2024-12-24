import numpy as np
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR

import lightning as L

from .architectures import HyPepToxFuse_NLP, HyPepToxFuse_Hybrid
from .metrics import calculate_metrics
from .focal_loss import focal_loss
from .ntxent_loss import nt_bxent_loss

class HyPepToxFuse_NLP_Trainer(L.LightningModule):
    def __init__(self, model_config, trainer_config):
        super().__init__()
        self.model = HyPepToxFuse_NLP(**vars(model_config))
        self.trainer_config = trainer_config
        self.model_config = model_config
        
        self.loss_fn_name = self.trainer_config.loss_fn
        
        if self.loss_fn_name == 'focal':
            self.loss_fn = focal_loss(alpha=self.trainer_config.alpha_focal, gamma=self.trainer_config.beta_focal, device=self.device)
        else:
            self.loss_fn = F.binary_cross_entropy
            
        self.ntxent_loss = nt_bxent_loss
    
    def forward(self, X, mask_X):
        return self.model(X[0], X[1], X[2], mask_X[0], mask_X[1], mask_X[2])
    
    def training_step(self, batch, batch_idx):
        _, X, mask_X, y, pos_indices = batch
        logits, embeddings = self.forward(X, mask_X)
        
        if self.loss_fn_name == 'focal':
            y_probs = F.softmax(logits, dim=-1)
            losses = self.loss_fn(y_probs, y)
        else:
            y_probs = F.sigmoid(logits)
            losses = self.loss_fn(y_probs.squeeze(1), y.float())
        
        ntxent_losses_weight = 0.2
        losses_weight = 0.8
        
        ntxent_losses = self.ntxent_loss(embeddings, pos_indices, temperature=self.trainer_config.ntxent_temperature)
        
        total_losses = losses_weight*losses + ntxent_losses_weight*ntxent_losses
        
        self.log('train_loss', total_losses.item(), prog_bar=True, logger=True)
        return total_losses
    
    def on_validation_epoch_start(self):
        self.all_y = []
        self.all_y_pred = []
        self.all_probs = []
        
    def validation_step(self, batch, batch_idx):
        _, X, mask_X, y, _ = batch
            
        logits, _ = self.forward(X, mask_X)
                
        if self.loss_fn_name == 'focal':
            y_probs = F.softmax(logits, dim=-1)
            y_pred = torch.argmax(y_probs, dim=-1).unsqueeze(1)
        else:
            y_probs = F.sigmoid(logits)
            y_pred = (y_probs >= self.trainer_config.threshold).float()
        
        self.all_y.append(y.cpu().numpy())
        self.all_y_pred.extend(y_pred.cpu().numpy())
        self.all_probs.extend(y_probs.cpu().numpy())
        
    def on_validation_epoch_end(self):
        self.all_y = np.concatenate(self.all_y)
        self.all_y_pred = np.concatenate(self.all_y_pred)
        
        if self.model_config.n_classes == 1:
            self.all_probs = np.concatenate(self.all_probs)
        else:
            self.all_probs = [probs.max() for probs in self.all_probs]
            self.all_probs = np.array(self.all_probs)
            
            # reverse value of probability of class 0
            for i in range(len(self.all_y_pred)):
                if self.all_y_pred[i] == 0:
                    self.all_probs[i] = 1.0 - self.all_probs[i]      

        output_metrics = calculate_metrics(self.all_probs, self.all_y_pred, self.all_y)
        self.log_dict(output_metrics, prog_bar=True, logger=True)
        
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.trainer_config.lr)
        scheduler = {
            "scheduler": StepLR(optimizer, step_size=self.trainer_config.optimizer_scheduler.step_size, gamma=self.trainer_config.optimizer_scheduler.gamma, verbose=True),
            "name": "learning_rate",
            "interval": "epoch",
        }

        return [optimizer], [scheduler]

    
class HyPepToxFuse_Hybrid_Trainer(L.LightningModule):
    def __init__(self, model_config, trainer_config):
        super().__init__()
        self.model = HyPepToxFuse_Hybrid(**vars(model_config))
        self.trainer_config = trainer_config
        self.model_config = model_config
        
        self.loss_fn_name = self.trainer_config.loss_fn
        
        if self.loss_fn_name == 'focal':
            self.loss_fn = focal_loss(alpha=self.trainer_config.alpha_focal, gamma=self.trainer_config.beta_focal, device=self.device)
        else:
            self.loss_fn = F.binary_cross_entropy
            
        self.ntxent_loss = nt_bxent_loss
    
    def forward(self, X, mask_X):
        return self.model(X[0], X[1], X[2], X[3], mask_X[0], mask_X[1], mask_X[2])
    
    def training_step(self, batch, batch_idx):
        _, X, mask_X, y, pos_indices = batch
        logits, embeddings = self.forward(X, mask_X)
        
        if self.loss_fn_name == 'focal':
            y_probs = F.softmax(logits, dim=-1)
            losses = self.loss_fn(y_probs, y)
        else:
            y_probs = F.sigmoid(logits)
            losses = self.loss_fn(y_probs.squeeze(1), y.float())
        
        ntxent_losses_weight = 0.2
        losses_weight = 0.8
        
        ntxent_losses = self.ntxent_loss(embeddings, pos_indices, temperature=self.trainer_config.ntxent_temperature)
        
        total_losses = losses_weight*losses + ntxent_losses_weight*ntxent_losses
        
        self.log('train_loss', total_losses.item(), prog_bar=True, logger=True)
        return total_losses
    
    def on_validation_epoch_start(self):
        self.all_y = []
        self.all_y_pred = []
        self.all_probs = []
        
    def validation_step(self, batch, batch_idx):
        _, X, mask_X, y, _ = batch
            
        logits, _ = self.forward(X, mask_X)
                
        if self.loss_fn_name == 'focal':
            y_probs = F.softmax(logits, dim=-1)
            y_pred = torch.argmax(y_probs, dim=-1).unsqueeze(1)
        else:
            y_probs = F.sigmoid(logits)
            y_pred = (y_probs >= self.trainer_config.threshold).float()
        
        self.all_y.append(y.cpu().numpy())
        self.all_y_pred.extend(y_pred.cpu().numpy())
        self.all_probs.extend(y_probs.cpu().numpy())
        
    def on_validation_epoch_end(self):
        self.all_y = np.concatenate(self.all_y)
        self.all_y_pred = np.concatenate(self.all_y_pred)
        
        if self.model_config.n_classes == 1:
            self.all_probs = np.concatenate(self.all_probs)
        else:
            self.all_probs = [probs.max() for probs in self.all_probs]
            self.all_probs = np.array(self.all_probs)
            
            # reverse value of probability of class 0
            for i in range(len(self.all_y_pred)):
                if self.all_y_pred[i] == 0:
                    self.all_probs[i] = 1.0 - self.all_probs[i]      

        output_metrics = calculate_metrics(self.all_probs, self.all_y_pred, self.all_y)
        self.log_dict(output_metrics, prog_bar=True, logger=True)
        
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.trainer_config.lr)
        scheduler = {
            "scheduler": StepLR(optimizer, step_size=self.trainer_config.optimizer_scheduler.step_size, gamma=self.trainer_config.optimizer_scheduler.gamma, verbose=True),
            "name": "learning_rate",
            "interval": "epoch",
        }

        return [optimizer], [scheduler]
