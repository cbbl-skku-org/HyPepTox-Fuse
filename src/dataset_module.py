import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
import glob, os

class NLP_Dataset(Dataset):
    def __init__(self, dataset_root, 
                 feature_1_name, 
                 feature_2_name, 
                 feature_3_name, 
                 is_training=True, 
                 mean_feature=True,
                 max_length=None):
        if is_training:
            self.pos_f1 = os.path.join(dataset_root, feature_1_name, 'train_pos')
            self.neg_f1 = os.path.join(dataset_root, feature_1_name, 'train_neg')
            self.pos_f2 = os.path.join(dataset_root, feature_2_name, 'train_pos')
            self.neg_f2 = os.path.join(dataset_root, feature_2_name, 'train_neg')
            self.pos_f3 = os.path.join(dataset_root, feature_3_name, 'train_pos')
            self.neg_f3 = os.path.join(dataset_root, feature_3_name, 'train_neg') 
        else:
            self.pos_f1 = os.path.join(dataset_root, feature_1_name, 'test_pos')
            self.neg_f1 = os.path.join(dataset_root, feature_1_name, 'test_neg')
            self.pos_f2 = os.path.join(dataset_root, feature_2_name, 'test_pos')
            self.neg_f2 = os.path.join(dataset_root, feature_2_name, 'test_neg')
            self.pos_f3 = os.path.join(dataset_root, feature_3_name, 'test_pos')
            self.neg_f3 = os.path.join(dataset_root, feature_3_name, 'test_neg') 
            
        self.mean_feature = mean_feature
        self.max_length = max_length
        
        self.set_up_keys_and_labels()
        self.preload_data()
    
    def __len__(self):
        return len(self.pep_keys)
    
    def set_up_keys_and_labels(self):
        self.pep_keys = [f'Negative_{i}' for i in range(len(os.listdir(self.neg_f1)))] \
                        + [f'Positive_{i}' for i in range(len(os.listdir(self.pos_f1)))]
        self.labels = []
        for key in self.pep_keys:
            if key.startswith('Negative'):
                self.labels.append(0)
            else:
                self.labels.append(1)
    
    def preload_data(self):
        self.data = {}
        embedding_type = 'mean_representations' if self.mean_feature else 'representations'
        
        for key in self.pep_keys:
            if key.startswith('Negative'):
                f1 = torch.load(os.path.join(self.neg_f1, key + ".pt"), weights_only=True)[embedding_type]
                f2 = torch.load(os.path.join(self.neg_f2, key + ".pt"), weights_only=True)[embedding_type]
                f3 = torch.load(os.path.join(self.neg_f3, key + ".pt"), weights_only=True)[embedding_type]
            else:
                f1 = torch.load(os.path.join(self.pos_f1, key + ".pt"), weights_only=True)[embedding_type]
                f2 = torch.load(os.path.join(self.pos_f2, key + ".pt"), weights_only=True)[embedding_type]
                f3 = torch.load(os.path.join(self.pos_f3, key + ".pt"), weights_only=True)[embedding_type]
            
            self.data[key] = (f1, f2, f3)
    
    def __getitem__(self, index):
        pep_key = self.pep_keys[index]
        f1, f2, f3 = self.data[pep_key]
        label = self.labels[index]
        
        len_f1 = f1.shape[0] if f1.ndim > 1 else 1
        len_f2 = f2.shape[0] if f2.ndim > 1 else 1
        len_f3 = f3.shape[0] if f3.ndim > 1 else 1
        
        mask_f1 = torch.zeros(len_f1, dtype=torch.bool)
        mask_f2 = torch.zeros(len_f2, dtype=torch.bool)
        mask_f3 = torch.zeros(len_f3, dtype=torch.bool)
        
        f1 = f1.detach()
        f2 = f2.detach()
        f3 = f3.detach()
        
        if not self.mean_feature:
            if self.max_length:
                f1 = F.pad(f1, (0, 0, 0, self.max_length - f1.size(0)), value=0)
                f2 = F.pad(f2, (0, 0, 0, self.max_length - f2.size(0)), value=0)
                f3 = F.pad(f3, (0, 0, 0, self.max_length - f3.size(0)), value=0)
                
                # Create attention masks
                mask_f1 = torch.ones(self.max_length, dtype=torch.bool)
                mask_f2 = torch.ones(self.max_length, dtype=torch.bool)
                mask_f3 = torch.ones(self.max_length, dtype=torch.bool)
                
                mask_f1[:len_f1] = False
                mask_f2[:len_f2] = False
                mask_f3[:len_f3] = False
                
        f1.requires_grad_(False)
        f2.requires_grad_(False)
        f3.requires_grad_(False)
        
        return pep_key, (f1, f2, f3), (mask_f1, mask_f2, mask_f3), label
    
class Hybrid_Dataset(Dataset):
    def __init__(self, dataset_root, 
                 feature_1_name, 
                 feature_2_name, 
                 feature_3_name,
                 handcraft_name,
                 top_handcraft_feature=100,
                 is_training=True, 
                 mean_feature=True,
                 max_length=None):
        if is_training:
            self.pos_f1 = os.path.join(dataset_root, feature_1_name, 'train_pos')
            self.neg_f1 = os.path.join(dataset_root, feature_1_name, 'train_neg')
            self.pos_f2 = os.path.join(dataset_root, feature_2_name, 'train_pos')
            self.neg_f2 = os.path.join(dataset_root, feature_2_name, 'train_neg')
            self.pos_f3 = os.path.join(dataset_root, feature_3_name, 'train_pos')
            self.neg_f3 = os.path.join(dataset_root, feature_3_name, 'train_neg')
            # Handcraft features
            self.neg_hc_f = os.path.join(dataset_root, handcraft_name, 'train_neg')
            self.pos_hc_f = os.path.join(dataset_root, handcraft_name, 'train_pos')
        else:
            self.pos_f1 = os.path.join(dataset_root, feature_1_name, 'test_pos')
            self.neg_f1 = os.path.join(dataset_root, feature_1_name, 'test_neg')
            self.pos_f2 = os.path.join(dataset_root, feature_2_name, 'test_pos')
            self.neg_f2 = os.path.join(dataset_root, feature_2_name, 'test_neg')
            self.pos_f3 = os.path.join(dataset_root, feature_3_name, 'test_pos')
            self.neg_f3 = os.path.join(dataset_root, feature_3_name, 'test_neg')
            # Handcraft features
            self.neg_hc_f = os.path.join(dataset_root, handcraft_name, 'test_neg')
            self.pos_hc_f = os.path.join(dataset_root, handcraft_name, 'test_pos')
            
        self.mean_feature = mean_feature
        self.max_length = max_length
        self.top_handcraft_feature = top_handcraft_feature
        
        self.set_up_keys_and_labels()
        self.preload_data()
    
    def __len__(self):
        return len(self.pep_keys)
    
    def set_up_keys_and_labels(self):
        self.pep_keys = [f'Negative_{i}' for i in range(len(os.listdir(self.neg_f1)))] \
                        + [f'Positive_{i}' for i in range(len(os.listdir(self.pos_f1)))]
        self.labels = []
        for key in self.pep_keys:
            if key.startswith('Negative'):
                self.labels.append(0)
            else:
                self.labels.append(1)
    
    def preload_data(self):
        self.data = {}
        embedding_type = 'mean_representations' if self.mean_feature else 'representations'
        
        for key in self.pep_keys:
            if key.startswith('Negative'):
                f1 = torch.load(os.path.join(self.neg_f1, key + ".pt"), weights_only=True)[embedding_type]
                f2 = torch.load(os.path.join(self.neg_f2, key + ".pt"), weights_only=True)[embedding_type]
                f3 = torch.load(os.path.join(self.neg_f3, key + ".pt"), weights_only=True)[embedding_type]
                hc_f = torch.load(os.path.join(self.neg_hc_f, key + ".pt"), weights_only=True)['mean_representations'][:self.top_handcraft_feature]
            else:
                f1 = torch.load(os.path.join(self.pos_f1, key + ".pt"), weights_only=True)[embedding_type]
                f2 = torch.load(os.path.join(self.pos_f2, key + ".pt"), weights_only=True)[embedding_type]
                f3 = torch.load(os.path.join(self.pos_f3, key + ".pt"), weights_only=True)[embedding_type]
                hc_f = torch.load(os.path.join(self.pos_hc_f, key + ".pt"), weights_only=True)['mean_representations'][:self.top_handcraft_feature]
            
            self.data[key] = (f1, f2, f3, hc_f)
    
    def __getitem__(self, index):
        pep_key = self.pep_keys[index]
        f1, f2, f3, hc_f = self.data[pep_key]
        label = self.labels[index]
        
        len_f1 = f1.shape[0] if f1.ndim > 1 else 1
        len_f2 = f2.shape[0] if f2.ndim > 1 else 1
        len_f3 = f3.shape[0] if f3.ndim > 1 else 1
        
        mask_f1 = torch.zeros(len_f1, dtype=torch.bool)
        mask_f2 = torch.zeros(len_f2, dtype=torch.bool)
        mask_f3 = torch.zeros(len_f3, dtype=torch.bool)
        
        f1 = f1.detach()
        f2 = f2.detach()
        f3 = f3.detach()
        hc_f = hc_f.detach()
        
        if not self.mean_feature:
            if self.max_length:
                f1 = F.pad(f1, (0, 0, 0, self.max_length - f1.size(0)), value=0)
                f2 = F.pad(f2, (0, 0, 0, self.max_length - f2.size(0)), value=0)
                f3 = F.pad(f3, (0, 0, 0, self.max_length - f3.size(0)), value=0)
                
                # Create attention masks
                mask_f1 = torch.ones(self.max_length, dtype=torch.bool)
                mask_f2 = torch.ones(self.max_length, dtype=torch.bool)
                mask_f3 = torch.ones(self.max_length, dtype=torch.bool)
                
                mask_f1[:len_f1] = False
                mask_f2[:len_f2] = False
                mask_f3[:len_f3] = False
                
        f1.requires_grad_(False)
        f2.requires_grad_(False)
        f3.requires_grad_(False)
        hc_f.requires_grad_(False)
        
        return pep_key, (f1, f2, f3, hc_f), (mask_f1, mask_f2, mask_f3), label