from argparse import Namespace
import torch

def set_nested_attr(obj, key, value):
    if isinstance(value, dict):
        if not hasattr(obj, key):
            setattr(obj, key, Namespace())
        
        for subkey in value:
            set_nested_attr(getattr(obj, key), subkey, value[subkey])
    else:
        setattr(obj, key, value)
        
def namespace_to_dict(namespace):
    result = {}
    for key, value in vars(namespace).items():
        if isinstance(value, Namespace):
            result[key] = namespace_to_dict(value)
        else:
            result[key] = value
    return result

def dataset_collate_fn(batch, added_ccd=False):
    keys, features, masks, labels = zip(*batch)
    
    if added_ccd:
        f1, f2, f3, hc_f = zip(*features)
        hc_f = torch.stack(hc_f)
    else:
        f1, f2, f3 = zip(*features)
    
    f1 = torch.stack(f1)
    f2 = torch.stack(f2)
    f3 = torch.stack(f3)
    
    
    mask_f1, mask_f2, mask_f3 = zip(*masks)
    mask_f1 = torch.stack(mask_f1)
    mask_f2 = torch.stack(mask_f2)
    mask_f3 = torch.stack(mask_f3)
    
    labels = torch.tensor(labels)
    
    # Identify positive indices
    pos_indices = construct_pos_indices(labels)
    
    tupled_features = (f1, f2, f3, hc_f) if added_ccd else (f1, f2, f3)
    
    return keys, tupled_features, (mask_f1, mask_f2, mask_f3), labels, pos_indices

def construct_pos_indices(labels):
    device = labels.device
    batch_size = labels.size(0)
    pos_indices = []

    # Create a matrix where each element (i, j) is True if labels[i] == labels[j]
    label_matrix = (labels.unsqueeze(0) == labels.unsqueeze(1)).to(device)

    # Get the indices of the upper triangle of the matrix (excluding the diagonal)
    upper_tri_indices = torch.triu_indices(batch_size, batch_size, offset=0, device=device)

    # Filter the indices where the labels are the same
    same_label_indices = upper_tri_indices[:, label_matrix[upper_tri_indices[0], upper_tri_indices[1]]]

    # Convert to list of tuples
    pos_indices = [(i.item(), j.item()) for i, j in zip(same_label_indices[0], same_label_indices[1])]

    return torch.tensor(pos_indices, device=device)

