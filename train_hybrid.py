import os
import copy
import yaml
import torch
from argparse import ArgumentParser
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader

from lightning import Trainer, seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint

from src.utils import set_nested_attr, dataset_collate_fn
from src.lightning_module import HyPepToxFuse_Hybrid_Trainer
from src.dataset_module import Hybrid_Dataset
from src.metrics import calculate_metrics

from functools import partial

def model_checkpoint_callback(output_path, filename):
    return ModelCheckpoint(
        dirpath=output_path,
        filename=filename,
        monitor='mcc',
        mode='max',
        verbose=True,
        save_top_k=1
    )

def test_kfold_models(model, ckpt_dir, test_dataloader, device):
    MODELS_LIST = []
    for ckpt_file in os.listdir(ckpt_dir):
        if not ckpt_file.endswith('.ckpt'):
            continue
        
        pt_file_path = os.path.join(ckpt_dir, ckpt_file)
        state_dict = torch.load(pt_file_path)['state_dict']
        model.load_state_dict(state_dict)
        model.eval()
        model.to(device)
        
        MODELS_LIST.append(copy.deepcopy(model))
    
    
    ALL_Y = []
    ALL_Y_PRED = []
    ALL_PROBS = []
    ALL_Y_PRED_VOTING = []
    for batch in test_dataloader:
        total_prob_batch = []
        _, X, mask_X, y, _ = batch
        X = [x.to(device) for x in X]
        mask_X = [mask_x.to(device) for mask_x in mask_X]
        y = y.to(device)
        
        for model in MODELS_LIST:
            with torch.no_grad():
                logits, _ = model(X, mask_X)
                
            total_prob_batch.append(logits.softmax(dim=-1))
        
        avg_prob_batch = torch.sum(torch.stack(total_prob_batch), dim=0) / len(MODELS_LIST)
        
        y_pred = torch.argmax(avg_prob_batch, dim=-1)
        
        y_pred_voting = torch.mode(torch.argmax(torch.stack(total_prob_batch), dim=-1), dim=0).values
        
        max_probs = avg_prob_batch.max(dim=-1)
        max_probs = torch.abs((max_probs.indices + 1) % 2 - max_probs.values) # Reverse probability when class is 0
        
        ALL_PROBS.append(max_probs)
        ALL_Y_PRED.append(y_pred)
        ALL_Y_PRED_VOTING.append(y_pred_voting)
        ALL_Y.append(y)
    
    ALL_Y_PRED = torch.concat(ALL_Y_PRED)
    ALL_Y_PRED_VOTING = torch.concat(ALL_Y_PRED_VOTING)
    ALL_Y = torch.concat(ALL_Y)
    ALL_PROBS = torch.concat(ALL_PROBS)
    
    avgprob_output_metrics = calculate_metrics(
        ALL_PROBS.detach().cpu().numpy(),
        ALL_Y_PRED.detach().cpu().numpy(),
        ALL_Y.detach().cpu().numpy()
    )
    
    voting_output_metrics = calculate_metrics(
        ALL_PROBS.detach().cpu().numpy(),
        ALL_Y_PRED_VOTING.detach().cpu().numpy(),
        ALL_Y.detach().cpu().numpy()
    )
    
    with open(os.path.join(ckpt_dir, 'test_avg_result.txt'), 'w') as f:
        for metric_name, value in avgprob_output_metrics.items():
            f.write(f'{metric_name}: {value}\n')
            
    with open(os.path.join(ckpt_dir, 'test_voting_result.txt'), 'w') as f:
        for metric_name, value in voting_output_metrics.items():
            f.write(f'{metric_name}: {value}\n')
            
def evaluate_model(model, ckpt_dir, fold, dataloader, device):
    ckpt_files = [f for f in os.listdir(ckpt_dir) if f'fold-{fold}' in f and f.endswith('.ckpt')]
    if not ckpt_files:
        raise FileNotFoundError(f"No checkpoint file found for fold {fold} in directory {ckpt_dir}")
    
    pt_file_path = os.path.join(ckpt_dir, ckpt_files[0])
    state_dict = torch.load(pt_file_path)['state_dict']
    model.load_state_dict(state_dict)
    model.eval()
    model.to(device)
    
    ALL_Y = []
    ALL_Y_PRED = []
    ALL_PROBS = []
    for batch in dataloader:
        _, X, mask_X, y, _ = batch
        X = [x.to(device) for x in X]
        mask_X = [mask_x.to(device) for mask_x in mask_X]
        y = y.to(device)
        
        with torch.no_grad():
            logits, _ = model(X, mask_X)
        
        y_pred = torch.argmax(logits, dim=-1)
        max_probs = logits.softmax(dim=-1).max(dim=-1)
        max_probs = torch.abs((max_probs.indices + 1) % 2 - max_probs.values) # Reverse probability when class is 0
        
        ALL_PROBS.append(max_probs)
        ALL_Y_PRED.append(y_pred)
        ALL_Y.append(y)
    
    ALL_Y_PRED = torch.concat(ALL_Y_PRED)
    ALL_Y = torch.concat(ALL_Y)
    ALL_PROBS = torch.concat(ALL_PROBS)
    
    output_metrics = calculate_metrics(
        ALL_PROBS.detach().cpu().numpy(),
        ALL_Y_PRED.detach().cpu().numpy(),
        ALL_Y.detach().cpu().numpy()
    )
    
    with open(os.path.join(ckpt_dir, f'fold-{fold}-result.txt'), 'w') as f:
        for metric_name, value in output_metrics.items():
            f.write(f'{metric_name}: {value}\n')


def main(args):
    device = torch.device('cuda' if args.cuda else 'cpu')
    
    # Initialize dataset
    print('Initialize training dataset...')
    train_dataset = Hybrid_Dataset(
        args.dataset.dataset_root,
        args.dataset.feature_1_name,
        args.dataset.feature_2_name,
        args.dataset.feature_3_name,
        args.dataset.handcraft_name,
        args.dataset.top_handcraft_feature,
        mean_feature=args.dataset.mean_feature,
        max_length=args.dataset.max_length,
        is_training=True
    )
    
    print('Initialize independent dataset...')
    test_dataset = Hybrid_Dataset(
        args.dataset.dataset_root,
        args.dataset.feature_1_name,
        args.dataset.feature_2_name,
        args.dataset.feature_3_name,
        args.dataset.handcraft_name,
        args.dataset.top_handcraft_feature,
        mean_feature=args.dataset.mean_feature,
        max_length=args.dataset.max_length,
        is_training=False
    )
    
    # Initialize test dataloader
    test_dataloader = DataLoader(
        test_dataset, 
        batch_size=args.trainer_config.batch_size,
        shuffle=False,
        num_workers=args.trainer_config.num_workers,
        pin_memory=True,
        collate_fn=partial(dataset_collate_fn, added_ccd=True)
    )
    
    # Define kfold
    kfold = StratifiedKFold(
        n_splits=args.trainer_config.k_fold,
        shuffle=True,
        random_state=args.seed
    )
    
    for fold, (train_indices, val_indices) in enumerate(kfold.split(train_dataset.pep_keys, train_dataset.labels)):
        # Split dataset
        train_subset = torch.utils.data.Subset(train_dataset, train_indices)
        val_subset = torch.utils.data.Subset(train_dataset, val_indices)
        
        # Initialize train and val dataloader
        train_dataloader = DataLoader(
            train_subset,
            batch_size=args.trainer_config.batch_size,
            shuffle=True,
            num_workers=args.trainer_config.num_workers,
            pin_memory=True,
            collate_fn=partial(dataset_collate_fn, added_ccd=True)
        )
        val_dataloader = DataLoader(
            val_subset,
            batch_size=args.trainer_config.batch_size,
            shuffle=False,
            num_workers=args.trainer_config.num_workers,
            pin_memory=True,
            collate_fn=partial(dataset_collate_fn, added_ccd=True)
        )
        
        # Initialize model and trainer
        model = HyPepToxFuse_Hybrid_Trainer(args.model_config, args.trainer_config)
        model.to(device)
        
        checkpoint_callback = model_checkpoint_callback(args.trainer_config.output_path, f'fold-{fold}-' + 'ckpt-{epoch}-{mcc:.4f}')
        
        trainer = Trainer(
            max_epochs=args.trainer_config.epochs,
            check_val_every_n_epoch=1,
            gradient_clip_val=10.,
            callbacks=[checkpoint_callback],
            accelerator='gpu' if args.cuda else 'cpu',
            accumulate_grad_batches=args.trainer_config.grad_accum,
            deterministic=True
        )
        trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
        
        # Evaluate fold model
        evaluate_model(model, args.trainer_config.output_path, fold, val_dataloader, device)

    # Average all evaluation folds from .txt file
    avg_eval_metrics = {}
    for fold in range(args.trainer_config.k_fold):
        with open(os.path.join(args.trainer_config.output_path, f'fold-{fold}-result.txt'), 'r') as f:
            for line in f:
                metric_name, value = line.strip().split(': ')
                if metric_name not in avg_eval_metrics:
                    avg_eval_metrics[metric_name] = []
                
                avg_eval_metrics[metric_name].append(float(value))
    
    # Write avg_eval_metrics to a file
    with open(os.path.join(args.trainer_config.output_path, 'avg_eval_result.txt'), 'w') as f:
        for metric_name, values in avg_eval_metrics.items():
            avg_value = sum(values) / len(values)
            f.write(f'{metric_name}: {avg_value}\n')
    
    # Run test kfold models
    test_kfold_models(model, ckpt_dir=args.trainer_config.output_path, test_dataloader=test_dataloader, device=device)
    

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/hypeptox_fuse_hybrid.yml')
    parser.add_argument('--cuda', action='store_true')
    
    args = parser.parse_args()
    config = yaml.safe_load(open(args.config, 'r'))
    for key, value in config.items():
        set_nested_attr(args, key, value)
    
    # Set seed for deterministic
    seed_everything(args.seed)
    
    main(args)