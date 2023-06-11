import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from train import compute_score_with_logits
from dataset import XVNLIFeaturesDataset
from classifier import XVNLIXMLCLIP
from train import train
import open_clip
import wandb

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=64) # 64
    parser.add_argument('--grad_acc_steps', type=int, default=2)
    parser.add_argument('--agg_method', type=str, default='cat')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # device = "cpu"
    print(device)
    clip_model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms('xlm-roberta-base-ViT-B-32', pretrained='laion5b_s13b_b90k', device=device)
    tokenizer = open_clip.get_tokenizer('xlm-roberta-base-ViT-B-32')
    args = parse_args()

    wandb.init(
        # set the wandb project where this run will be logged
        project="iglue_XVNLIXMLCLIP",
        
        # track hyperparameters and run metadata
        config={
        "learning_rate": 2e-5,
        "architecture": "xmlroberta",
        "dataset": "XVNLI",
        "epochs": 5,
        }
    )

    train_dset = XVNLIFeaturesDataset('train', preprocess_train, tokenizer) # 测试结果
    eval_dset = XVNLIFeaturesDataset('dev', preprocess_val, tokenizer)
    # import ipdb
    # ipdb.set_trace()
    # batch_size = args.batch_size
    if args.agg_method == 'cat':
        in_dim = 512 * 2
    else:
        in_dim = 512
    out_dim = 3
    model = XVNLIXMLCLIP(clip_model, in_dim, out_dim, args.agg_method).to(device)

    train_loader = DataLoader(train_dset, batch_size=args.batch_size, shuffle=True)
    eval_loader = DataLoader(eval_dset, batch_size=2048, shuffle=True)
    train(model, train_loader, eval_loader, args.epochs, device, args.batch_size, args.grad_acc_steps)