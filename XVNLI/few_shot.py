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
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--grad_acc_steps', type=int, default=1)
    parser.add_argument('--agg_method', type=str, default='cat')
    parser.add_argument('--shot_num', type=int, default=1)
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--language', type=str, default='')
    parser.add_argument('--train_split', type=str, default='train')
    parser.add_argument('--classifier', type=str, default='simple')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # device = "cpu"
    print(device)
    clip_model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms('xlm-roberta-base-ViT-B-32', pretrained='laion5b_s13b_b90k', device=device)
    tokenizer = open_clip.get_tokenizer('xlm-roberta-base-ViT-B-32')
    args = parse_args()

    shot2batch = {1: 2, 5: 16, 10: 32, 20: 64, 25: 64, 48: 64}
    batch_size = shot2batch[args.shot_num]

    wandb.init(
        # set the wandb project where this run will be logged
        project="iglue_XVNLIXMLCLIP_fewshot",
        
        # track hyperparameters and run metadata
        config={
        "learning_rate": args.lr,
        "architecture": "xmlroberta",
        "dataset": "XVNLI",
        "epochs": 20
        },

        name=f"{args.language}_{args.shot_num}"
    )

    print('few-shot for', args.train_split, args.language)
    few_shot = args.language
    train_dset = XVNLIFeaturesDataset(args.train_split, preprocess_train, tokenizer, few_shot) # 测试结果
    eval_dset = XVNLIFeaturesDataset('dev', preprocess_val, tokenizer)
    test_dset = XVNLIFeaturesDataset(f'{few_shot}_test', preprocess_val, tokenizer)

    train_loader = DataLoader(train_dset, batch_size=batch_size, shuffle=True)
    eval_loader = DataLoader(eval_dset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dset, batch_size=512, shuffle=True)
    # import ipdb
    # ipdb.set_trace()
    # batch_size = args.batch_size
    if args.agg_method == 'cat':
        in_dim = 512 * 2
    else:
        in_dim = 512
    out_dim = 3

    model = XVNLIXMLCLIP(clip_model, in_dim, out_dim, args.agg_method, args.classifier).to(device)
    PATH = "/home/ndp689/iglue_xvnli/iglue_XVNLIXMLCLIP_SimpleClassifier.pt"
    model.load_state_dict(torch.load(PATH))

    train(model, train_loader, eval_loader, args.epochs, device, batch_size, args.grad_acc_steps, few_shot, test_loader, args.lr)