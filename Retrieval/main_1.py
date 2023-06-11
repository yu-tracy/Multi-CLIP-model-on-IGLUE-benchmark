import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
# from train import compute_score_with_logits
from dataset import RetrievalDataset
from classifier import RetrievalXMLCLIP
from train import train
import open_clip
import wandb

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=64) # 64
    parser.add_argument('--grad_acc_steps', type=int, default=1)
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
        project="iglue_RetrievalXMLCLIP",
        
        # track hyperparameters and run metadata
        config={
        "learning_rate": 2e-5,
        "architecture": "xmlroberta",
        "dataset": "flickr30k",
        "epochs": 10,
        },

        name="re_train"
    )
    
    # task_name = flickr30k, COCO, xFlickrCO
    train_dset =  RetrievalDataset('flickr30k', 'train_ann', 'train', tokenizer, preprocess_train)
    eval_dset = RetrievalDataset('flickr30k', 'valid_ann', 'val', tokenizer, preprocess_val)
    train_loader = DataLoader(train_dset, batch_size=args.batch_size, shuffle=True)
    eval_loader = DataLoader(eval_dset, batch_size=512)

    in_dim = 512 * 2
    # import ipdb
    # ipdb.set_trace()
    out_dim = 1
    model = RetrievalXMLCLIP(clip_model, in_dim, out_dim).to(device)
    PATH = "/home/ndp689/iglue_retrieval/iglue_RetrievalXMLCLIP.pt"
    model.load_state_dict(torch.load(PATH))

    train(model, train_loader, eval_loader, args.epochs, device, args.batch_size, args.grad_acc_steps)