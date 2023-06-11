import os
import time
import torch
import torch.nn as nn
from torch.autograd import Variable
import wandb
from pytorch_transformers.optimization import AdamW, WarmupLinearSchedule
from torch.utils.data import DataLoader
from dataset import XVNLIFeaturesDataset


def instance_bce_with_logits(logits, labels, criterion):
    assert logits.dim() == 2

    loss = criterion(logits, labels)
    loss *= labels.size(1)
    return loss

def compute_score_with_logits(logits, labels):
    logits = torch.max(logits, 1)[1].data # argmax
    one_hots = torch.zeros(*labels.size()).cuda()
    one_hots.scatter_(1, logits.view(-1, 1), 1)
    scores = (one_hots * labels)
    return scores

def train(model, train_loader, eval_loader, num_epochs, device, batch_size, grad_acc_steps = 1, few_shot='', test_loader=None, lr=2e-5, eval_batch_size=32):
    criterion = nn.BCEWithLogitsLoss(reduction="mean")
    optimizer = AdamW(model.parameters(), lr=lr, eps=1e-6, weight_decay= 10e-4, correct_bias=False) 

    train_size = len(train_loader.dataset)
    num_train_optim_steps = (train_size * num_epochs) // (batch_size * grad_acc_steps)
    warmup_steps = 0.1 * num_train_optim_steps
    scheduler = WarmupLinearSchedule(optimizer, warmup_steps=warmup_steps, t_total=num_train_optim_steps)

    iter_step = 1
    model.train()
    wandb.watch(model, criterion, log_freq=100)
    for epoch in range(num_epochs):
        train_loss = 0
        train_score = 0
        print(f'epoch - {epoch}')

        for i, (text, image, target) in enumerate(train_loader):
            # import ipdb
            # ipdb.set_trace()
            text, target, image = text.to(device), target.to(device), image.to(device)
            pred = model(image, text)

            loss = instance_bce_with_logits(pred, target, criterion)
            batch_score = compute_score_with_logits(pred, target).sum() / float(text.shape[0]) # average
            train_score += batch_score
            if grad_acc_steps > 1:
                loss = loss / grad_acc_steps
            train_loss += loss.item()

            if iter_step % 1000 == 0:
                wandb.log({"train_batch_loss": loss.item(), "train_batch_acc": batch_score}, step=iter_step)
            loss.backward()

            if (i + 1) % grad_acc_steps == 0:
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                model.zero_grad()
            wandb.log({'learning rate': scheduler.get_last_lr()[0]}, step=iter_step)
            # if iter_step % 20 == 0:
            #     batch_score = compute_score_with_logits(pred, target).sum()
            #     # total_loss /= len(train_loader.dataset)
            #     # train_score = train_score / len(train_loader.dataset)
            #     wandb.log({"train_batch_loss": loss.data.item(), "train_batch_acc": batch_score}, step=iter_step)
            iter_step += 1

        train_loss /= len(train_loader)
        train_score /= len(train_loader)
        model.eval()
        eval_score, eval_loss = evaluate(model, eval_loader, criterion, device, eval_batch_size)
        wandb.log({"train_norm_loss": train_loss, "train_norm_acc": train_score, "eval_norm_acc": eval_score, "eval_norm_loss": eval_loss}, step=iter_step-1)
        test_name = f'{few_shot}_test'
        print(test_name)
        score, _ = evaluate(model, test_loader, criterion, device)
        print(f'{test_name} score:', score)
        model.train()

    model.eval()
    # if few_shot and test_loader:
        
    # else:
    #     PATH = f"/home/ndp689/iglue_xvnli/iglue_XVNLIXMLCLIP_{few_shot}.pt"
    #     # save model
    #     torch.save(model.state_dict(), PATH)
    wandb.finish()

def evaluate(model, val_loader, criterion, device, batch_size = 32):
    eval_score = 0
    eval_loss = 0
    with torch.no_grad():
        for text, image, target in iter(val_loader):
            text, target, image = text.to(device), target.to(device), image.to(device)
            pred = model(image, text)
            batch_loss = instance_bce_with_logits(pred, target, criterion)
            batch_score = compute_score_with_logits(pred, target).sum() / float(text.shape[0])
            eval_score += batch_score
            eval_loss += batch_loss.item()

    loss = eval_loss / len(val_loader)
    score = eval_score / len(val_loader)
    return score, loss
