from models_and_dataset import ReasoningXMLCLIP, get_data
import torch
import open_clip
import json
import torch.nn as nn
from pytorch_transformers.optimization import AdamW, WarmupLinearSchedule
import wandb

device = "cuda" if torch.cuda.is_available() else "cpu"
device = "cuda"
print(device)
clip_model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms('xlm-roberta-base-ViT-B-32', pretrained='laion5b_s13b_b90k', device=device)
tokenizer = open_clip.get_tokenizer('xlm-roberta-base-ViT-B-32')

def accuracy(out, yb):
    preds = torch.argmax(out, dim=1)
    yb = torch.argmax(yb, dim=1)
    return (preds == yb).float().mean()

def test_reasoning(model, criterion, device, devOrTest_loader, is_test=False, lang='en'):
    model.eval()
    score = 0
    num_batch = 0
    loss_total = 0
    with torch.no_grad():
        for left_image, right_image, text, target in devOrTest_loader:
            images = torch.cat((left_image, right_image), dim=0).to(device)
            text, target = text.to(device), target.to(device)
            output = model(images, text)
            loss = criterion(output, target)
            acc = accuracy(output,target)
            score += acc
            loss_total += loss
            num_batch += 1
            # out = torch.squeeze(out, dim=1)

    score = score / num_batch
    loss_total = loss_total / num_batch
    model.train()
    if is_test:
        print(f"{lang} - accuracy: {score}, loss: {loss_total}")
        return 
    else:
	    return score, loss_total

def test_reasoning_batch(model, device, left_image, right_image, text, target, is_test=False, lang='en'):
    model.eval()
    score = 0.
    num_batch = 0
    with torch.no_grad():
        images = torch.cat((left_image, right_image), dim=0).to(device)
        output = model(images, text)
        acc = accuracy(output,target)
        score += acc

    score = score / num_batch
    model.train()
    if is_test:
        print(f"accuracy - {lang}: {score}")
    else:
	    return score


wandb.init(
    # set the wandb project where this run will be logged
    project="iglue_ReasoningXMLCLIP_LN",
    
    # track hyperparameters and run metadata
    config={
    "learning_rate": 1e-5,
    "architecture": "xmlroberta",
    "dataset": "nlvr2",
    "epochs": 10,
    }
)

dataset = get_data((preprocess_train, preprocess_val), tokenizer)

print("train start")

BATCH_SIZE = 64
NUM_EPOCH = 10
lr = 1e-5
in_dim = 512 * 3
hid_dim = 256
out_dim = 2

model = ReasoningXMLCLIP(clip_model, in_dim, hid_dim, out_dim).to(device)
criterion = nn.BCEWithLogitsLoss(reduction="mean")
optimizer = AdamW(model.parameters(), lr=lr, eps=1e-6, weight_decay=1e-4, correct_bias=False) 

# train_path = '/Users/wy/Documents/KU/Master_thesis/thesis_project/iglue-main/datasets/nlvr2/annotations/train.jsonl'
train_path = '/home/ndp689/iglue_rsn/train.jsonl'
train_json = [json.loads(line) for line in open(train_path).readlines()]
num_train_optim_steps = (len(train_json) * NUM_EPOCH) // BATCH_SIZE
warmup_steps = 0.1 * num_train_optim_steps
scheduler = WarmupLinearSchedule(optimizer, warmup_steps=warmup_steps, t_total=num_train_optim_steps)

model.train()
iter_step = 1
for i in range(1, NUM_EPOCH + 1):
    for left_image, right_image, text, target in dataset['train']:
        images = torch.cat((left_image, right_image), dim=0).to(device)
        text, target = text.to(device), target.to(device)

        output = model(images, text)
        loss = criterion(output, target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if iter_step % 10 == 0:
            batch_acc = accuracy(output,target)
            eval_acc, eval_loss = test_reasoning(model,criterion, device, dataset['eval'], is_test=False, lang='en')
            # last_lr = scheduler.get_last_lr()
            wandb.log({"train_batch_loss": loss, "train_batch_acc": batch_acc, "eval_norm_acc": eval_acc, "eval_norm_loss": eval_loss}, step=iter_step)
        iter_step += 1
        scheduler.step()
        model.zero_grad()

PATH = "/home/ndp689/iglue_rsn/iglue_ReasoningXMLCLIP_LN.pt"
# save model
torch.save(model.state_dict(), PATH)
wandb.finish()

test_reasoning(model, criterion, device, dataset['test-en'], is_test=True, lang='en')
test_reasoning(model, criterion, device, dataset['test-zh'], is_test=True, lang='zh')
