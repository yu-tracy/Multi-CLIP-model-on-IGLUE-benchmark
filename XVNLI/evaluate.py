import torch
from torch.utils.data import DataLoader
from classifier import XVNLIXMLCLIP
from dataset import XVNLIFeaturesDataset
import open_clip
from train import compute_score_with_logits

def evaluate(model, val_loader, device, batch_size = 2048):
    eval_score = 0
    with torch.no_grad():
        for text, image, target in iter(val_loader):
            text, target, image = text.to(device), target.to(device), image.to(device)
            pred = model(image, text)
            batch_score = compute_score_with_logits(pred, target).sum() / float(text.shape[0])
            eval_score += batch_score

    score = eval_score / len(val_loader)
    return score

device = "cuda" if torch.cuda.is_available() else "cpu"
# device = "cpu"
print(device)
clip_model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms('xlm-roberta-base-ViT-B-32', pretrained='laion5b_s13b_b90k', device=device)
tokenizer = open_clip.get_tokenizer('xlm-roberta-base-ViT-B-32')

agg_method = 'cat'
if agg_method == 'cat':
    in_dim = 512 * 2
else:
    in_dim = 512
out_dim = 3

model = XVNLIXMLCLIP(clip_model, in_dim, out_dim, agg_method).to(device)
PATH = "/home/ndp689/iglue_xvnli/iglue_XVNLIXMLCLIP.pt"
model.load_state_dict(torch.load(PATH))
model.eval()

test_names = ['test', 'ar_test', 'es_test', 'fr_test', 'ru_test']
for name in test_names:
    print(name)
    test_dset = XVNLIFeaturesDataset(name, preprocess_val, tokenizer)
    test_loader = DataLoader(test_dset, batch_size=2048, shuffle=True)
    score = evaluate(model, test_loader, device)
    print(f'{name} score:', score)