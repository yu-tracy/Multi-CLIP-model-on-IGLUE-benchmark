import torch
import json
from PIL import ImageFile
from PIL import Image
import torch.nn as nn
import math
from torch.nn import functional as F
# import torch.optim as optim
import torch.utils.data as Data
# from pytorch_transformers.optimization import AdamW, WarmupLinearSchedule
ImageFile.LOAD_TRUNCATED_IMAGES = True

# ==================================================================================================================== #
#                                               helper func and class                                                  #
# ==================================================================================================================== #
def gelu(x):
    """Implementation of the gelu activation function.
        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
        Also see https://arxiv.org/abs/1606.08415
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

class GeLU(nn.Module):
    """Implementation of the gelu activation function.
        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
        Also see https://arxiv.org/abs/1606.08415
    """

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return gelu(x)

try:
    from apex.normalization.fused_layer_norm import FusedLayerNorm as BertLayerNorm
except ImportError:

    class BertLayerNorm(nn.Module):
        def __init__(self, hidden_size, eps=1e-12):
            """Construct a layernorm module in the TF style (epsilon inside the square root).
            """
            super(BertLayerNorm, self).__init__()
            self.weight = nn.Parameter(torch.ones(hidden_size))
            self.bias = nn.Parameter(torch.zeros(hidden_size))
            self.variance_epsilon = eps

        def forward(self, x):
            u = x.mean(-1, keepdim=True)
            s = (x - u).pow(2).mean(-1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.variance_epsilon)
            return self.weight * x + self.bias
        
# ==================================================================================================================== #
#                                                       models                                                         #
# ==================================================================================================================== #
    
class SimpleClassifier(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, layer_norm_eps=1e-12, dropout_prob=0.0):
        super().__init__()

        self.logit_fc = nn.Sequential(
            nn.Linear(in_dim, hid_dim),
            GeLU(),
            BertLayerNorm(hid_dim, eps=layer_norm_eps),
            nn.Linear(hid_dim, out_dim),
        )

        self.apply(self.init_weights)

    def init_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, BertLayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, hidden_states):
        return self.logit_fc(hidden_states)

class ReasoningXMLCLIP(nn.Module):
    def __init__(self, clipModel, in_dim, hid_dim, out_dim=2):
        super(ReasoningXMLCLIP, self).__init__()
        embedding_dim = 512
        self.clipModel = clipModel
        self.classifier = SimpleClassifier(in_dim, hid_dim, out_dim)
        self.layer_norm = nn.LayerNorm(embedding_dim)

    def forward(self, images, text):
        '''
            images: preprocessed images
            text: tokenized text
        '''
        image_features, text_features, _ = self.clipModel(images, text) 
        left_img_features, right_img_features = torch.split(image_features, image_features.shape[0]//2)
        # import ipdb
        # ipdb.set_trace()
        text_features, left_img_features, right_img_features = self.layer_norm(text_features), self.layer_norm(left_img_features), self.layer_norm(right_img_features)
        input_features = torch.cat((text_features, left_img_features, right_img_features), dim=1)
        output = self.classifier(input_features)
        return output

class VQAXMLCLIP(nn.Module):
    def __init__(self, clipModel, in_dim, hid_dim, out_dim=2):
        super(VQAXMLCLIP, self).__init__()
        embedding_dim = 512
        self.clipModel = clipModel
        self.classifier = SimpleClassifier(in_dim, hid_dim, out_dim)
        self.layer_norm = nn.LayerNorm(embedding_dim)

    def forward(self, image, text):
        '''
            image: preprocessed image
            text: tokenized text
        '''
        image_feature, text_features, _ = self.clipModel(image, text) 
        # import ipdb
        # ipdb.set_trace()
        text_features, image_feature = self.layer_norm(text_features), self.layer_norm(image_feature)
        input_features = torch.cat((text_features, image_feature), dim=1)
        output = self.classifier(input_features)
        return output

class XVNLIXMLCLIP(nn.Module):
    def __init__(self, clipModel, in_dim, out_dim=3, cat_method='mul'):
        super(XVNLIXMLCLIP, self).__init__()
        self.clipModel = clipModel
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(in_dim, out_dim)
        self.cat_method = cat_method

    def forward(self, image, text):
        '''
            image: preprocessed image
            text: tokenized text
        '''
        image_feature, text_features, _ = self.clipModel(image, text) 
        # import ipdb
        # ipdb.set_trace()
        if self.cat_method == 'mul':
            input_features = self.dropout(text_features * image_feature)
        else:
            input_features = self.dropout(torch.cat((text_features, image_feature), dim=1))
        input_features = self.dropout(input_features)
        output = self.classifier(input_features)
        return output

# ==================================================================================================================== #
#                                                       dataset                                                        #
# ==================================================================================================================== #

# ======================================= Reasoning ================================================== #

class GetENDataset(Data.Dataset):
    def __init__(self, data_json, image_dic, transforms, tokenizer, is_train):
        super(GetENDataset, self).__init__()
        self.data_json = data_json 
        self.length = len(self.data_json)
        self.image_dic = image_dic
        self.is_train = is_train
        self.transforms = transforms
        self.tokenizer = tokenizer
    
    def __getitem__(self, index):
        item = self.data_json[index]
        label = [0., 1.] if item["label"] == 'True' else [1., 0.]

        # get images' name
        split_id = item["identifier"].split("-")
        image_id = "-".join(split_id[:3])
        left_image_name = image_id + "-img0.png"
        right_image_name = image_id + "-img1.png"
        if self.is_train:
            directory = item["directory"]
            # get images
            left_image = self.transforms(Image.open(f"{self.image_dic}{directory}/{left_image_name}"))
            right_image = self.transforms(Image.open(f"{self.image_dic}{directory}/{right_image_name}"))
        else:
            left_image = self.transforms(Image.open(f"{self.image_dic}{left_image_name}"))
            right_image = self.transforms(Image.open(f"{self.image_dic}{right_image_name}"))
        # get text
        text = self.tokenizer(item['sentence'])

        return left_image, right_image, torch.squeeze(text, dim=0), torch.FloatTensor(label)

    def __len__(self):
        return self.length

# other language
class GetOtherLangSet(Data.Dataset):
    def __init__(self, data_json, image_dic, transforms, tokenizer):
        super(GetOtherLangSet, self).__init__()
        self.data_json = data_json 
        self.length = len(self.data_json)
        self.image_dic = image_dic
        self.transforms = transforms
        self.tokenizer = tokenizer

    def __getitem__(self, index):
        item = self.data_json[index]
        
        left_image_name = item["left_img"]
        right_image_name = item["right_img"]
        concept = item["concept"]
        left_image = self.transforms(Image.open(f"{self.image_dic}{concept}/{left_image_name}"))
        right_image = self.transforms(Image.open(f"{self.image_dic}{concept}/{right_image_name}"))
        label = [0., 1.] if item['label'] else [1., 0.]
        text = self.tokenizer(item['caption'])

        return left_image, right_image, torch.squeeze(text, dim=0), torch.FloatTensor(label)

    def __len__(self):
        return self.length
    
def get_en_dataset(input_filename, image_dic, preprocess_fn, is_train, batch_size, tokenizer=None):
    json_data = [json.loads(line) for line in open(input_filename).readlines()]
    dataset = GetENDataset(json_data, image_dic, preprocess_fn, tokenizer, is_train)

    dataloader = Data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=is_train
    )
    return dataloader

def get_other_dataset(input_filename, image_dic, preprocess_fn, batch_size, tokenizer=None):
    json_data = [json.loads(line) for line in open(input_filename).readlines()]
    dataset = GetOtherLangSet(json_data, image_dic, preprocess_fn, tokenizer)
    dataloader = Data.DataLoader(
        dataset,
        batch_size=batch_size
    )
    return dataloader


def get_data(preprocess_fns, tokenizer=None):
    preprocess_train, preprocess_val = preprocess_fns
    data = {}
    # get train dataset
    data['train'] = get_en_dataset(
                        input_filename = "/home/ndp689/iglue_rsn/train.jsonl", 
                        image_dic = "/home/ndp689/iglue_rsn/images/train/", 
                        preprocess_fn = preprocess_train, 
                        is_train = True, 
                        batch_size = 64, 
                        tokenizer=tokenizer)

    # get eval dataset
    data['eval'] = get_en_dataset(
                        input_filename = "/home/ndp689/iglue_rsn/dev.jsonl", 
                        image_dic = "/home/ndp689/iglue_rsn/images/dev/", 
                        preprocess_fn = preprocess_val, 
                        is_train = False, 
                        batch_size = 512, 
                        tokenizer=tokenizer)

    # get test-en dataset
    data['test-en'] = get_en_dataset(
                        input_filename = "/home/ndp689/iglue_rsn/test.jsonl", 
                        image_dic = "/home/ndp689/iglue_rsn/images/test/", 
                        preprocess_fn = preprocess_val, 
                        is_train = False, 
                        batch_size = 512, 
                        tokenizer=tokenizer)

    # get test-zh
    data['test-zh'] = get_other_dataset(
                        input_filename = "/home/ndp689/iglue_rsn/marvl-zh.jsonl", 
                        image_dic = "/home/ndp689/iglue_rsn/images/zh/images/", 
                        preprocess_fn =preprocess_val, 
                        batch_size=512, 
                        tokenizer=tokenizer)
    return data