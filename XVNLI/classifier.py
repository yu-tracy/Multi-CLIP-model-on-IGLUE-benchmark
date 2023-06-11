import torch.nn as nn
from torch.nn.utils.weight_norm import weight_norm
import torch
import math

# class SimpleClassifier(nn.Module):
#     def __init__(self, in_dim, hid_dim, out_dim, dropout):
#         super(SimpleClassifier, self).__init__()
#         layers = [
#             weight_norm(nn.Linear(in_dim, hid_dim), dim=None),
#             nn.ReLU(),
#             nn.Dropout(dropout, inplace=True),
#             weight_norm(nn.Linear(hid_dim, out_dim), dim=None)
#         ]
#         self.main = nn.Sequential(*layers)

#     def forward(self, x):
#         logits = self.main(x)
#         return logits

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

class SimpleClassifier(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, layer_norm_eps=1e-12):
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

class XVNLIXMLCLIP(nn.Module):
    def __init__(self, clipModel, in_dim, out_dim, agg_method, classifier= 'linear', hid_dim=256, dropout_prob = 0.1):
        super(XVNLIXMLCLIP, self).__init__()
        # embedding_dim = 512
        self.clipModel = clipModel
        if classifier == 'linear':
            self.classifier = nn.Linear(in_dim, out_dim)
        else:
            self.classifier = SimpleClassifier(in_dim, hid_dim, out_dim)
        self.dropout = nn.Dropout(dropout_prob)
        self.agg_method = agg_method

    def forward(self, image, text):
        '''
            image: preprocessed image
            text: tokenized text
        '''
        image_feature, text_features, _ = self.clipModel(image, text) 
        # import ipdb
        # ipdb.set_trace()
        if self.agg_method == 'mul':
            input_features = self.dropout(text_features * image_feature)
        else:
            input_features = self.dropout(torch.cat((text_features, image_feature), dim=1))
        output = self.classifier(input_features)
        return output