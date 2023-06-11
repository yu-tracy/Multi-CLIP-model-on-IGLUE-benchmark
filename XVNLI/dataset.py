from __future__ import print_function
import os
import json
import _pickle as cPickle
import numpy as np
# import h5py
import torch
from torch.utils.data import Dataset
from PIL import ImageFile
from PIL import Image
ImageFile.LOAD_TRUNCATED_IMAGES = True

# def _create_entry(question, answer):
#     answer.pop('image_id')
#     answer.pop('question_id')
#     entry = {
#         'question_id' : question['question_id'],
#         'image_id'    : question['image_id'],
#         'question'    : question['question'],
#         'answer'      : answer}
#     return entry

# def _load_dataset(dataroot, name):
#     """Load entries
#     dataroot: root path of dataset
#     name: 'train', 'val'
#     """
#     question_path = os.path.join(
#         dataroot, 'v2_OpenEnded_mscoco_%s2014_questions.json' % name)
#     questions = sorted(json.load(open(question_path))['questions'],
#                        key=lambda x: x['question_id'])
#     answer_path = os.path.join(dataroot, 'cache', '%s_target.pkl' % name)
#     answers = cPickle.load(open(answer_path, 'rb'))
#     answers = sorted(answers, key=lambda x: x['question_id'])

#     utils.assert_eq(len(questions), len(answers))
#     entries = []
#     for question, answer in zip(questions, answers):
#         utils.assert_eq(question['question_id'], answer['question_id'])
#         utils.assert_eq(question['image_id'], answer['image_id'])
#         entries.append(_create_entry(question, answer))

#     return entries

class XVNLIFeaturesDataset(Dataset):
    def __init__(self, name, transforms, tokenizer, few_shot='', dataroot='data'):
        super(XVNLIFeaturesDataset, self).__init__()
        annotations_path = os.path.join(dataroot, few_shot, f'{name}.jsonl')
        self.annotations = [json.loads(line) for line in open(annotations_path).readlines()]  
        self.image_dic = '/home/ndp689/iglue_xvnli/images/flickr30k-images/'
        # self.image_dic = '/Users/wy/Documents/KU/Master_thesis/thesis_project/flickr30k-images/'
        self.transforms = transforms
        self.tokenizer = tokenizer
        self.label2idx = {'contradiction': 2, 'neutral': 1, 'entailment': 0}
    
    def __getitem__(self, index):
        cur_annotation = self.annotations[index]
        ans, sentence, image_name = cur_annotation['gold_label'], cur_annotation['sentence2'], cur_annotation['Flikr30kID'] + '.jpg'
        # print(len(self.annotations))
        label = torch.zeros(3)
        if ans not in self.label2idx:
            pass
        else: label[self.label2idx[ans]] = 1.
        # get needed data 
        image = self.transforms(Image.open(f"{self.image_dic}{image_name}"))
        text = self.tokenizer(sentence)
        return torch.squeeze(text, dim=0), image, label

    def __len__(self):
        return len(self.annotations)