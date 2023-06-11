from __future__ import print_function
import os
import json
import _pickle as cPickle
import numpy as np
import utils
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


class VQAFeatureDataset(Dataset):
    def __init__(self, name, transforms, tokenizer, dataroot='data_gqa'):
        super(VQAFeatureDataset, self).__init__()
        assert name in ['train', 'val']
        self.name = name

        ans2label_path = os.path.join(dataroot, 'cache', 'trainval_ans2label.pkl')
        label2ans_path = os.path.join(dataroot, 'cache', 'trainval_label2ans.pkl')
        self.ans2label = cPickle.load(open(ans2label_path, 'rb'))
        self.label2ans = cPickle.load(open(label2ans_path, 'rb'))
        self.num_ans_candidates = len(self.ans2label)
        self.transforms = transforms
        self.tokenizer = tokenizer
        # self.image_dic = '/Users/wy/Downloads/images/'
        self.image_dic = f'/home/ndp689/iglue_gqa/images/'

        target_path = os.path.join(dataroot, 'cache', f'{name}_target.pkl')
        self.entries = cPickle.load(open(target_path, 'rb'))
        self.tensorize()

    def tensorize(self):
        for entry in self.entries:
            labels = np.array(entry['labels'])
            scores = np.array(entry['scores'], dtype=np.float32)
            if len(labels):
                labels = torch.from_numpy(labels)
                scores = torch.from_numpy(scores)
                entry['labels'] = labels
                entry['scores'] = scores
            else:
                entry['labels'] = None
                entry['scores'] = None

    def __getitem__(self, index):
        entry = self.entries[index]
        
        question = self.tokenizer(entry['question'])
        labels = entry['labels']
        scores = entry['scores']
        target = torch.zeros(self.num_ans_candidates)
        if labels is not None:
            target.scatter_(0, labels, scores)
        # image_name = 'COCO_' + self.name + '2014_' + str(entry['image_id']).zfill(12) + '.jpg'
        image_name = f"{entry['image_id']}.jpg"
        image = self.transforms(Image.open(f"{self.image_dic}{image_name}"))

        return torch.squeeze(question, dim=0), image, target

    def __len__(self):
        return len(self.entries)
