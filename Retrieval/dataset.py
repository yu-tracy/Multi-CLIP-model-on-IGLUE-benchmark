# Copyright (c) Facebook, Inc. and its affiliates.
# Copyright (c) 2020, Emanuele Bugliarello (@e-bug).

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import sys
import random
import jsonlines
import _pickle as cPickle
from PIL import ImageFile
from PIL import Image
ImageFile.LOAD_TRUNCATED_IMAGES = True
import numpy as np
import torch
from torch.utils.data import Dataset


def assert_eq(real, expected):
    assert real == expected, "%s (true) vs %s (expected)" % (real, expected)

def _load_annotations(annotations_jsonpath, task):
    with jsonlines.open(annotations_jsonpath) as reader:
        # Build an index which maps image id with a list of caption annotations.
        # entries: contains 'caption' and 'image_name'
        # imgid2entry: a mapping from 'image_name' to corrsponding index related to the same image
        entries = []
        imgid2entry = {}
        count = 0
        for annotation in reader:
            if task == "COCO":
                image_id = annotation["id"]
            elif task == "flickr30k": # or task.startswith("RetrievalMulti30k"):
                image_id = int(annotation["img_path"].split(".")[0])
            imgid2entry[image_id] = []

            for sentences in annotation["sentences"]:
                entries.append({"caption": sentences, "image_id": image_id})
                imgid2entry[image_id].append(count)
                count += 1
    return entries, imgid2entry


class RetrievalDataset(Dataset):
    def __init__(
        self,
        task: str,
        annotation_name: str,
        split: str,
        tokenizer,
        transforms
    ):
        # All the keys in `self._entries` would be present in `self._image_features_reader`
        self._entries, self.imgid2entry = _load_annotations(f'data/{task}/{annotation_name}.jsonl', task)
        self.image_id_list = [*self.imgid2entry] # list of image_name

        self._transforms = transforms
        self._tokenizer = tokenizer
        self.num_labels = 1
        self._split = split
        # self._image_dic = '/Users/wy/Documents/KU/Master_thesis/thesis_project/flickr30k-images/'
        self._image_dic = '/home/ndp689/iglue_xvnli/images/flickr30k-images/'
        dataroot = 'data'
        if self._split == "train":
            image_info = cPickle.load(open(os.path.join(dataroot, "hard_negative" + ".pkl"), "rb"))
            for key, value in image_info.items():
                setattr(self, key, value) 
            # train_image_list: is image_name
            # train_imgId2pool: a mapping from image_name to index
            self.train_imgId2pool = {imageId: i for i, imageId in enumerate(self.train_image_list)}

        os.makedirs(os.path.join('data', "cache"), exist_ok=True)
        cache_path = os.path.join(
            'data', "cache",
            task
            + "_"
            + split
            + ".pkl"
        )

        if not os.path.exists(cache_path):
            self.tokenize()
            self.tensorize() 
            cPickle.dump(self._entries, open(cache_path, "wb"))
        else:
            print("loading entries from %s" % cache_path)
            self._entries = cPickle.load(open(cache_path, "rb"))

    def tokenize(self):
        """Tokenizes the captions.

        This will add caption_tokens in each entry of the dataset.
        """
        for entry in self._entries:
            tokens = self._tokenizer(entry["caption"])
            entry["token"] = tokens

    def tensorize(self):
        for entry in self._entries:
            token = torch.from_numpy(np.array(entry["token"]))
            entry["token"] = token

    def __getitem__(self, index):
        entry = self._entries[index]
        image_id = entry["image_id"] # image_name

        image1 = self._transforms(Image.open(f"{self._image_dic}{image_id}.jpg")) # image features
        caption1 = entry["token"]

        # negative samples.
        # 1: correct one, 2: random caption wrong, 3: random image wrong. 4: hard image wrong.

        while True:
            # sample a random image:
            img_id2 = random.choice(self.image_id_list)
            if img_id2 != image_id:
                entry2 = self._entries[random.choice(self.imgid2entry[img_id2])]
                break

        image2 = image1
        caption2 = entry2["token"]

        # random image wrong
        while True:
            # sample a random image:
            img_id3 = random.choice(self.image_id_list)
            if img_id3 != image_id:
                break
            # elif len(self.image_id_list) == 1:
            #     img_id3 = random.choice(self._image_features_reader._image_ids).decode()
            #     break

        image3 = self._transforms(Image.open(f"{self._image_dic}{img_id3}.jpg")) # image features
        caption3 = caption1

        if self._split == "train":
            # random hard caption.
            rand_img_id_pool = self.train_hard_pool[self.train_imgId2pool[image_id]]
            pool_img_idx = int(rand_img_id_pool[np.random.randint(1, len(rand_img_id_pool))])
            img_id4 = self.train_image_list[pool_img_idx]
            entry4 = self._entries[random.choice(self.imgid2entry[img_id4])]
        else:
            while True:
                # sample a random image:
                img_id4 = random.choice(self.image_id_list)
                if img_id4 != image_id:
                    entry4 = self._entries[random.choice(self.imgid2entry[img_id4])]
                    break

        image4 = image1
        caption4 = entry4["token"]

        features = torch.stack([image1, image2, image3, image4], dim=0)
        caption = torch.stack([caption1, caption2, caption3, caption4], dim=0)
        caption = torch.squeeze(caption, dim=1)
        target = 0

        return features, caption, target, image_id

    def __len__(self):
        return len(self._entries)


def _load_annotationsVal(annotations_jsonpath, task):
    with jsonlines.open(annotations_jsonpath) as reader:
        # Build an index which maps image id with a list of caption annotations.
        image_entries = {}
        caption_entries = []
        for annotation in reader:
            if task == "COCO":
                image_id = annotation["id"]
            elif task == "flickr30k":
                image_id = int(annotation["img_path"].split(".")[0])
            elif task == "xFlickrCO":
                image_id = annotation["img_path"]
            image_entries[image_id] = 1

            for sentences in annotation["sentences"]:
                caption_entries.append({"caption": sentences, "image_id": image_id})
    image_entries = [*image_entries]
    return image_entries, caption_entries


class RetrievalDatasetVal(Dataset):
    def __init__(
        self,
        task: str,
        annotation_name: str,
        split: str,
        tokenizer,
        transforms,
        num_subiters=1,
    ):
        # _image_entries: list of image_name
        # _caption_entries: contains 'caption' and 'image_id'
        self._image_entries, self._caption_entries = _load_annotationsVal(f'data/{task}/{annotation_name}.jsonl', task)
        self._tokenizer = tokenizer
        self._transforms = transforms
        # self._image_dic = '/Users/wy/Documents/KU/Master_thesis/thesis_project/iglue_code/xFlickerCo_images/'
        self._image_dic = '/home/ndp689/iglue_xvnli/images/xFlickerCo_images/'
        self._split = split
        self.num_labels = 1

        self.num_subiters = num_subiters
        self.num_images = len(self._image_entries)
        self.num_entries = len(self._caption_entries)
        self.max_num_images = self.num_images // self.num_subiters + int(self.num_images % self.num_subiters > 0)
        # self.max_num_images = 3

        os.makedirs(os.path.join('data', "cache"), exist_ok=True)
        cache_path = os.path.join(
            'data',
            "cache",
            task
            + "_"
            + split
            + ".pkl",
        )
        if not os.path.exists(cache_path):
            self.tokenize()
            self.tensorize()
            cPickle.dump(self._caption_entries, open(cache_path, "wb"))
        else:
            print("loading entries from %s" % cache_path)
            self._caption_entries = cPickle.load(open(cache_path, "rb"))

        self.features_all = []
        for i, image_id in enumerate(self._image_entries):
            image = self._transforms(Image.open(f"{self._image_dic}{image_id}")) # image features

            self.features_all.append(torch.unsqueeze(image, dim=0))

            sys.stdout.write("%d/%d\r" % (i, len(self._image_entries))) # progress
            sys.stdout.flush()
        self.features_all = torch.cat(self.features_all, dim=0)

    def tokenize(self):
        """Tokenizes the captions.

        This will add caption_tokens in each entry of the dataset.
        -1 represents nil, and should be treated as padding_idx in embedding.
        """
        for entry in self._caption_entries:
            tokens = self._tokenizer(entry["caption"])
            entry["token"] = tokens

    def tensorize(self):
        for entry in self._caption_entries:
            token = torch.from_numpy(np.array(entry["token"])).long()
            entry["token"] = token

    def __getitem__(self, index):
        
        # we iterate through every caption here.
        caption_idx = int(index / self.num_subiters)
        image_idx = index % self.num_subiters

        # return self.max_num_images images
        image_entries = self._image_entries[self.max_num_images * (image_idx):self.max_num_images * (image_idx + 1)]
        features_all = self.features_all[self.max_num_images * (image_idx):self.max_num_images * (image_idx + 1)]

        entry = self._caption_entries[caption_idx]
        caption = entry["token"]
        caption = torch.squeeze(caption, dim=0)

        target_all = torch.zeros(len(image_entries))
        for i, image_id in enumerate(image_entries):
            if image_id == entry["image_id"]:
                target_all[i] = 1
        
        return (
            features_all,
            caption,
            target_all,
            caption_idx,
            image_idx
        )

    def __len__(self):
        return len(self._caption_entries) * self.num_subiters