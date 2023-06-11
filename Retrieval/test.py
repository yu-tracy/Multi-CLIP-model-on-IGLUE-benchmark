import os
import json
from io import open
from tqdm import tqdm

import numpy as np

import torch
import torch.nn as nn
import torch.distributed as dist
import argparse
from torch.utils.data import DataLoader
# from train import compute_score_with_logits
from dataset import RetrievalDatasetVal
from classifier import RetrievalXMLCLIP
import open_clip
import wandb




def parse_args():
    parser = argparse.ArgumentParser()

#     # Model
#     parser.add_argument("--from_pretrained", default="bert-base-uncased", type=str,
#                         help="Bert pre-trained model selected in the list: bert-base-uncased, "
#                              "bert-large-uncased, bert-base-cased, bert-base-multilingual, bert-base-chinese.")
#     parser.add_argument("--bert_model", default="bert-base-uncased", type=str,
#                         help="Bert pre-trained model selected in the list: bert-base-uncased, "
#                              "bert-large-uncased, bert-base-cased, bert-base-multilingual, bert-base-chinese.")
#     parser.add_argument("--config_file", default="config/bert_config.json", type=str,
#                         help="The config file which specified the model details.")
#     parser.add_argument("--is_m3p", action='store_true', default=False,
#                         help="Use M3P.")
#     # Output
#     parser.add_argument("--output_dir", default="results", type=str,
#                         help="The output directory where the model checkpoints will be written.")
#     parser.add_argument("--save_name", default="", type=str,
#                         help="save name for training.")
#     # Task
#     parser.add_argument("--tasks_config_file", default="config_tasks/vilbert_trainval_tasks.yml", type=str,
#                         help="The config file which specified the tasks details.")
#     parser.add_argument("--task", default="", type=str,
#                         help="training task number")
#     parser.add_argument("--val_annotations_jsonpath", default="", type=str)
    parser.add_argument("--num_subiters", default=1, type=int)
#     parser.add_argument("--caps_per_image", default=5, type=int,
#                         help="Num captions per image")
#     parser.add_argument("--val_features_lmdbpath", default="", type=str)
#     # Text
#     parser.add_argument("--do_lower_case", action='store_true', default=False,
#                         help="Whether to lower case the input text. True for uncased models, False for cased models.")
#     # Evaluation
#     parser.add_argument("--split", default="", type=str,
#                         help="which split to use.")
    parser.add_argument("--batch_size", default=1, type=int,
                        help="batch size.")
#     parser.add_argument("--drop_last", action="store_true",
#                         help="whether to drop last incomplete batch")
#     # Seed
#     parser.add_argument("--seed", type=int, default=42,
#                         help="random seed for initialization")
#     # Distributed
#     parser.add_argument("--local_rank", type=int, default=-1,
#                         help="local_rank for distributed training on gpus")
#     parser.add_argument("--num_workers", type=int, default=16,
#                         help="Number of workers in the dataloader.")
#     parser.add_argument("--num_val_workers", type=int, default=10)
#     parser.add_argument("--in_memory", default=False, type=bool,
#                         help="whether use chunck for parallel training.")
#     parser.add_argument("--use_chunk", default=0, type=float,
#                         help="whether use chunck for parallel training.")

    return parser.parse_args()


def main():
    zero_shot = True
    savePath = 'results'
    args = parse_args()

    # Devices
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    clip_model, _, preprocess_val = open_clip.create_model_and_transforms('xlm-roberta-base-ViT-B-32', pretrained='laion5b_s13b_b90k', device=device)
    tokenizer = open_clip.get_tokenizer('xlm-roberta-base-ViT-B-32')

    in_dim = 512 * 2
    # import ipdb
    # ipdb.set_trace()
    out_dim = 1
    model = RetrievalXMLCLIP(clip_model, in_dim, out_dim).to(device)
    PATH = "/home/ndp689/iglue_retrieval/iglue_RetrievalXMLCLIP.pt"
    model.load_state_dict(torch.load(PATH))
    model.eval()

    batch_size = args.batch_size

    test_sets = ['test_en', 'test_de', 'test_es', 'test_id', 'test_ja', 'test_ru', 'test_tr', 'test_zh']
    for test_set in test_sets:
        print(f'language - {test_set}')

        dset_val = RetrievalDatasetVal('xFlickrCO', test_set, test_set, tokenizer, preprocess_val, args.num_subiters)
        dl_val = DataLoader(dset_val, batch_size=batch_size)
        num_iters = len(dl_val)

        results = []
        score_matrix = np.zeros((dset_val.num_entries, dset_val.num_images))
        target_matrix = np.zeros((dset_val.num_entries, dset_val.num_images))
        rank_matrix = np.ones(dset_val.num_entries) * dset_val.num_images
        count = 0
        for i, batch in tqdm(enumerate(dl_val), total=num_iters):
            batch = tuple(t.cuda(device=device, non_blocking=True) for t in batch)
            features, question, target, caption_idx, image_idx = batch
            features = features.squeeze(0)

            question = question.repeat(features.size(0), 1)
            target = target.view(-1).float().cpu().numpy()

            with torch.no_grad():
                if zero_shot:
                    vil_logit = model(features, question)
                    vil_logit = vil_logit.view(-1)

                    score_matrix[
                        caption_idx, image_idx * dset_val.max_num_images: image_idx * dset_val.max_num_images + len(target)
                    ] = (torch.softmax(vil_logit, dim=0).cpu().numpy())
                    target_matrix[
                        caption_idx, image_idx * dset_val.max_num_images: image_idx * dset_val.max_num_images + len(target)
                    ] = (target)

                # else:
                #     vil_logit, _, _, _ = model(question, features, spatials, task, segment_ids, input_mask, image_mask)

                #     score_matrix[
                #         caption_idx, image_idx * dset_val.max_num_images: image_idx * dset_val.max_num_images + len(target)
                #     ] = (vil_logit.view(-1).cpu().numpy())
                #     target_matrix[
                #         caption_idx, image_idx * dset_val.max_num_images: image_idx * dset_val.max_num_images + len(target)
                #     ] = (target)

                if image_idx.item() == (args.num_subiters - 1):
                    rank = np.where((np.argsort(-score_matrix[caption_idx])== np.where(target_matrix[caption_idx] == 1)[0][0])== 1)[0][0]
                    rank_matrix[caption_idx] = rank
                    rank_matrix_tmp = rank_matrix[: caption_idx + 1]
                    r1 = np.sum(rank_matrix_tmp < 1) / len(rank_matrix_tmp)
                    r5 = np.sum(rank_matrix_tmp < 5) / len(rank_matrix_tmp)
                    r10 = np.sum(rank_matrix_tmp < 10) / len(rank_matrix_tmp)

                    medr = np.floor(np.median(rank_matrix_tmp) + 1)
                    meanr = np.mean(rank_matrix_tmp) + 1

                    if count % 100 == 0:
                        print(
                            "%d Final r1:%.3f, r5:%.3f, r10:%.3f, mder:%.3f, meanr:%.3f"
                            % (count, r1, r5, r10, medr, meanr)
                        )
                        print(features.shape)

                    results.append(np.argsort(-score_matrix[caption_idx]).tolist()[:20])
            count += 1

        r1 = np.sum(rank_matrix < 1) / len(rank_matrix)
        r5 = np.sum(rank_matrix < 5) / len(rank_matrix)
        r10 = np.sum(rank_matrix < 10) / len(rank_matrix)

        medr = np.floor(np.median(rank_matrix) + 1)
        meanr = np.mean(rank_matrix) + 1

        print("************************************************")
        print("****************Image Retrieval*****************")
        print("************************************************")
        print(
            "Final r1:%.3f, r5:%.3f, r10:%.3f, mder:%.3f, meanr:%.3f"
            % (r1, r5, r10, medr, meanr)
        )
        print("************************************************")


        json_path = os.path.join(savePath, test_set)

        json.dump(results, open(json_path + "_result.json", "w"))

        # Text Retrieval
        rank_matrix = np.zeros(dset_val.num_images)
        for image_idx in range(dset_val.num_images):
            ranks = []
            tgt_captions = np.where(target_matrix[:, image_idx] == 1)[0]
            sorted_scores = np.argsort(-score_matrix[:, image_idx])
            for tgt_caption in tgt_captions:
                ranks.append(np.where((sorted_scores == tgt_caption) == 1)[0][0])
            rank_matrix[image_idx] = min(ranks)

        r1 = 100.0 * np.sum(rank_matrix < 1) / len(rank_matrix)
        r5 = 100.0 * np.sum(rank_matrix < 5) / len(rank_matrix)
        r10 = 100.0 * np.sum(rank_matrix < 10) / len(rank_matrix)

        medr = np.floor(np.median(rank_matrix) + 1)
        meanr = np.mean(rank_matrix) + 1

        print("************************************************")
        print("****************Text Retrieval******************")
        print("************************************************")
        print(
            "Final r1:%.3f, r5:%.3f, r10:%.3f, mder:%.3f, meanr:%.3f"
            % (r1, r5, r10, medr, meanr)
        )
        print("************************************************")


if __name__ == "__main__":
    main()