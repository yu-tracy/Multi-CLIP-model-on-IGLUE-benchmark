# Multi-CLIP-model-on-IGLUE-benchmark

The multilingual multi-modal (image and text) researches focus on collecting resources, building models, and evaluating models.

The [IGLUE benchmark tasks](https://arxiv.org/abs/2201.11732) aims to evaluate multi-modal models’ performance from different aspects, including 4 different tasks: visual natural language inference, visual question answering, visual reasoning, and image-text retrieval. The [Multi-CLIP model](https://huggingface.co/laion/CLIP-ViT-B-32-xlm-roberta-base-laion5B-s13B-b90k) is a multilingual version of [CLIP model](https://openai.com/research/clip) and trained with the full [LAION-5B](https://laion.ai/blog/laion-5b/) dataset.

This project aims to find out whether the Multi-CLIP model can be used for the IGLUE benchmark tasks, and to see how well the model works for each task compared with other models in IGLUE.

All the relevant datasets are from [here](https://github.com/e-bug/iglue/tree/main/datasets).