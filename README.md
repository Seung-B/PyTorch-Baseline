# pyTorch Baseline

This repository is the implementation of pyTorch Optimizer Experiment baseline. 

>📋  Optional: include a graphic explaining your approach/main result, bibtex entry, link to demos, blog posts and tutorials

## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```

>📋  Dataset will be downloaded automatically.

## Training

To train the model(s) and re-produce the numbers in the paper, run this shell:

```
paper_reproduce_shell.sh
```

To make your own experiments, here's arguments that you can use in this code.

### Arguments

Train: TT

## Evaluation

After the training is completed, the evaluation is automatically performed, and the accuracy is outputted. There is no need to enter a separate command.

## Pre-trained Models

You can use pre-trained models for ResNet18, ResNet50 and DenseNet121:

```
--pretrain --model <"Model Name : [resnet18, resnet50, densenet121]">
```

## Results

Our model achieves the following performance on :

### [Image Classification on ImageNet](https://paperswithcode.com/sota/image-classification-on-imagenet)

| Model name         | Top 1 Accuracy  | Top 5 Accuracy |
| ------------------ |---------------- | -------------- |
| My awesome model   |     85%         |      95%       |

>📋  Include a table of results from your paper, and link back to the leaderboard for clarity and context. If your main result is a figure, include that figure and link to the command or notebook to reproduce it. 
