# pyTorch Baseline

This repository is the implementation of pyTorch Optimizer Experiment baseline. 


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

```
--dataset, type=str, default='MNIST', help='Dataset to use (default: MNIST)'
--model, type=str, default='SimpleMLP', help='Model to use (default: SimpleMLP)'
--optimizer, type=str, default='SGD', help='Optimizer to use (default: SGD)'
--learning_rate, type=float, default=0.01, help='Learning rate (default: 0.01)'
--batch_size, type=int, default=64, help='Batch size (default: 64)'
--epochs, type=int, default=10, help='Number of epochs (default: 10)'
--pretrain, action='store_true', help='Use pretrained model'
--lr_decay, type=int, default=10, help='Number of decay epochs (default: 10)'
--train_verbose, action='store_true', help='Print training progress'
--momentum, type=float, default=0.9, help='Momentum for SGD optimizer (default: 0.9)'
--beta1, type=float, default=0.9, help='Beta1 for Adam optimizer (default: 0.9)'
--beta2, type=float, default=0.999, help='Beta2 for Adam optimizer (default: 0.999)'
```


If you include ```--train_verbose``` in your command, all the training loss will be saved into 'log' directory.

## Evaluation

After the training is completed, the evaluation is automatically performed, and the accuracy is outputted. There is no need to enter a separate command.

## Pre-trained Models

You can use pre-trained models for ResNet18, ResNet50 and DenseNet121:

```
--pretrain --model <"Model Name : [resnet18, resnet50, densenet121]">
```

\begin{table}[t]
\caption{Classification accuracies for ResNet18 on SVHN dataset}
\label{sample-table}
\vskip 0.15in
\begin{center}
\begin{small}
\begin{tabular}{lcccr}
\toprule
Optimizer & learning rate & Train Loss & Accuracy\\
\midrule
SGD       & 0.1         & 4.97$e^{-8}$ & \textcolor{red}{93.22} \\
Adam      & 0.1         & 1.94$e^{-3}$ & 19.59 \\
AdaBelief & 0.1         & 1.98$e^{-3}$ & 19.59 \\
SAM       & 0.1         & 2.66$e^{-10}$ & 92.88 \\
SAM+Adam  & 0.1         & 1.92$e^{-3}$ & 19.59 \\
SGD       & 0.01        & 3.51$e^{-8}$ & 90.56 \\
Adam      & 0.01        & 2.79$e^{-11}$ & 93.06 \\
AdaBelief & 0.01        & 8.63$e^{-11}$ & \textcolor{red}{93.81} \\
SAM       & 0.01        & 7.36$e^{-11}$ & \textcolor{red}{94.02} \\
SAM+Adam  & 0.01        & 1.42$e^{-10}$ & 93.29 \\
SGD       & 0.001       & 1.03$e^{-5}$ & 85.43 \\
Adam      & 0.001       & 8.12$e^{-11}$ & \textcolor{red}{93.63} \\
AdaBelief & 0.001       & 1.78$e^{-10}$ & 93.67 \\
SAM       & 0.001       & 4.15$e^{-7}$ & 92.51 \\
SAM+Adam  & 0.001       & 5.08$e^{-12}$ & \textcolor{red}{93.94} \\
SGD       & 0.0001      & 3.72$e^{-3}$ & 83.94 \\
Adam      & 0.0001      & 6.86$e^{-11}$ & 92.40 \\
AdaBelief & 0.0001      & 1.01$e^{-10}$ & 92.44 \\
SAM       & 0.0001      & 8.55$e^{-5}$ & 91.15 \\
SAM+Adam  & 0.0001      & 2.54$e^{-12}$ & 93.78 \\
\bottomrule
\end{tabular}
\end{small}
\end{center}
\vskip -0.1in
\end{table}
