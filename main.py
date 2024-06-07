import argparse
import torch
import torch.nn as nn

from model import SimpleMLP, SimpleCNN, get_resnet, get_densenet
from data import load_dataset
from optimizer import select_optimizer
from scheduler import select_scheduler
from train import train_model
from eval import evaluate_model


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    trainloader, testloader, input_channels, num_classes = load_dataset(args.dataset, args.batch_size)

    # 모델 선택
    if args.model == 'SimpleMLP':
        model = SimpleMLP(input_channels, num_classes)
    elif args.model == 'SimpleCNN':
        model = SimpleCNN(input_channels, num_classes)
    elif args.model in ['resnet18', 'resnet50']:
        model = get_resnet(args.model, input_channels, num_classes, args.pretrain)
    elif args.model == 'densenet':
        model = get_densenet(input_channels, num_classes, args.pretrain)
    else:
        raise ValueError("Unsupported model")

    criterion = nn.CrossEntropyLoss()

    optimizer = select_optimizer(args.optimizer, model, args.learning_rate, args.momentum, (args.beta1, args.beta2))
    scheduler = select_scheduler(optimizer, args.epochs, args.lr_decay)
    train_model(model, trainloader, criterion, optimizer, scheduler, args.epochs, args.train_verbose, device, args)

    evaluate_model(model, testloader, criterion, device, optimizer, args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='MNIST', help='Dataset to use (default: MNIST)')
    parser.add_argument('--model', type=str, default='SimpleMLP', help='Model to use (default: SimpleMLP)')
    parser.add_argument('--optimizer', type=str, default='SGD', help='Optimizer to use (default: SGD)')
    parser.add_argument('--learning_rate', type=float, default=0.01, help='Learning rate (default: 0.01)')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size (default: 64)')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs (default: 10)')
    parser.add_argument('--pretrain', action='store_true', help='Use pretrained model')
    parser.add_argument('--lr_decay', type=int, default=10, help='Number of decay epochs (default: 10)')
    parser.add_argument('--train_verbose', action='store_true', help='Print training progress')
    parser.add_argument('--momentum', type=float, default=0.9, help='Momentum for SGD optimizer (default: 0.9)')
    parser.add_argument('--beta1', type=float, default=0.9, help='Beta1 for Adam optimizer (default: 0.9)')
    parser.add_argument('--beta2', type=float, default=0.999, help='Beta2 for Adam optimizer (default: 0.999)')

    args = parser.parse_args()
    main(args)
