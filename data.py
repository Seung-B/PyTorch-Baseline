import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST, CIFAR10, CIFAR100, SVHN


def load_dataset(dataset_name, batch_size):
    if dataset_name == 'MNIST':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        train_dataset = MNIST(root='./data', train=True, transform=transform, download=True)
        test_dataset = MNIST(root='./data', train=False, transform=transform, download=True)
        input_channels = 1
        num_classes = 10
    elif dataset_name == 'CIFAR10':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        train_dataset = CIFAR10(root='./data', train=True, transform=transform, download=True)
        test_dataset = CIFAR10(root='./data', train=False, transform=transform, download=True)
        input_channels = 3
        num_classes = 10
    elif dataset_name == 'CIFAR100':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        train_dataset = CIFAR100(root='./data', train=True, transform=transform, download=True)
        test_dataset = CIFAR100(root='./data', train=False, transform=transform, download=True)
        input_channels = 3
        num_classes = 100
    elif dataset_name == 'SVHN':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        train_dataset = SVHN(root='./data', split='train', transform=transform, download=True)
        test_dataset = SVHN(root='./data', split='test', transform=transform, download=True)
        input_channels = 3
        num_classes = 10
    else:
        raise ValueError("Unsupported dataset")

    trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    testloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return trainloader, testloader, input_channels, num_classes
