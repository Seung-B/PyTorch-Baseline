import os
import torch
import csv

def evaluate_model(model, testloader, criterion, device, optimizer, args):
    model.eval()
    model.to(device)
    correct = 0
    total = 0
    test_loss = 0.0
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    print(f'Model: {model.__class__.__name__}, Optimizer: {optimizer.__class__.__name__}')
    print(f'Test Loss: {test_loss/len(testloader)}, Accuracy: {accuracy}%')
    os.makedirs('log', exist_ok=True)
    with open(f"log/{args.model}_pretrain_{args.pretrain}_{args.dataset}_{args.optimizer}_{args.learning_rate}_test_acc.csv", 'w') as f:
        write = csv.writer(f)
        write.writerow([accuracy])
