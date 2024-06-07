import torch

def evaluate_model(model, testloader, criterion, device, optimizer):
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
