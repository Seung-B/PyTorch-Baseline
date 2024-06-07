import torch

def train_model(model, trainloader, criterion, optimizer, scheduler, epochs, verbose, device, args):
    model.train()
    model.to(device)
    for epoch in range(epochs):
        running_loss = 0.0
        for inputs, labels in trainloader:
            inputs, labels = inputs.to(device), labels.to(device)
            if "SAM" in args.optimizer:
                def closure():
                    loss = criterion(model(inputs), labels)
                    loss.backward()
                    return loss
                loss = criterion(model(inputs), labels)
                loss.backward()
                optimizer.step(closure)
                optimizer.zero_grad()
            else:
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
        scheduler.step()
        running_loss += loss.item()
        if verbose:
            print(f'Epoch {epoch+1}, Loss: {running_loss/len(trainloader)}')
