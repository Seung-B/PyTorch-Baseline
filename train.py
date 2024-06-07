import torch

def train_model(model, trainloader, criterion, optimizer, scheduler, epochs, verbose, device, args):
    model.train()
    model.to(device)
    for epoch in range(epochs):
        running_loss = 0.0
        for inputs, labels in trainloader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            if "SAM" in args.optimizer:
                optimizer.first_step(zero_grad=True)
                criterion(outputs, model(inputs)).backward()
                optimizer.second_step(zero_grad=True)
            else:
                optimizer.step()
        scheduler.step()
        running_loss += loss.item()
        if verbose:
            print(f'Epoch {epoch+1}, Loss: {running_loss/len(trainloader)}')
