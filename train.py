import torch
import csv
import os

def train_model(model, trainloader, criterion, optimizer, scheduler, epochs, verbose, device, args):
    model.train()
    model.to(device)
    loss_log = []
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
            loss_log.append([epoch+1, running_loss/len(trainloader)])
            print(f'Epoch {epoch+1}, Loss: {running_loss/len(trainloader)}')

    if verbose:
        os.makedirs('log', exist_ok=True)
        with open(f"log/{args.model}_pretrain_{args.pretrain}_{args.dataset}_{args.optimizer}_{args.learning_rate}_train_loss.csv", 'w') as f:
            write = csv.writer(f)
            write.writerow(loss_log)

