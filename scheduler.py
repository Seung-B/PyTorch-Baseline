from torch.optim.lr_scheduler import LambdaLR

def select_scheduler(optimizer, epochs, lr_decay):
    def lr_lambda(current_epoch):
        return max(0.0, float(epochs - current_epoch) / max(1, epochs - lr_decay))
    return LambdaLR(optimizer, lr_lambda)
