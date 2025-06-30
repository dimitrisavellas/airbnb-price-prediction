import numpy as np
import matplotlib.pyplot as plt

def lr_scheduler(optimizer, epoch=0, interval=5, start=0):
    # decreases optimizer's learning rate in half 
    if epoch > start and epoch % interval == 0:
        for param_group in optimizer.param_groups:
            param_group["lr"] *= 0.5
        # print("Learning rate decreased")
    return optimizer

def early_stop(val_loss, val_list, epoch=0, patience=5):
    # early stopping
    if val_loss >= min(val_list) and (epoch - np.argmin(np.array(val_list))) > patience:
        print(f"Training stopped at epoch: {epoch}\n")
        return True
    return False

def plot_loss(train_losses, val_losses):
    # plot training and validation loss per epoch
    epochs = len(train_losses)
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, epochs + 1), train_losses, label="Train Loss")
    plt.plot(range(1, epochs + 1), val_losses, label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss per Epoch")
    plt.legend()
    plt.show()
    plt.close()