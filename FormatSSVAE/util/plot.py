import matplotlib.pyplot as plt

def plot_losses(losses, folder: str = "result", filename: str = "checkpoint.png"):
    x = list(range(len(losses)))
    plt.plot(x, losses, 'b--', label="Unsupervised Loss")
    plt.title("Loss Progression")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(loc='upper left')
    plt.savefig(f"{folder}/{filename}")
    plt.close()
    