import matplotlib.pyplot as plt
import seaborn as sns
import os


def plot_heatmap(matrix, title, dir, name):
    """Plot heatmap"""
    if not os.path.exists(dir):
        os.makedirs(dir)
    path = os.path.join(dir, name + ".png")

    sns.heatmap(
        matrix,
        cbar=True,
        annot=True,  # Inject numbers
        square=True,  # Cells are square
        fmt=".2f",  # String format
        cmap="rainbow",  # Color
    )

    plt.title(title)
    plt.xlabel("w")
    plt.ylabel("h")
    plt.savefig(path)
    plt.close()


def plot_loss_and_metric(loss_dict, metric_dict, title, dir, name):
    """Combine and plot loss and metrics"""
    if not os.path.exists(dir):
        os.makedirs(dir)

    # Create a 2x1 subplot and plot loss and metrics separately
    fig, axes = plt.subplots(2, 1, figsize=(10, 10))

    # Plot loss graph
    for loss_key, loss_value in loss_dict.items():
        axes[0].plot(loss_value, label=loss_key)
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("loss")
    axes[0].legend()
    axes[0].grid()
    axes[0].set_title("Loss")

    # Plot metrics graph
    for metric_key, metric_value in metric_dict.items():
        axes[1].plot(metric_value, label=metric_key)
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("metric")
    axes[1].legend()
    axes[1].grid()
    axes[1].set_title("Metric")

    # Set overall title
    fig.suptitle(title, fontsize=16)

    # Save the combined image
    path = os.path.join(dir, name + ".png")
    plt.savefig(path)
    plt.close()
