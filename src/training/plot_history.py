import matplotlib.pyplot as plt

def plot_history(history):

    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    axes[0].plot(
        epochs,
        history["train_loss"],
        label="Training Loss"
    )

    axes[0].plot(
        epochs,
        history["val_loss"],
        label="Validation Loss"
    )

    axes[0].set_title("Training and Validation Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(
        epochs,
        history["train_f1"],
        label="Training Macro F1"
    )

    axes[1].plot(
        epochs,
        history["val_f1"],
        label="Validation Macro F1"
    )

    axes[1].set_title("Model Performance")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Macro F1")
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.savefig(
        "reports/figures/training_history.png",
        dpi=150
    )

    plt.show()
