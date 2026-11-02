from src.training.trainer import train_model
from src.training.plot_history import plot_history

def main():
    history = train_model(
        epochs=50,
        batch_size=16,
        learning_rate=0.0005,
        patience=10,
        seed=42
    )
    plot_history(history)

if __name__ == "__main__":
    main()
