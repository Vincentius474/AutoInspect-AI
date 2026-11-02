from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix
)

def calculate_metrics(y_true, y_pred):
    """Calculates various classification metrics based on the true and predicted labels."""
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = (
        precision_recall_fscore_support(
            y_true,
            y_pred,
            labels=[0, 1, 2],
            average="macro",
            zero_division=0
        )
    )

    per_class = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=[0, 1, 2],
        average=None,
        zero_division=0
    )

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "class_precision": per_class[0],
        "class_recall": per_class[1],
        "class_f1": per_class[2],
        "confusion_matrix": confusion_matrix(
            y_true,
            y_pred,
            labels=[0, 1, 2]
        )
    }
