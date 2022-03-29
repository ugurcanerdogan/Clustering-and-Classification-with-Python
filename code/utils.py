import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix, roc_curve, \
    plot_precision_recall_curve
import seaborn as sns


def display_confusion_matrix(confusion_matrix):
    # Heatmap display for confusion matrix
    labels = ["True Neg", "False Pos", "False Neg", "True Pos"]
    length = len(max(labels)) + 10
    labels = np.asarray(labels).reshape(2, 2)

    annots = [f"{str(label)}({str(value)})" for array in np.dstack((labels, confusion_matrix)) for (label, value) in
              array]
    annots = np.asarray(annots).reshape(2, 2).astype(str)
    plt.figure(figsize=(12, 7))
    plt.title("Confusion Matrix of LogReg")
    sns.heatmap(confusion_matrix, annot=annots, fmt=f".{length}")
    plt.show()


# %%
def logistic_roc_curve(y_test, predictions):
    log_fpr, log_tpr, log_threshold = roc_curve(y_test, predictions)
    plt.figure(figsize=(12, 8))
    plt.title('Logistic Regression ROC Curve', fontsize=16)
    plt.plot(log_fpr, log_tpr, 'b-', linewidth=2)
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlabel('False Positive Rate', fontsize=16)
    plt.ylabel('True Positive Rate', fontsize=16)
    plt.axis([-0.01, 1, 0, 1])


# %%
def print_metric(y_test, predictions):
    precision = precision_score(y_test, predictions)
    recall = recall_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)
    accuracy = accuracy_score(y_test, predictions)
    _confusion_matrix = confusion_matrix(y_test, predictions)
    display_confusion_matrix(_confusion_matrix)
    print(f"Precision: {precision}, Recall: {recall}, F1: {f1}, Accuracy: {accuracy}")
