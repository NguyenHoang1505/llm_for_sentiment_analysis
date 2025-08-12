import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def evaluate(y_true, y_pred, output_file):
    # Define labels and mapping for sentiment categories
    labels = ['tích cực', 'trung lập', 'tiêu cực']
    mapping = {'tích cực': 2, 'trung lập': 1, 'none': 1, 'tiêu cực': 0}

    # Function to map labels to numerical values
    def map_func(x):
        return mapping.get(x, 1)

    # Map true and predicted labels using the mapping function
    y_true = np.vectorize(map_func)(y_true)
    y_pred = np.vectorize(map_func)(y_pred)

    # Calculate overall accuracy
    accuracy = accuracy_score(y_true=y_true, y_pred=y_pred)
    overall_accuracy = f'Overall Accuracy: {accuracy:.3f}'
    print(overall_accuracy)

    # Generate accuracy report for each sentiment label
    unique_labels = set(y_true)  # Get unique labels
    label_accuracies = []

    for label in unique_labels:
        # Find indices where true labels match the current sentiment label
        label_indices = [i for i in range(len(y_true)) if y_true[i] == label]
        label_y_true = [y_true[i] for i in label_indices]
        label_y_pred = [y_pred[i] for i in label_indices]

        # Calculate accuracy for the current sentiment label
        accuracy = accuracy_score(label_y_true, label_y_pred)
        label_accuracy = f'Accuracy for label {labels[label]}: {accuracy:.3f}'
        label_accuracies.append(label_accuracy)
        print(label_accuracy)

    # Generate classification report for precision, recall, and F1-score
    class_report = classification_report(y_true=y_true, y_pred=y_pred, target_names=labels)
    print('\nClassification Report:')
    print(class_report)

    # Generate confusion matrix to analyze the performance of the model
    conf_matrix = confusion_matrix(y_true=y_true, y_pred=y_pred, labels=[0, 1, 2])
    print('\nConfusion Matrix:')
    print(conf_matrix)

    # Save all evaluation information to the output file
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(overall_accuracy + '\n\n')
        f.write('\n'.join(label_accuracies) + '\n\n')
        f.write('Classification Report:\n')
        f.write(class_report + '\n\n')
        f.write('Confusion Matrix:\n')
        f.write(np.array2string(conf_matrix) + '\n')