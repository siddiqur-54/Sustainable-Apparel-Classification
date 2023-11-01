import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.utils.multiclass import unique_labels
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

from human_interact import human_expertise


def execute_model(model,x_train,y_train,x_test,y_test):

    # Splitting the original training data training and validation sets
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=42)

    # Compiling the model
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    # Training the model on the training data
    history = model.fit(x_train.values.reshape(-1, 28, 28, 1), y_train, epochs=10, batch_size=64,
                        validation_data=(x_val.values.reshape(-1, 28, 28, 1), y_val))

    # Plotting training & validation accuracy values
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(['Train', 'Validation'], loc='upper left')

    # Plotting training & validation loss values
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(['Train', 'Validation'], loc='upper left')

    # Predicting probabilities for each class
    y_pred_probabilities = model.predict(x_test.values.reshape(-1, 28, 28, 1))
    
    # Converting probabilities to class labels
    y_pred = np.argmax(y_pred_probabilities, axis=1)
    
    # Computing classification accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print("Classification Accuracy:", accuracy)

    # Generating the classification report
    class_labels = ['Class {}'.format(i) for i in range(len(np.unique(y_test)))]
    print(classification_report(y_test, y_pred, target_names=class_labels))

    # Generating confusion matrix
    classes = unique_labels(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred, labels=classes)

    # Plotting confusion matrix with annotations
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = classes
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)

    # Annotate the cells with the actual values
    for i in range(len(classes)):
        for j in range(len(classes)):
            plt.text(j, i, str(cm[i, j]), ha='center', va='center', color='blue')

    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show(block=False)


    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred)

    # Calculating accuracy for each class without human intervention
    class_accuracies = []
    for class_label in range(len(np.unique(y_test))):
        class_mask = (y_test == class_label)
        class_accuracy = accuracy_score(y_test[class_mask], y_pred[class_mask])
        class_accuracies.append((class_label, class_accuracy))

    # Finding class with maximum accuracy
    max_accuracy_class = max(class_accuracies, key=lambda x: x[1])

    # Finding class with minimum accuracy
    min_accuracy_class = min(class_accuracies, key=lambda x: x[1])

    # Placing human in the loop
    y_pred_probabilities, updated_accuracy=human_expertise(y_pred_probabilities,y_test)
    print(f"Accuracy after human-in-the-loop intervention: {updated_accuracy}")

    # Converting probabilities to class labels
    y_pred = np.argmax(y_pred_probabilities, axis=1)

    # Calculating accuracy for each class after human intervention
    class_accuracies_human = []
    for class_label_human in range(len(np.unique(y_test))):
        class_mask_human = (y_test == class_label)
        class_accuracy_human = accuracy_score(y_test[class_mask_human], y_pred[class_mask_human])
        class_accuracies_human.append((class_label_human, class_accuracy_human))

    # Finding class with maximum accuracy
    max_accuracy_class_human = max(class_accuracies_human, key=lambda x: x[1])

    # Finding class with minimum accuracy
    min_accuracy_class_human = min(class_accuracies_human, key=lambda x: x[1])

    # Additional insights or observations
    additional_insights = "\nWrite additional insights or observations here.\n"

    # Save the outputs to output.txt
    with open('output.txt', 'w') as f:
        f.write("Model's Architecture Summary:\n")
        model.summary(print_fn=lambda x: f.write(x + '\n'))

        f.write("\nEvaluation Metric(s) Obtained:\n")
        f.write(f"Classification Accuracy: {accuracy:.4f}\n")
        f.write("\nConfusion Matrix:\n")
        np.savetxt(f, conf_matrix, fmt='%d')
        f.write("\nClassification Report:\n")
        f.write(class_report)

        f.write(f"\n\nAccuracy before human-in-the-loop intervention: {accuracy:.4f}")
        f.write(f"\n\nAccuracy after human-in-the-loop intervention: {updated_accuracy:.4f}\n\n")
        
        f.write("\nAdditional Insights or Observations:\n")
        f.write("\nWithout Human Intervention:\n")
        for class_label, class_accuracy in class_accuracies:
            f.write("Class {}: Accuracy {:.4f}\n".format(class_label, class_accuracy))
        f.write("\nClass with Maximum Accuracy: Class {}, Accuracy {:.2f}".format(max_accuracy_class[0], max_accuracy_class[1]))
        f.write("\nClass with Minimum Accuracy: Class {}, Accuracy {:.2f}\n".format(min_accuracy_class[0], min_accuracy_class[1]))

        f.write("\nWith Human Intervention:\n")
        for class_label_human, class_accuracy_human in class_accuracies_human:
            f.write("Class {}: Accuracy {:.4f}\n".format(class_label_human, class_accuracy_human))
        f.write("\nClass with Maximum Accuracy: Class {}, Accuracy {:.2f}".format(max_accuracy_class_human[0], max_accuracy_class_human[1]))
        f.write("\nClass with Minimum Accuracy: Class {}, Accuracy {:.2f}\n".format(min_accuracy_class_human[0], min_accuracy_class_human[1]))
        f.write(additional_insights)

        print("Output saved to output.txt")
    
    plt.show(block=True)