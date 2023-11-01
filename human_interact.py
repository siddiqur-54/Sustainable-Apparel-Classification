import numpy as np
from sklearn.metrics import accuracy_score

def human_expertise(predictions,y_test):
    
    # Threshold for uncertainty
    threshold = 0.8

    # Initializing a list of uncertain predictions
    indices = []

    # Identifying uncertain predictions
    for i, prediction in enumerate(predictions):
        max_probability = max(prediction)
        if max_probability < threshold:
            indices.append(i)

    # Correcting uncertain predictions manually
    for index in indices:
        uncertain_prediction = predictions[index]
        # Showing the uncertain prediction to the human expert
        print(f"Uncertain Prediction Details - Index: {index}, Predictions: {uncertain_prediction}, Actual Label: {y_test.iloc[index]}\n")
        
        # Asking human expert for the correction
        #corrected_prediction = input("Enter corrected prediction label: ")
        
        corrected_prediction=y_test.iloc[index]
        predictions[index] = [0] * 10
        predictions[index][int(corrected_prediction)] = 1

    # Evaluating the updated predictions
    updated_accuracy = accuracy_score(y_test, [np.argmax(pred) for pred in predictions])

    return predictions, updated_accuracy