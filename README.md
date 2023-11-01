## 1. Overview

This project focuses on the identification and classification of sustainable apparel products using the Fashion MNIST dataset. The goal is to develop an AI solution aligned with NITEX's vision, emphasizing sustainable fashion.



## 2. Project Structure
- "__evaluate_model.py__": This is the main file of the Project.
- "__data_process.py__": This is for the data preprocessing.
- "__data_visualize.py__": This is for the visualization of the dataset sample.
- "__data_analyze__.py": This is for analyzing the dataset.
- "__model_define.py__": This is for creating a custom model.
- "__model_execute.py__": This is for executing the model.
- "__human_interact.py__": This is for interacting with the project prediction manually.
- "__output.txt__": This file contains the exectuion results of the model
- "__requirements.txt__": This file contains all the dependencies of the project
## 3. Getting Started

### I. Clone the repository to your local machine:
```bash
git clone <repository_url>
cd Sustainable-Apparel-Classification
```
### II. Create a virtual environment (optional but recommended):
```bash
python -m venv myvenv
```
Here myvenv is the name of the  virtual environement. You can choose any name according to your project.

### III. Activate the virtual environment:
__For windows:__
```bash
myvenv\Scripts\activate
```
__For Linux/macOS:__
```bash
source myvenv/bin/activate
```
### IV. Install project dependencies:
```bash
pip install -r requirements.txt
```
### V. Download the dataset
Download the "Fashion MNIST" dataset (https://www.kaggle.com/datasets/zalando-research/fashionmnist). Inside the dataset folder the training file "fashion-mnist_train.csv" and the testing file "fashion-mnist_test.csv" should be existed with appropriate given name.


## 4. Running the Evaluation Script
```bash
python evaluate_model.py path_to_dataset_folder
```
- __Example__: If the dataset folder is named "Fashion MNIST" which contains "fashion-mnist_train.csv" and "fashion-mnist_test.csv" files and the path of the folder "Fashion MNIST" is "C:\Users\HP\Documents\AI Task\Fashion MNIST", then the command would be:
```bash
python evaluate_model.py "C:\Users\HP\Documents\AI Task\Fashion MNIST"
```
This will generate an output.txt file with model details and evaluation metrics.
## 5. About Dataset
### 5.1 Context
Fashion-MNIST is a dataset of Zalando's article images—consisting of a training set of 60,000 examples and a test set of 10,000 examples. Each example is a 28x28 grayscale image, associated with a label from 10 classes.

### 5.2 Content
Each image is 28 pixels in height and 28 pixels in width, for a total of 784 pixels in total. Each pixel has a single pixel-value associated with it, indicating the lightness or darkness of that pixel, with higher numbers meaning darker. This pixel-value is an integer between 0 and 255. The training and test data sets have 785 columns.

- To locate a pixel on the image, suppose that we have decomposed x as x = i * 28 + j, where i and j are integers between 0 and 27. The pixel is located on row i and column j of a 28 x 28 matrix.
- For example, pixel31 indicates the pixel that is in the fourth column from the left, and the second row from the top, as in the ascii-diagram below.

### 5.3 Labels
Each training and test example is assigned to one of the following labels:

- 0 T-shirt/top
- 1 Trouser
- 2    Pullover
- 3    Dress
- 4    Coat
- 5    Sandal
- 6    Shirt
- 7    Sneaker
- 8    Bag
- 9    Ankle boot

### 5.4 Acknowledgements
- Original dataset was downloaded from https://github.com/zalandoresearch/fashion-mnist
## 6. About Model
### 6.1 Model Architecture

The neural network model used for this project is a Convolutional Neural Network (CNN) designed for image classification tasks. Here's a breakdown of the model architecture:
  
  1. __Reshape Layer:__ Reshapes the input into (28, 28, 1) to match the image dimensions.
  2. __Convolutional Layer 1:__ 32 filters of size (3, 3) with ReLU activation.
  3. __Batch Normalization:__ Normalizes the activations of the previous layer.
  4. __MaxPooling:__ Pooling layer with pool size (2, 2) to reduce spatial dimensions.
  5. __Convolutional Layer 2:__ 64 filters of size (3, 3) with ReLU activation.
  6. __Batch Normalization:__ Normalizes the activations of the previous layer.
  7. __MaxPooling:__ Pooling layer with pool size (2, 2) to reduce spatial dimensions.
  8. __Flatten Layer:__ Flattens the output from the previous layer.
  9. __Dense Layer:__ 128 units with ReLU activation.
  10. __Dropout Layer:__ Dropout layer with a dropout rate of 0.5 to prevent overfitting.
  11. __Output Layer:__ 10 units with softmax activation, representing the 10 classes in Fashion MNIST dataset.

This architecture allows the model to learn hierarchical features from the input images and make predictions for the respective apparel classes.

###  6.2 Model Architecture Summary

| Layer (type) | Output Shape | Param # |
| :----------  | :----------- | :------ |
| `reshape (Reshape)` | `(None, 28, 28, 1)` | `0` |
| `conv2d (Conv2D)` | `(None, 26, 26, 32)` | `320` |
| `batch_normalization (Batch Normalization)` | `(None, 26, 26, 32)`| `128` |
| `max_pooling2d (MaxPooling2D` | `(None, 13, 13, 32)`| `0`|
| `conv2d_1 (Conv2D)` | `(None, 11, 11, 64)` | `18496` |
| `batch_normalization_1 (Batch Normalization)` | `(None, 11, 11, 64)` | `256`|
| `max_pooling2d_1 (MaxPooling2D)` | `(None, 5, 5, 64)` | `0` |
| `flatten (Flatten)` | `(None, 1600)` | `0` |
| `dense (Dense)` | `(None, 128)` | `204928` |
| `dropout (Dropout)` | `(None, 128)` | `0` |
| `dense_1 (Dense)` | `(None, 10)` | `1290` |

- __Total params:__ 225418 (880.54 KB)
- __Trainable params:__ 225226 (879.79 KB)
- __Non-trainable params:__ 192 (768.00 Byte)



## 7. Evaluation Metrics
I. __Accuracy:__ Accuracy measures how accurately a model classifies examples across all classes. By taking into account both genuine positives and true negatives, it provides an extensive overview of the model’s performance. When working with datasets where one class occurs over the others, it might not be the ideal option. It is calculated as the sum of true positives (correctly predicted positive instances) and true negatives (correctly predicted negative instances) divided by the total number of instances.

II. __Precision:__ Precision highlights the accuracy of successful predictions. It shows how many of the events that the model predicted as positive were actually true positives. High precision means the model is cautious when labeling events as positive, lowering false positives. In situations where false positives are expensive or deceptive, it is useful. It is calculated as the number of true positives divided by the sum of true positives and false positives (instances wrongly predicted as positive).

III. __Recall:__ Recall, often referred to as sensitivity, measures how well the model is able to identify every instance of success. It demonstrates how successfully the model distinguishes true positives from all other positive situations. In situations where false negatives are a concern, a high recall indicates that the model is sensitive to not missing positive cases, which is significant. It is calculated as the number of true positives divided by the sum of true positives and false negatives (instances wrongly predicted as negative).


IV. __F1-Score__: The F1 score serves as a harmonious equilibrium between the aspects of recall and precision. It becomes particularly beneficial when the need arises to mitigate both false positives and false negatives. Functioning as a comprehensive indicator of a model’s classification performance, the F1 score considers the interplay between precision and recall. Calculated as the harmonic mean of these two metrics, the F1 score encapsulates the overall performance of the model in classification tasks.
