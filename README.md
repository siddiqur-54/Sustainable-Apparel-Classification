## 1. Introduction

This project focuses on the identification and classification of sustainable apparel products using the Fashion MNIST dataset. The goal is to develop an AI solution aligned with NITEX's vision, emphasizing sustainable fashion. NITEX's vision underscores the transformative potential of sustainable fashion, emphasizing the fusion of style and ethics. By harnessing the capabilities of AI, this project aspires to enhance the fashion landscape, drive positive change, and inspire a more sustainable approach to clothing. Through a harmonious blend of technology and environmental consciousness, we embark on a journey towards a future where fashion is not only a statement of style but also a testament to our commitment to the planet.



## 2. Project Structure
- __evaluate_model.py:__ Main script responsible for evaluating the trained machine learning or AI model using the provided dataset folder. It generates an output.txt file containing the model's architecture summary, evaluation metrics, and additional insights or observations.
- __data_process.py:__ Handles preprocessing tasks, including data cleaning, feature scaling, and transformation, ensuring the data is in the appropriate format for training the model.
- __data_visualize.py:__ Focuses on visualizing dataset samples, providing insights into data characteristics through graphs, bars or other visual representations.
- __data_analyze.py:__ Analyzes the dataset in-depth, exploring statistical properties, class distributions, and potential feature correlations, aiding in understanding the dataset's structure.
- __model_define.py:__ Defines the machine learning or AI model architecture, specifying layers, activation functions, and connections between neurons, forming the foundation of the predictive model.
- __model_execute.py:__ Executes the defined model, including the training process and evaluation on the test dataset, providing accuracy metrics and performance evaluation.
- __human_interact.py:__ Allows manual interaction with the project predictions, facilitating human expertise integration to improve accuracy.
- __output.txt:__ Contains the execution results of the model, including model architecture summary, evaluation metrics, and additional insights or observations.
- __requirements.txt:__ Lists all the project dependencies, ensuring consistent environment setup for reproducibility.
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
Here, myvenv is the name of the virtual environment. You can choose any name according to your project.

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
Download the "Fashion MNIST" dataset (https://www.kaggle.com/datasets/zalando-research/fashionmnist). Inside the dataset folder, the training file "fashion-mnist_train.csv" and the testing file "fashion-mnist_test.csv" should exist with the appropriate given name.


## 4. Running the Evaluation Script
```bash
python evaluate_model.py path_to_dataset_folder
```
- __Example__: If the dataset folder is named "Fashion MNIST" which contains "fashion-mnist_train.csv" and "fashion-mnist_test.csv" files, and the path of the folder "Fashion MNIST" is "C:\Users\HP\Documents\AI Task\Fashion MNIST", then the command would be:
```bash
python evaluate_model.py "C:\Users\HP\Documents\AI Task\Fashion MNIST"
```
This will generate an output.txt file with model details and evaluation metrics.
## 5. About Dataset
### 5.1 Context
Fashion-MNIST is a dataset of Zalando's article images, consisting of a training set of 60,000 examples and a test set of 10,000 examples. Each example is a 28x28 grayscale image associated with a label from 10 classes. Zalando intends Fashion-MNIST to serve as a direct drop-in replacement for the original MNIST dataset for benchmarking machine learning algorithms. It shares the same image size and structure of training and testing splits.

### 5.2 Content
Each image is 28 pixels in height and 28 pixels in width, for a total of 784 pixels. Each pixel has a single pixel value associated with it, indicating the lightness or darkness of that pixel, with higher numbers meaning darker. This pixel value is an integer between 0 and 255. The training and test data sets have 785 columns.

- To locate a pixel on the image, suppose that we have decomposed x as x = i * 28 + j, where i and j are integers between 0 and 27. The pixel is located on row i and column j of a 28 x 28 matrix.
- For example, pixel 31 indicates the pixel that is in the fourth column from the left and the second row from the top, as in the ascii diagram below.

### 5.3 Labels
Each training and test example is assigned to one of the following labels:

- 0 T-shirt/top
- 1 Trouser
- 2 Pullover
- 3 Dress
- 4 Coat
- 5 Sandal
- 6 Shirt
- 7 Sneaker
- 8 Bag
- 9 Ankle boot

### 5.4 Acknowledgements
- The dataset has been downloaded from https://www.kaggle.com/datasets/zalando-research/fashionmnist.

## 6. About Model
A Convolutional Neural Network (CNN) has been defined for this project. The Convolutional Neural Network (CNN) is a machine learning model, specifically a type of deep learning model. CNNs are a class of models that have proven to be highly effective for image recognition and classification tasks. CNNs are particularly effective in analyzing visual data due to their ability to capture spatial patterns. This model was chosen for its proven effectiveness in image classification tasks, making it well-suited for sustainable identification and classification of apparel products from the Fashion MNIST dataset.
### 6.1 Model Architecture

The neural network model used for this project is a Convolutional Neural Network (CNN) designed for image classification tasks. Here's a breakdown of the model architecture:
  
  1. __Reshape Layer:__ This layer reshapes the input data into a format suitable for processing by subsequent layers. In this case, it transforms the input into a 3D tensor with dimensions (28, 28, 1), matching the image dimensions of the Fashion MNIST dataset.
  2. __Convolutional Layer 1:__ This layer applies 32 convolutional filters of size (3, 3) to the input data. These filters scan the input images, learning to recognize various features using the ReLU activation function, which introduces non-linearity to the model.
  3. __Batch Normalization:__ After the convolutional layer, batch normalization is applied. It normalizes the activations of the previous layer, improving the stability and speed of training by ensuring that inputs have similar scales.
  4. __MaxPooling:__ This layer performs max pooling with a pool size of (2, 2). Max pooling reduces the spatial dimensions of the data, retaining the most important features while reducing computational complexity.
  5. __Convolutional Layer 2:__ Similar to the first convolutional layer, this layer applies 64 filters of size (3, 3) to the input. It further learns intricate patterns and features from the data.
  6. __Batch Normalization:__ Batch normalization is applied again after the second convolutional layer to maintain the stability of the model during training.
  7. __MaxPooling:__ Another max pooling layer is used to further reduce the spatial dimensions of the data before passing it to the dense layers.
  8. __Flatten Layer:__ This layer flattens the output from the previous layer into a 1D vector. It prepares the data for processing by the densely connected layers.
  9. __Dense Layer:__ A densely connected layer with 128 units and ReLU activation follows the flattened layer. This layer learns complex patterns and representations from the flattened input.
  10. __Dropout Layer:__ Dropout is applied to this layer with a dropout rate of 0.5, meaning during training, 50% of the neurons in this layer will be randomly set to zero. Dropout helps prevent overfitting by introducing noise during training, forcing the model to learn more robust features.
  11. __Output Layer:__ The final layer consists of 10 units, each representing one class in the Fashion MNIST dataset. The softmax activation function is used to convert the raw scores into probabilities, determining the class prediction for the input image.

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
## 8. Figures & Screenshots
### 8.1 Dataset Samples
![App Screenshot](https://github.com/siddiqur-54/Sustainable-Apparel-Classification/blob/master/snapshots/sample%20images.PNG)

### 8.2 Training vs Testing Distribution
![App Screenshot](https://github.com/siddiqur-54/Sustainable-Apparel-Classification/blob/master/snapshots/training%20vs%20testing%20distribution.PNG)

### 8.3 Distribution of Training Set Classes
![App Screenshot](https://github.com/siddiqur-54/Sustainable-Apparel-Classification/blob/master/snapshots/training%20distribution.PNG)

### 8.4 Distribution of Testing Set Classes
![App Screenshot](https://github.com/siddiqur-54/Sustainable-Apparel-Classification/blob/master/snapshots/testing%20distribution.PNG)

### 8.5 Distribution of Sample Image Pixel Values
![App Screenshot](https://github.com/siddiqur-54/Sustainable-Apparel-Classification/blob/master/snapshots/pixel%20distribution.PNG)

### 8.6 Model Architeture
![App Screenshot](https://github.com/siddiqur-54/Sustainable-Apparel-Classification/blob/master/snapshots/model%20architecture.png)

### 8.7 Accuracy and Loss of Training and Validation 
![App Screenshot](https://github.com/siddiqur-54/Sustainable-Apparel-Classification/blob/master/snapshots/accuracy%20and%20loss.PNG)

### 8.8 Confusion Matrix
![App Screenshot](https://github.com/siddiqur-54/Sustainable-Apparel-Classification/blob/master/snapshots/confusion%20matrix.PNG)



## 9. Conclusion
In this project, we tackled the challenge of sustainable apparel classification using the Fashion MNIST dataset, aligning our efforts with NITEX's vision for sustainable fashion. Employing a custom-designed Convolutional Neural Network (CNN), we delved into extensive data preprocessing and exploratory analysis to understand the dataset's nuances. The model, characterized by its intricate layers such as convolutional, pooling, dense, and dropout, excelled at capturing diverse patterns within the images. Leveraging key metrics like accuracy, precision, recall, and F1-score, our model demonstrated impressive performance. What made our approach unique was the integration of human expertise through a human-in-the-loop mechanism. This intervention allowed for manual correction of uncertain predictions, enhancing the model's accuracy and reinforcing the synergy between artificial intelligence and human intuition. The successful collaboration between cutting-edge machine learning techniques and human insights not only met but surpassed our objectives, showcasing the potential for transformative impact in the sustainable fashion landscape.
