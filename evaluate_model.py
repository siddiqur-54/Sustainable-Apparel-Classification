import os
import sys
import pandas as pd

from data_process import data_processing
from data_visualize import data_visualizing
from data_analyze import data_distribution
from model_define import custom_model
from model_execute import execute_model


# Checking if the correct number of command-line arguments is provided
if len(sys.argv) != 2:
    print("Usage: python script_name.py <parent_folder_path>")
    sys.exit(1)

# Getting the path to the parent folder from the command line
parent_folder_path = sys.argv[1]

if not os.path.exists(parent_folder_path):
    print(f"Error: The specified folder '{parent_folder_path}' does not exist.")
    sys.exit(1)

# Defining the file names for train and test datasets
train_file = "fashion-mnist_train.csv"
test_file = "fashion-mnist_test.csv"


# Constructing full paths for train and test datasets
train_dataset_path = os.path.join(parent_folder_path, train_file)
test_dataset_path = os.path.join(parent_folder_path, test_file)

# Loading the train and test datasets
train_set = pd.read_csv(train_dataset_path)
test_set = pd.read_csv(test_dataset_path)

# Processing the dataset
x_train,y_train,x_test,y_test=data_processing(train_set,test_set)

# Visualizing the dataset
data_visualizing(x_train,y_train,x_test,y_test)

# Analyzing the dataset
data_distribution(train_set,test_set,x_train)

# Defining a custom model
model=custom_model()

# Executing the model
execute_model(model,x_train,y_train,x_test,y_test)

# Exiting cleanly after generating the output.txt file
sys.exit(0)