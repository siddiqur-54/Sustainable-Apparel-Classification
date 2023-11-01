import matplotlib.pyplot as plt

def data_visualizing(x_train,y_train,x_test,y_test):

    #Displaying sample pixel values from the dataset
    print(x_train.head(5))
    print(y_train.head())

    print(x_test.head(5))
    print(y_test.head())

    # Displaying sample images from the training dataset
    labels = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

    plt.figure(figsize=(10, 10))
    for i in range(25):
        plt.subplot(5, 5, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(x_train.iloc[i, :].values.reshape(28, 28), cmap=plt.cm.binary)
        plt.xlabel(labels[y_train.iloc[i]])