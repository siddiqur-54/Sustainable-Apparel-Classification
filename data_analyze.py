import matplotlib.pyplot as plt

def data_distribution(train_set,test_set,x_train):

    # Percentage distribution of training and testing data
    data_sizes = [len(train_set), len(test_set)]
    data_labels = ['Training Data', 'Testing Data']
    
    plt.figure(figsize=(6, 6))
    plt.pie(data_sizes, labels=data_labels, autopct='%1.1f%%', startangle=90, colors=['skyblue', 'lightgreen'])
    plt.title('Data Distribution: Training vs Testing')

    # Distribution of classes in the training dataset
    plt.figure(figsize=(8, 6))
    train_set['label'].value_counts().plot(kind='bar', color='skyblue')
    plt.title('Distribution of Classes in Training Dataset')
    plt.xlabel('Class Labels')
    plt.ylabel('Count')
    plt.xticks(rotation=0)

    # Distribution of classes in the training dataset
    plt.figure(figsize=(8, 6))
    test_set['label'].value_counts().plot(kind='bar', color='skyblue')
    plt.title('Distribution of Classes in Testing Dataset')
    plt.xlabel('Class Labels')
    plt.ylabel('Count')
    plt.xticks(rotation=0)

    # Distribution of pixel values in a sample image
    sample_image_idx = 0
    plt.figure(figsize=(8, 6))
    plt.hist(x_train.iloc[sample_image_idx, :], bins=50, color='skyblue', alpha=0.7)
    plt.title('Distribution of Pixel Values in a Sample Image')
    plt.xlabel('Pixel Value')
    plt.ylabel('Frequency')
    plt.show(block=False)

    # Statistical summary of the features
    print("Statistical Summary of Features:")
    print(x_train.describe())
