def data_processing(train_set,test_set):

    # Processing the train set of the dataset
    x_train = train_set.iloc[:, 1:]
    y_train = train_set.iloc[:, 0]

    # Processing the test set of the dataset
    x_test = test_set.iloc[:, 1:]
    y_test = test_set.iloc[:, 0]

    return x_train,y_train,x_test,y_test