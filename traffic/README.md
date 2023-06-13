In the first trial, a convolutional neural network (CNN) model was trained with 32 filters in the convolutional layer, a pooling size of (2,2), and 128 neurons in the hidden layer with a dropout rate of 0.5. However, the accuracy achieved was only 0.0565, indicating that the model performed poorly.

In the second trial, the CNN model was modified with 50 filters in the convolutional layer, a pooling size of (1,1), and 200 neurons in the hidden layer with a dropout rate of 0.5. Unfortunately, the accuracy obtained was even lower at 0.0553, suggesting that these changes did not improve the model's performance.

In the third trial, the CNN model was further adjusted with 50 filters in the convolutional layer, a pooling size of (1,1), and 200 neurons in the hidden layer, but without any dropout regularization. Surprisingly, the accuracy increased significantly to 0.9633. However, this high accuracy might be an indication of overfitting, where the model has learned the training data too well and struggles to generalize to unseen examples.

NOTE: The gtsrb (German Traffic Sign Recognition Benchmark) data set is not included in this folder, due to its enormous size
