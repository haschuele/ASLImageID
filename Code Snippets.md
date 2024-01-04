# Build CNN Model
```python
import numpy as np
import pandas as pd
import tensorflow as tf
from keras import models
from keras import layers
from matplotlib import pyplot as plt

def cnn_model(input_data = X_train, label_data = Y_train_onehot, 
                   input_val_data = X_val, label_val_data = Y_val_onehot,
                   kernel_size = (3,3), strides = (1,1), pool_size = (2,2), 
                   optimizer = 'adam', learning_rate = 0.001):
    """ 
    Runs CNN Model based on input specifications
    
    Parameters: 
    ----------
    input_data: (N, 64, 64, 3) array
    label_data: (N, 24) one-hot encoded array
    input_val_data: (N, 64, 64, 3) array
    label_val_data: (N, 24) one-hot encoded array
    kernel_size: tuple
    strides: tuple
    pool_size: tuple
    optimizer: 'adam' or 'SGD' 
    learning_rate: float
    """
    
    # Clear session and remove randomness
    tf.keras.backend.clear_session()
    tf.random.set_seed(0)
    
    # Create model
    model = tf.keras.Sequential()
    
    # Add first convolutional layer
    model.add(tf.keras.layers.Conv2D(
        filters = 32, 
        kernel_size = kernel_size, 
        strides = strides,
        activation = 'relu', 
        name='conv_1', 
        input_shape = (64, 64, 3)))
    
    # add first max pooling layer
    model.add(tf.keras.layers.MaxPool2D(
        pool_size = pool_size,
        name = 'pool_1'))
    
    # Add second convolutional layer
    model.add(tf.keras.layers.Conv2D(
        filters = 64, 
        kernel_size = kernel_size, 
        strides = strides, 
        activation = 'relu',
        name = 'conv_2'))
    
    # add second max pooling layer
    model.add(tf.keras.layers.MaxPool2D(
        pool_size = pool_size,
        name = 'pool_2'))
    
    # Add third convolutional layer
    model.add(tf.keras.layers.Conv2D(
        filters = 64, 
        kernel_size = kernel_size,
        activation = 'relu',
        name = 'conv_3'))
    
    # Flatten the feature maps
    model.add(tf.keras.layers.Flatten()) 
    
    # Add fully connected layers
    model.add(tf.keras.layers.Dense(
        units = 64, 
        name='fc_1',
        activation='relu'))
    
    model.add(tf.keras.layers.Dense(
        units = 24, 
        name='fc_2',
        activation='softmax'))
    
    # Determine the optimizer and compile the model
    if optimizer == 'adam':
        model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate = learning_rate),
                      loss = 'categorical_crossentropy',
                      metrics = ['accuracy'])
    elif optimizer == 'SGD':
        model.compile(optimizer = tf.keras.optimizers.SGD(learning_rate = learning_rate),
                      loss = 'categorical_crossentropy',
                      metrics = ['accuracy'])
        
    # Print the model summary
    model.summary()
    
    # Fit the model
    history = model.fit(input_data, label_data,
                        epochs = 5, 
                        batch_size = 32, 
                        validation_data = (input_val_data, label_val_data))
    
    # Set up plots 
    hist = history.history
    x_arr = np.arange(len(hist['loss'])) + 1
    fig = plt.figure(figsize = (12, 4))
    
    # Plot loss 
    ax = fig.add_subplot(1, 2, 1)
    ax.plot(x_arr, hist['loss'], '-o', label = 'Train loss')
    ax.plot(x_arr, hist['val_loss'], '--<', label = 'Validation loss')
    ax.legend(fontsize = 15)
    ax.set_xlabel('Epoch', size = 15)
    ax.set_ylabel('Loss', size = 15)

    # Plot accuracy
    ax = fig.add_subplot(1, 2, 2)
    ax.plot(x_arr, hist['accuracy'], '-o', label = 'Train acc.')
    ax.plot(x_arr, hist['val_accuracy'], '--<', label = 'Validation acc.')
    ax.legend(fontsize = 15)
    ax.set_xlabel('Epoch', size = 15)
    ax.set_ylabel('Accuracy', size = 15)
    
    # Show plots
    plt.show()
    
    # Return the model
    return model
```

# Test Models
```python
model_1 = cnn_model(kernel_size = (5,5), 
          pool_size = (2,2),
          strides = (1,1),
          learning_rate = 0.001,
          optimizer = 'adam')
```
# Best Model Results
![Model Parameters]("/../Best Model Parameters.png"?raw=true)

