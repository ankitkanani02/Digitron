Sure! Here's the complete Markdown README for the provided code:

# MNIST Digit Recognition

This repository contains code for training and evaluating a convolutional neural network (CNN) model for digit recognition using the MNIST dataset. The model is implemented using Keras and TensorFlow.

## Prerequisites

Before running the code, make sure you have the following dependencies installed:

- Keras
- Gradio
- TensorFlow

You can install the dependencies by running the following command:

```shell
pip install keras gradio tensorflow
```

## Dataset

The MNIST dataset consists of 60,000 training images and 10,000 testing images of handwritten digits from 0 to 9. The dataset is automatically downloaded and split into training and testing sets using the `mnist.load_data()` function from Keras.

## Model Architecture

The CNN model architecture used for digit recognition consists of multiple convolutional layers followed by max pooling, dropout, and dense layers. The model summary is as follows:

```python
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(256, kernel_regularizer=l2(0.001), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))
```

## Training

The model is trained using the training set with the following hyperparameters:

- Batch size: 128
- Number of epochs: 20
- Loss function: Categorical cross-entropy
- Optimizer: Adam

The training progress is displayed during the training process, and the model's accuracy is evaluated on the testing set after each epoch.

## Evaluation

After training, the model's performance is evaluated on the testing set. The final test loss and accuracy are printed:

```
Test loss: 0.09064210951328278
Test accuracy: 0.9940000176429749
```

## Saving the Model

The trained model is saved in the file `mnist.h5` using the `model.save()` function.

## Sketchpad Digit Recognition

This repository also includes code for a Gradio interface that allows you to draw a digit on a sketchpad and make predictions using the trained model. To run the interface, execute the following code:

```python
import gradio as gr
import numpy as np
import tensorflow as tf

# Load the trained MNIST model
model = tf.keras.models.load_model('mnist.h5')

# Define the preprocessing and prediction functions

# ...

# Set up the Gradio interface
interface = gr.Interface(fn=sketchpad_digit_recognition, inputs="sketchpad", outputs=output_text, title='Sketchpad Digit Recognition')
interface.launch()
```

The interface will open in a web browser, and you can draw a digit on the sketchpad. The predicted digit and its probability will be displayed.

## License

This project is licensed under the [MIT License](LICENSE).
```

Please note that the code blocks and output are represented using Markdown syntax to maintain the formatting.
