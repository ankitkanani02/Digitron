import gradio as gr
import numpy as np
import tensorflow as tf
from PIL import Image

# Load the trained MNIST model
model = tf.keras.models.load_model('mnist.h5')

# Define a function to preprocess the input sketchpad image
def preprocess_image(image):
    image = Image.fromarray(np.uint8(image))
    # Convert the image to grayscale
    image = image.convert('L')
    # Resize the image to 28x28 pixels (MNIST image size)
    image = image.resize((28, 28))
    # Convert the image to a NumPy array
    
    image = np.array(image)
    # Normalize the image pixel values
    image = image / 255.0
    # Reshape the image to match the model's input shape
    image = image.reshape(1, 28, 28, 1)
    return image

# Define a function to make predictions on the preprocessed image
def predict_image(image):
    # Make a prediction using the model
    predictions = model.predict(image)
    # Get the predicted digit and its probability
    digit = np.argmax(predictions)
    probability = predictions[0][digit]
    return digit, probability

# Define the Gradio interface
def sketchpad_digit_recognition(image):
    # Preprocess the image
    preprocessed_image = preprocess_image(image)
    # Make predictions on the preprocessed image
    digit, probability = predict_image(preprocessed_image)
    # Return the predicted digit and its probability
    return digit, probability

# Set up the Gradio interface
output_text = gr.outputs.Textbox()
interface = gr.Interface(fn=sketchpad_digit_recognition, inputs="sketchpad", outputs=output_text, title='Sketchpad Digit Recognition')

# Run the interface
interface.launch()
