import streamlit as st
from streamlit_drawable_canvas import st_canvas
import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps
import cv2
import matplotlib.pyplot as plt

tf.keras.backend.clear_session()

generator = tf.keras.models.load_model("generator.keras")


generator = tf.keras.models.load_model("generator.keras")



# Load the trained model
model = tf.keras.models.load_model("model.keras")



# Function to preprocess the image
def preprocess_image(image):
    """
    Preprocess the canvas image for the model:
    - Convert to grayscale
    - Center the digit
    - Resize to 28x28
    - Normalize to [0, 1]
    - Add batch and channel dimensions
    """
    # Convert RGBA to grayscale
    image = np.array(image)
    if image.shape[-1] == 4:  # Check for RGBA
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2GRAY)
    elif image.shape[-1] == 3:  # Check for RGB
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # Threshold the image to binary
    _, binary_image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Find bounding box of the digit
    coords = cv2.findNonZero(binary_image)
    if coords is not None:
        x, y, w, h = cv2.boundingRect(coords)
        # Add padding around the digit
        padding = 10
        x_start = max(0, x - padding)
        y_start = max(0, y - padding)
        x_end = min(binary_image.shape[1], x + w + padding)
        y_end = min(binary_image.shape[0], y + h + padding)
        image = binary_image[y_start:y_end, x_start:x_end]

    # Resize to 28x28
    image = cv2.resize(image, (28, 28), interpolation=cv2.INTER_AREA)

    # Normalize to [0, 1]
    image = image / 255.0

    # Add batch and channel dimensions
    image = np.expand_dims(image, axis=(0, -1))

    return image



# Streamlit App
st.title("Real-Time Digit Recognizer")
st.write("Draw a digit in the canvas below, and the prediction will update in real-time.")

# Create a canvas for drawing
canvas_result = st_canvas(
    fill_color="white",  # Background color
    stroke_width=10,  # Pen width
    stroke_color="white",  # Pen color
    background_color="black",  # Canvas background color
    width=280,
    height=280,
    drawing_mode="freedraw",
    key="canvas",
)

# Handle real-time prediction
if canvas_result.image_data is not None:
    # Convert canvas data to an image
    image = canvas_result.image_data.astype("uint8")
    image = Image.fromarray(image)

    # Preprocess the image
    preprocessed_image = preprocess_image(image)

    # Predict using the model
    prediction = model.predict(preprocessed_image).argmax(axis=-1)[0]

    # Display preprocessed image with matplotlib
    st.write("### Preprocessed Image:")
    fig, ax = plt.subplots()
    ax.imshow(preprocessed_image[0, :, :, 0], cmap="gray")  # Grayscale colormap
    ax.axis("off")
    st.pyplot(fig)

    # Display prediction
    st.write(f"### Predicted Digit: {prediction}")

else:
    st.write("### Please draw a digit on the canvas.")

st.title("Real-Time Digit Generator")

st.write("Generator for gerneating digits")

def generate_image():
    noise = tf.random.normal([100, 100])
    predictions = generator(noise, training=False)
    return predictions

if st.button("Generate Image"):
    generated_image = generate_image()
    for i in range(len(generated_image)):
        plt.subplot(10, 10, i + 1)
        plt.imshow(generated_image[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
        plt.axis('off')
    
    st.pyplot(plt)

