
# MNIST Digit Recognition and Generation Tool

This project combines the power of deep learning with an intuitive user interface to showcase digit recognition and generation capabilities. It features:
- A **real-time digit classifier** where users can draw digits on a canvas for instant recognition.
- A **digit generator** powered by a custom-designed GAN (Generative Adversarial Network) that creates realistic MNIST-style digits.

---

## Features

### **1. Real-Time Digit Recognizer**
- An interactive interface where users can draw digits directly on a canvas.
- The drawn digit is processed and recognized using a **Convolutional Neural Network (CNN)** trained on the MNIST dataset.
- Real-time feedback on predictions, including a preprocessed view of the input image.

### **2. GAN-Based Digit Generator**
- A custom-built **Generative Adversarial Network (GAN)** that generates realistic MNIST-style digits.
- Users can generate batches of random digits by clicking a button.
- Digit images are displayed in a grid layout for easy viewing.

---

## Technologies Used

### **Deep Learning**
- **TensorFlow/Keras** for building and training the CNN classifier and GAN.
- CNN for accurate digit classification with preprocessing to handle variations in user inputs.
- GAN with custom generator and discriminator models designed from scratch.

### **Frontend Interface**
- **Streamlit** for creating an intuitive, interactive user interface.
- **streamlit-drawable-canvas** for enabling real-time drawing and input.

### **Data and Preprocessing**
- **MNIST Dataset** for training and validation.
- Data augmentation (flipping, normalization) to improve robustness.

---

## Installation

### **Prerequisites**
- Python 3.8 or higher
- Install dependencies:
  ```bash
  pip install tensorflow tensorflow-datasets streamlit streamlit-drawable-canvas matplotlib opencv-python pillow
  ```

### **Running Locally**
1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo/mnist-project.git
   cd mnist-project
   ```
2. Launch the Streamlit app:
   ```bash
   streamlit run streamlit.py
   ```
3. Access the app at `http://localhost:8501`.

---

## Usage

### **Digit Recognizer**
1. Open the app and navigate to the **Digit Recognizer** section.
2. Draw a digit (0-9) on the canvas.
3. The app will preprocess your input and display:
   - The preprocessed grayscale version of your digit.
   - The predicted digit with confidence scores.

### **Digit Generator**
1. Navigate to the **Digit Generator** section.
2. Click the **Generate Image** button to create new digits.
3. View the generated digits in a 10x10 grid.

---

## Model Training

### **Classifier**
- A CNN model with:
  - Three convolutional layers for feature extraction.
  - Dropout layers for regularization.
  - Dense layers for classification into 10 classes (digits 0-9).
- Trained for 10 epochs on the MNIST dataset, achieving **98% accuracy** on validation.

### **GAN**
- A custom-designed generator and discriminator:
  - **Generator**: Creates 28x28 grayscale images resembling digits.
  - **Discriminator**: Distinguishes between real and generated images.
- Trained using adversarial loss on the MNIST dataset.

---

## Key Highlights
- **Interactive and Visual**: Combines deep learning with a user-friendly interface.
- **Educational**: Demonstrates the practical use of CNNs and GANs.
- **Custom Models**: GAN and classifier designed and implemented from scratch.

---

# Sample Photos produce by GAN. 
![alt text](https://github.com/Nimanoro/number-detector/blob/main/GAN-examples.png)

---

## License
This project is licensed under the MIT License. See `LICENSE` for details.

---

## Contact
For inquiries or feature requests, contact [Nima Norouzi](mailto:your-email@example.com).
