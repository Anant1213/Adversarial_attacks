import streamlit as st
from PIL import Image
import tensorflow as tf
import numpy as np
import io

# FGSM attack implementation
def fgsm_attack(model, image, epsilon):
    image = tf.convert_to_tensor(image, dtype=tf.float32)
    image = tf.expand_dims(image, axis=0)  # Add batch dimension
    
    with tf.GradientTape() as tape:
        tape.watch(image)
        prediction = model(image)
        # Get the class with the highest prediction probability
        target_class = tf.argmax(prediction[0])
        target_class = tf.expand_dims(target_class, axis=0)  # Ensure shape matches
        
        # Compute the loss for that class
        loss = tf.keras.losses.sparse_categorical_crossentropy(target_class, prediction)
    
    # Compute the gradient of the loss w.r.t the image
    gradient = tape.gradient(loss, image)
    perturbation = epsilon * tf.sign(gradient)
    adversarial_image = image + perturbation
    
    # Clip the values to ensure they are within the valid range [0, 1]
    adversarial_image = tf.clip_by_value(adversarial_image, 0.0, 1.0)
    
    return tf.squeeze(adversarial_image).numpy()

# Function to convert numpy array to downloadable image
def convert_array_to_image(adversarial_array):
    adversarial_image = (adversarial_array * 255).astype(np.uint8)
    pil_image = Image.fromarray(adversarial_image)
    return pil_image

# Load model (replace with your model or use a pre-trained one)
model = tf.keras.applications.MobileNetV2(weights="imagenet")

# Streamlit interface
st.title("FGSM Adversarial Attack Demo")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png"])

if uploaded_file is not None:
    # Open the image and display it
    image = Image.open(uploaded_file)
    st.image(image, caption="Original Image", use_column_width=True)
    
    # Convert image to array
    img_array = np.array(image.resize((224, 224))) / 255.0  # Rescale to [0,1]
    
    # Get epsilon value from user
    epsilon = st.slider("Select epsilon value", 0.0, 0.3, 0.1)
    
    # Perform FGSM attack
    adversarial_array = fgsm_attack(model, img_array, epsilon)
    
    # Convert numpy array to image
    adversarial_image = convert_array_to_image(adversarial_array)
    
    # Display adversarial image
    st.image(adversarial_image, caption="Adversarial Image (FGSM)", use_column_width=True)
    
    # Save adversarial image to an in-memory file
    img_buffer = io.BytesIO()
    adversarial_image.save(img_buffer, format="PNG")
    img_buffer.seek(0)
    
    # Create download button
    st.download_button(
        label="Download Adversarial Image",
        data=img_buffer,
        file_name="adversarial_image.png",
        mime="image/png"
    )
