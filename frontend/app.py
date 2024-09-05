import streamlit as st
import requests
from PIL import Image
import io

# Streamlit interface
st.title("Adversarial Attack Demo")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png"])

if uploaded_file is not None:
    # Display the original image
    image = Image.open(uploaded_file)
    st.image(image, caption="Original Image", use_column_width=True)

    # Choose attack type
    attack_type = st.selectbox("Select Attack Type", ["fgsm"])

    # Choose epsilon value for FGSM
    epsilon = st.slider("Select epsilon value", 0.0, 0.3, 0.1)

    # Process the image
    if st.button("Run Attack"):
        # Convert image to bytes
        img_buffer = io.BytesIO()
        image.save(img_buffer, format='PNG')
        img_buffer.seek(0)
        
        # Send image and parameters to backend
        files = {"image": img_buffer.getvalue()}
        data = {"attack_type": attack_type, "epsilon": epsilon}
        
        response = requests.post("http://localhost:8000/attack", files=files, data=data)

        if response.status_code == 200:
            # Convert the response content to an image
            adversarial_image = Image.open(io.BytesIO(response.content))
            st.image(adversarial_image, caption="Adversarial Image (FGSM)", use_column_width=True)
        else:
            st.error("Error processing the image")
