import streamlit as st
import numpy as np
from keras.models import load_model
from PIL import Image, ImageOps

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Load the model
model = load_model("keras_Model.h5", compile=False)

# Load the labels
with open("labels.txt", "r") as f:
    class_names = f.readlines()

# Streamlit UI
st.title("DHV Graphology Personality Trait Prediction")

# Create the array of the right shape to feed into the Keras model
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

# Image upload or camera capture
uploaded_file = st.file_uploader("Upload Handwriting Image", type=["png", "jpg", "jpeg"])

if st.button("Take Photo"):
    camera_input = st.camera_input("Capture your handwriting")
    if camera_input:
        image = Image.open(camera_input).convert("RGB")
        st.image(image, caption="Captured Image", use_column_width=True)
        
        # Process the image
        size = (224, 224)
        image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
        image_array = np.asarray(image)
        normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
        data[0] = normalized_image_array

        # Make prediction
        prediction = model.predict(data)
        index = np.argmax(prediction)
        class_name = class_names[index].strip()  # Remove newline characters
        confidence_score = prediction[0][index] * 100  # Convert to percentage

        # Display results
        st.write("### Prediction Results:")
        st.write(f"Class: {class_name}")
        st.write(f"Confidence Score: {confidence_score:.2f}%")

# If an image is uploaded
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    # Process the image
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
    image_array = np.asarray(image)
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
    data[0] = normalized_image_array

    # Make prediction
    prediction = model.predict(data)
    index = np.argmax(prediction)
    class_name = class_names[index].strip()  # Remove newline characters
    confidence_score = prediction[0][index] * 100  # Convert to percentage

    # Display results
    st.write("### Prediction Results:")
    st.write(f"Class: {class_name}")
    st.write(f"Confidence Score: {confidence_score:.2f}%")

# Run the Streamlit app