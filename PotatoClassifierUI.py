import PotatoClassifier
import streamlit as st
from PIL import Image
import numpy as np

# st.write("# Potato disease classifier")

st.title('Predicting an Image based on 3 different categories')

st.markdown("This is a model based on Tensorflow and Keras for classification of potato plant images.")
st.markdown("Depending on the image you upload, the model will try to categorize it in one of 3 options available.")

img_early_blight = Image.open('assets/early_blight.jpg')
img_late_blight = Image.open('assets/late_blight.jpg')
img_healthy = Image.open('assets/healthy.jpg')

col1, col2, col3 = st.columns(3)

col1.header("Early Blight")
col1.image(img_early_blight)

col2.header("Late Blight")
col2.image(img_late_blight)

col3.header("Healthy")
col3.image(img_healthy)

image_to_classify = st.file_uploader("Please upload your potato leaf image", type=["jpg", "jpeg", "png"])
if image_to_classify:
    potato_image = np.array(Image.open(image_to_classify))  # Read the image as numpy array
    img_predicted = PotatoClassifier.Predict_Image(potato_image)
    st.image(potato_image, caption=f"Your prediction is: {img_predicted}")