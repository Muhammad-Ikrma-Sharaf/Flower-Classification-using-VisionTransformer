import streamlit as st
from PIL import Image
import torch
from torchvision.transforms import ToTensor
import torchvision.transforms.functional as TF

# Load the saved model
model = torch.load('pretrained_vit_model.h5')
model.eval()

# Define the label classes
label_classes = ['daisy', 'lavender']

# Streamlit app
st.title("Image Classification with Pretrained ViT")
st.write("Please upload an image to classify.")

# Image uploader
uploaded_file = st.file_uploader("Choose an image...", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Preprocess the image
    image = ToTensor()(image)
    image = TF.resize(image, (224, 224))
    image = image.unsqueeze(0)

    # Perform inference
    with torch.no_grad():
        output = model(image)
        predicted_class = label_classes[torch.argmax(output)]

    # Display the predicted label
    st.write(f"Predicted Label: {predicted_class}")
