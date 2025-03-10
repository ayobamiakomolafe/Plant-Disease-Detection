import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import time
import random

# Load the trained model
def classification(path):
    
    model = tf.keras.models.Sequential([
    # Note the input shape is the desired size of the image 150x150 with 3 bytes color
    # This is the first convolution
    tf.keras.layers.Conv2D(64, (3,3), activation='relu', input_shape=(150, 150, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    # The second convolution
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    # The third convolution
    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    # The fourth convolution
    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    # Flatten the results to feed into a DNN
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dropout(0.5),
    # 512 neuron hidden layer
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(24, activation='softmax')
    ])
    
    LABELS = ['Apple___Apple_scab','Apple___Black_rot','Apple___Cedar_apple_rust','Apple___healthy','Blueberry___healthy',
		  'Cherry___healthy','Cherry___Powdery_mildew','Grape___Black_rot','Grape___Esca_Black_Measles','Grape___healthy',
		  'Grape___Leaf_blight_Isariopsis_Leaf_Spot','Orange___Haunglongbing','Peach___Bacterial_spot','Peach___healthy',
		  'Pepper_bell___Bacterial_spot','Pepper_bell___healthy','Potato___Early_blight','Potato___healthy','Potato___Late_blight',
		  'Raspberry___healthy','Soybean___healthy','Squash___Powdery_mildew','Strawberry___healthy','Strawberry___Leaf_scorch']

    model.load_weights("rps.h5") #load pretrained model
    img = image.load_img(path,target_size=(150,150))  
    x = image.img_to_array(img)
    x = np.expand_dims(x , axis=0)
    images = np.vstack([x])
    classes = model.predict(images , batch_size =10) # predicting images
    result = np.argmax(classes)
    cn = LABELS[result]
    return cn
 



# Streamlit app
st.title("🌿 Plant Disease Detector")
st.write("Upload a leaf image, and I'll identify the disease (if any).")

# File uploader
uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
        # Display uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)

        # Predict activity
        with st.spinner('Predicting...'):
             predicted_class =classification(uploaded_file)
        st.success(f'Plant State: **{predicted_class}**')
        if "healthy" not in predicted_class:
            st.warning("The plant has a disease. Consider consulting an expert or applying suitable treatments.")
        else:
            st.success("The plant appears to be healthy! 🎉")