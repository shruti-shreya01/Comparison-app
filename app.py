# import streamlit as st
# import pickle
# import tensorflow as tf
# import os
# from tensorflow.keras.preprocessing.image import img_to_array, load_img
# from PIL import Image
# import numpy as np

# IMAGE_SIZE = 256

# # Class names for the three conditions
# class_names = {0: "Early Blight", 1: "Late Blight", 2: "Healthy"}

# # Function to load and preprocess the uploaded image
# def load_and_preprocess_image(image):
#     img = load_img(image, target_size=(IMAGE_SIZE, IMAGE_SIZE))
#     img_array = img_to_array(img)
#     img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
#     img_array = img_array / 255.0  # Normalize to [0, 1]
#     return img_array

# # Function to predict the class and confidence score
# def predict(model, img_array, model_type):
#     if model_type == "CNN":
#         predictions = model.predict(img_array)
#     else:
#         img_array_flat = img_array.flatten().reshape(1, -1)  # Flatten for KNN/SVM models
#         predictions = model.predict_proba(img_array_flat)  # Use predict_proba for KNN/SVM to get probabilities

#     predicted_class = class_names[np.argmax(predictions[0])]
#     confidence = round(100 * np.max(predictions[0]), 2)  # Convert confidence to percentage and round
#     return predicted_class, confidence

# # Function to load the selected model
# def load_model(model_choice):
#     if model_choice == "KNN":
#         with open("knn_model.pkl", "rb") as f:
#             model = pickle.load(f)
#     elif model_choice == "SVM":
#         with open("svm_model.pkl", "rb") as f:
#             model = pickle.load(f)
#     elif model_choice == "CNN":
#         with open("potato_pickle_final (1).pkl", "rb") as f:
#             data = pickle.load(f)
#             model = tf.keras.models.model_from_json(data["architecture"])
#             model.load_weights(data["weights"])
#     return model

# # Enhanced UI
# st.markdown("<h1 style='text-align: center; color: green;'>🍃 Potato Leaf Health Check 🍃</h1>", unsafe_allow_html=True)
# st.write("Welcome farmers! Upload a photo of your potato leaf and let our AI help you diagnose its health.")

# # File uploader for image input
# uploaded_file = st.file_uploader("Select an image of a potato leaf...", type=["jpg", "jpeg", "png"])

# # Model selection option
# model_choice = st.selectbox("Choose a model for prediction", ["KNN", "SVM", "CNN"])

# # If an image is uploaded and model is selected
# if uploaded_file is not None and model_choice is not None:
#     # Load the selected model
#     model = load_model(model_choice)

#     # Load and preprocess the uploaded image
#     img = Image.open(uploaded_file).convert('RGB')
#     st.image(img, caption="Uploaded Image", use_column_width=True)
    
#     img_array = load_and_preprocess_image(uploaded_file)
    
#     # Predict the class of the leaf disease using the selected model
#     predicted_class, confidence = predict(model, img_array, model_choice)

#     # Display the predicted class and confidence score
#     st.markdown(f"### 🌿 Predicted Disease: **{predicted_class}**")
#     st.write(f"Confidence Score: **{confidence:.2f}%**")

#     # Additional Tips for Farmers
#     st.markdown("#### 🛠 Tips:")
#     if predicted_class == "Early Blight":
#         st.write("⚠️ Early Blight detected. Consider using fungicides and practicing crop rotation.")
#     elif predicted_class == "Late Blight":
#         st.write("⚠️ Late Blight detected. Immediate attention is required, use disease-resistant potato varieties.")
#     elif predicted_class == "Healthy":
#         st.write("✅ Your leaf is healthy! Keep up the good farming practices.")
    
#     # Optional: Display a pie chart with prediction probabilities
#     st.write("### 🔢 Prediction Probabilities:")
#     prob_df = {class_names[i]: float(predictions[0][i]) * 100 for i in range(len(class_names))}
#     st.bar_chart(prob_df)

# # Sidebar enhancements
# st.sidebar.title("About the Disease Classifier")
# st.sidebar.info("This tool uses AI to detect common diseases in potato leaves. It's designed to help farmers identify potential issues early.")

# st.sidebar.subheader("Disease Types")
# st.sidebar.write("🌱 **Early Blight:** A common potato disease caused by a fungus.")
# st.sidebar.write("🌱 **Late Blight:** A serious disease that can devastate potato crops.")
# st.sidebar.write("🌱 **Healthy:** No signs of disease detected.")

# st.sidebar.subheader("How It Works")
# st.sidebar.write("Upload a clear image of your potato leaf, and our AI will predict its health based on trained models.")








import streamlit as st
import pickle
import tensorflow as tf
import numpy as np
import os
from PIL import Image
from tensorflow.keras.preprocessing.image import img_to_array, load_img

IMAGE_SIZE = 256

# Function to load and preprocess the uploaded image
def load_and_preprocess_image(image):
    img = load_img(image, target_size=(IMAGE_SIZE, IMAGE_SIZE))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = img_array / 255.0  # Normalize to [0, 1]
    return img_array

# Function to load pickle model
def load_pickle_model(file_path):
    with open(file_path, "rb") as f:
        model_data = pickle.load(f)
    return model_data

# Function to predict using TensorFlow model
def predict_tf_model(model, img_array):
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction, axis=1)[0]
    confidence = np.max(prediction)
    return predicted_class, confidence

# Function to predict using sklearn models (KNN, SVM)
def predict_sklearn_model(model, img_array):
    img_flatten = img_array.flatten().reshape(1, -1)  # Flatten the image
    prediction = model.predict(img_flatten)
    predicted_class = prediction[0]
    confidence = model.predict_proba(img_flatten)[0].max()  # Confidence score
    return predicted_class, confidence

# Map class indices to names
class_names = {0: "Early Blight", 1: "Late Blight", 2: "Healthy"}

# Model choices
model_options = {
    "CNN Model (potato_pickle_final.pkl)": "potato_pickle_final (1).pkl",
    "SVM Model (svm_model.pkl)": "svm_model.pkl",
    "KNN Model (knn_model.pkl)": "knn_model.pkl"
}

# Streamlit app interface
st.title("Potato Leaf Disease Classification")
st.write("Upload an image of a potato leaf to classify the disease using different models.")

# Select a model
selected_model = st.selectbox("Choose a model to use:", list(model_options.keys()))

# File uploader for image input
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# Check if the user uploaded an image and selected a model
if uploaded_file is not None and selected_model is not None:
    # Load and preprocess the image
    img_array = load_and_preprocess_image(uploaded_file)

    # Load the selected model
    model_path = model_options[selected_model]
    
    if os.path.exists(model_path):
        with open(model_path, "rb") as f:
            if "CNN" in selected_model:
                # For CNN model (TensorFlow/Keras)
                model = pickle.load(f)
                model = tf.keras.models.model_from_json(model["architecture"])
                model.load_weights(model["weights"])
                predicted_class, confidence = predict_tf_model(model, img_array)
            else:
                # For KNN and SVM models (sklearn)
                model = pickle.load(f)
                predicted_class, confidence = predict_sklearn_model(model, img_array)
    else:
        st.error(f"Model file not found at path: {model_path}")
        st.stop()

    # Map predicted class to the disease name
    disease_name = class_names.get(predicted_class, "Unknown")

    # Display the uploaded image
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

    # Display the results
    st.write(f"Predicted Disease: **{disease_name}**")
    st.write(f"Confidence Score: **{confidence:.2f}**")

