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
# # def load_and_preprocess_image(image):
# #     img = load_img(image, target_size=(IMAGE_SIZE, IMAGE_SIZE))
# #     img_array = img_to_array(img)
# #     img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
# #     img_array = img_array / 255.0  # Normalize to [0, 1]
# #     return img_array

# def load_and_preprocess_image(image, model_type):
#     img = load_img(image, target_size=(IMAGE_SIZE, IMAGE_SIZE))
#     img_array = img_to_array(img)
#     img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
#     img_array = img_array / 255.0  # Normalize to [0, 1]
    
#     if model_type != "CNN":  # Flatten for KNN/SVM models
#         img_array = img_array.flatten().reshape(1, -1)  # Reshape to 1 sample with flattened image

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
    
#     # img_array = load_and_preprocess_image(uploaded_file)
#     # Load and preprocess the uploaded image
#     img_array = load_and_preprocess_image(uploaded_file, model_choice)

    
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
#     # st.write("### 🔢 Prediction Probabilities:")
#     # prob_df = {class_names[i]: float(predictions[0][i]) * 100 for i in range(len(class_names))}
#     # st.bar_chart(prob_df)

# # Sidebar enhancements
# st.sidebar.title("About the Disease Classifier")
# st.sidebar.info("This tool uses AI to detect common diseases in potato leaves. It's designed to help farmers identify potential issues early.")

# st.sidebar.subheader("Disease Types")
# st.sidebar.write("🌱 **Early Blight:** A common potato disease caused by a fungus.")
# st.sidebar.write("🌱 **Late Blight:** A serious disease that can devastate potato crops.")
# st.sidebar.write("🌱 **Healthy:** No signs of disease detected.")

# st.sidebar.subheader("How It Works")
# st.sidebar.write("Upload a clear image of your potato leaf, and our AI will predict its health based on trained models.")







#knn

# import streamlit as st
# import pickle
# import tensorflow as tf
# from tensorflow.keras.preprocessing.image import img_to_array, load_img
# from PIL import Image
# import numpy as np

# # Set the IMAGE_SIZE based on your model's training
# IMAGE_SIZE = 90  # 90x90 for a total of 8100 features

# # Class names for the three conditions
# class_names = {0: "Early Blight", 1: "Late Blight", 2: "Healthy"}



# # Function to load and preprocess the uploaded image
# def load_and_preprocess_image(image, model_type):
#     img = load_img(image, target_size=(IMAGE_SIZE, IMAGE_SIZE))  # Resize to match scaler's expected input size
#     img_array = img_to_array(img)
#     img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
#     img_array = img_array / 255.0  # Normalize to [0, 1]
    
#     if model_type != "CNN":  # Flatten for KNN/SVM models
#         img_array = img_array.flatten().reshape(1, -1)  # Reshape to 1 sample with flattened image
#         if model_type in ["KNN", "SVM"]:
#             # Ensure correct number of features for KNN/SVM
#             img_array = img_array[:, :8100]  # Keep only the first 8100 features if needed

#     return img_array


# # Function to predict the class and confidence score
# def predict(model, img_array, model_type):
#     if model_type == "CNN":
#         predictions = model.predict(img_array)
#     else:
#         predictions = model.predict_proba(img_array)  # Use predict_proba for KNN/SVM to get probabilities

#     predicted_class = class_names[np.argmax(predictions[0])]
#     confidence = round(100 * np.max(predictions[0]), 2)  # Convert confidence to percentage and round
#     return predicted_class, confidence

# # Function to load the selected model
# def load_model(model_choice):
#     if model_choice == "KNN":
#         with open("knn_model.pkl", "rb") as f:
#             model = pickle.load(f)
#         with open("knn_scaler.pkl", "rb") as f:  # Load KNN scaler
#             scaler = pickle.load(f)
#     elif model_choice == "SVM":
#         with open("svm_model.pkl", "rb") as f:
#             model = pickle.load(f)
#         with open("svm_scaler.pkl", "rb") as f:  # Load SVM scaler
#             scaler = pickle.load(f)
#     elif model_choice == "CNN":
#         with open("potato_pickle_final (1).pkl", "rb") as f:
#             data = pickle.load(f)
#             model = tf.keras.models.model_from_json(data["architecture"])
#             model.load_weights(data["weights"])
#         scaler = None  # No scaler for CNN
#     return model, scaler

# # Enhanced UI
# st.markdown("<h1 style='text-align: center; color: green;'>🍃 Potato Leaf Health Check 🍃</h1>", unsafe_allow_html=True)
# st.write("Welcome farmers! Upload a photo of your potato leaf and let our AI help you diagnose its health.")

# # File uploader for image input
# uploaded_file = st.file_uploader("Select an image of a potato leaf...", type=["jpg", "jpeg", "png"])

# # Model selection option
# model_choice = st.selectbox("Choose a model for prediction", ["KNN", "SVM", "CNN"])

# # If an image is uploaded and model is selected
# if uploaded_file is not None and model_choice is not None:
#     # Load the selected model and scaler
#     model, scaler = load_model(model_choice)

#     # Load and preprocess the uploaded image
#     img = Image.open(uploaded_file).convert('RGB')
#     st.image(img, caption="Uploaded Image", use_column_width=True)
    
#     # Load and preprocess the uploaded image
#     img_array = load_and_preprocess_image(uploaded_file, model_choice)

#     # Scale the image array for KNN/SVM models
#     if model_choice in ["KNN", "SVM"]:
#         img_array = scaler.transform(img_array)  # Scale using the loaded scaler

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

# # Sidebar enhancements
# st.sidebar.title("About the Disease Classifier")
# st.sidebar.info("This tool uses AI to detect common diseases in potato leaves. It's designed to help farmers identify potential issues early.")

# st.sidebar.subheader("Disease Types")
# st.sidebar.write("🌱 **Early Blight:** A common potato disease caused by a fungus.")
# st.sidebar.write("🌱 **Late Blight:** A serious disease that can devastate potato crops.")
# st.sidebar.write("🌱 **Healthy:** No signs of disease detected.")

# st.sidebar.subheader("How It Works")
# st.sidebar.write("Upload a clear image of your potato leaf, and our AI will predict its health based on trained models.")





# knn and cnn

# import streamlit as st
# import pickle
# import tensorflow as tf
# import numpy as np
# from tensorflow.keras.preprocessing.image import img_to_array, load_img
# from PIL import Image

# # Set the IMAGE_SIZE based on your model's training
# IMAGE_SIZE_CNN = 256  # Size for CNN model
# IMAGE_SIZE_OTHER = 90  # Size for KNN and SVM models

# # Class names for the three conditions
# class_names = {0: "Early Blight", 1: "Late Blight", 2: "Healthy"}

# # Function to load and preprocess the uploaded image
# def load_and_preprocess_image(image, model_type):
#     if model_type == "CNN":
#         img = load_img(image, target_size=(IMAGE_SIZE_CNN, IMAGE_SIZE_CNN))  # Resize for CNN
#     else:
#         img = load_img(image, target_size=(IMAGE_SIZE_OTHER, IMAGE_SIZE_OTHER))  # Resize for KNN/SVM
#     img_array = img_to_array(img)
#     img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
#     img_array = img_array / 255.0  # Normalize to [0, 1]
    
#     if model_type != "CNN":  # Flatten for KNN/SVM models
#         img_array = img_array.flatten().reshape(1, -1)  # Reshape to 1 sample with flattened image
#         img_array = img_array[:, :8100]  # Keep only the first 8100 features if needed

#     return img_array

# # Function to predict the class and confidence score
# def predict(model, img_array, model_type):
#     if model_type == "CNN":
#         predictions = model.predict(img_array)
#     else:
#         predictions = model.predict_proba(img_array)  # Use predict_proba for KNN/SVM to get probabilities

#     predicted_class = class_names[np.argmax(predictions[0])]
#     confidence = round(100 * np.max(predictions[0]), 2)  # Convert confidence to percentage and round
#     return predicted_class, confidence

# # Function to load the selected model
# def load_model(model_choice):
#     if model_choice == "KNN":
#         with open("knn_model.pkl", "rb") as f:
#             model = pickle.load(f)
#         with open("knn_scaler.pkl", "rb") as f:  # Load KNN scaler
#             scaler = pickle.load(f)
#     elif model_choice == "SVM":
#         with open("svm_model.pkl", "rb") as f:
#             model = pickle.load(f)
#         with open("svm_scaler.pkl", "rb") as f:  # Load SVM scaler
#             scaler = pickle.load(f)
#     elif model_choice == "CNN":
#         with open("potato_pickle_final (1).pkl", "rb") as f:
#             data = pickle.load(f)
#             model = tf.keras.models.model_from_json(data["architecture"])
#             model.load_weights(data["weights"])
#         scaler = None  # No scaler for CNN
#     return model, scaler

# # Enhanced UI
# st.markdown("<h1 style='text-align: center; color: green;'>🍃 Potato Leaf Health Check 🍃</h1>", unsafe_allow_html=True)
# st.write("Welcome farmers! Upload a photo of your potato leaf and let our AI help you diagnose its health.")

# # File uploader for image input
# uploaded_file = st.file_uploader("Select an image of a potato leaf...", type=["jpg", "jpeg", "png"])

# # Model selection option
# model_choice = st.selectbox("Choose a model for prediction", ["KNN", "SVM", "CNN"])

# # If an image is uploaded and model is selected
# if uploaded_file is not None and model_choice is not None:
#     # Load the selected model and scaler
#     model, scaler = load_model(model_choice)

#     # Load and preprocess the uploaded image
#     img = Image.open(uploaded_file).convert('RGB')
#     st.image(img, caption="Uploaded Image", use_column_width=True)
    
#     img_array = load_and_preprocess_image(uploaded_file, model_choice)

#     # Scale the image array for KNN/SVM models
#     if model_choice in ["KNN", "SVM"]:
#         img_array = scaler.transform(img_array)  # Scale using the loaded scaler

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
#         st.write("⚠️ Late Blight detected. Immediate attention is required; use disease-resistant potato varieties.")
#     elif predicted_class == "Healthy":
#         st.write("✅ Your leaf is healthy! Keep up the good farming practices.")

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
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from PIL import Image
from skimage.transform import resize

# Set the IMAGE_SIZE based on your model's training
IMAGE_SIZE_CNN = 256  # Size for CNN model
IMAGE_SIZE_OTHER = 90  # Size for KNN and SVM models

# Class names for the three conditions
class_names = {0: "Early Blight", 1: "Late Blight", 2: "Healthy"}

# Function to load and preprocess the uploaded image
def load_and_preprocess_image(image, model_type, expected_features):
    if model_type == "CNN":
        img = load_img(image, target_size=(IMAGE_SIZE_CNN, IMAGE_SIZE_CNN))
        img_array = img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0
    else:
        img = Image.open(image).convert('RGB')
        img_array = np.array(img)
        img_array = resize(img_array, (IMAGE_SIZE_OTHER, IMAGE_SIZE_OTHER, 3))
        img_array = img_array.flatten().reshape(1, -1)
        
        # Adjust the number of features based on the model's expectation
        if img_array.shape[1] > expected_features:
            img_array = img_array[:, :expected_features]
        elif img_array.shape[1] < expected_features:
            img_array = np.pad(img_array, ((0, 0), (0, expected_features - img_array.shape[1])), 'constant')

    return img_array

# Function to predict the class and confidence score
def predict(model, img_array, model_type):
    if model_type == "CNN":
        predictions = model.predict(img_array)
    else:
        predictions = model.predict_proba(img_array)

    predicted_class = class_names[np.argmax(predictions[0])]
    confidence = round(100 * np.max(predictions[0]), 2)
    return predicted_class, confidence

# Function to load the selected model and get expected features
def load_model(model_choice):
    if model_choice == "KNN":
        with open("knn_model.pkl", "rb") as f:
            model = pickle.load(f)
        with open("knn_scaler.pkl", "rb") as f:
            scaler = pickle.load(f)
        expected_features = model.n_features_in_
    elif model_choice == "SVM":
        with open("svm_model.pkl", "rb") as f:
            model = pickle.load(f)
        with open("svm_scaler.pkl", "rb") as f:
            scaler = pickle.load(f)
        expected_features = model.n_features_in_
    elif model_choice == "CNN":
        with open("potato_pickle_final (1).pkl", "rb") as f:
            data = pickle.load(f)
            model = tf.keras.models.model_from_json(data["architecture"])
            model.load_weights(data["weights"])
        scaler = None
        expected_features = None  # Not applicable for CNN
    return model, scaler, expected_features

# UI Setup
st.markdown("<h1 style='text-align: center; color: green;'>🍃 Potato Leaf Health Check 🍃</h1>", unsafe_allow_html=True)
st.write("Welcome farmers! Upload a photo of your potato leaf and let our AI help you diagnose its health.")

uploaded_file = st.file_uploader("Select an image of a potato leaf...", type=["jpg", "jpeg", "png"])
model_choice = st.selectbox("Choose a model for prediction", ["KNN", "SVM", "CNN"])

if uploaded_file is not None and model_choice is not None:
    model, scaler, expected_features = load_model(model_choice)

    img = Image.open(uploaded_file).convert('RGB')
    st.image(img, caption="Uploaded Image", use_column_width=True)
    
    img_array = load_and_preprocess_image(uploaded_file, model_choice, expected_features)

    if model_choice in ["KNN", "SVM"]:
        img_array = scaler.transform(img_array)

    predicted_class, confidence = predict(model, img_array, model_choice)

    st.markdown(f"### 🌿 Predicted Disease: **{predicted_class}**")
    st.write(f"Confidence Score: **{confidence:.2f}%**")

    st.markdown("#### 🛠 Tips:")
    if predicted_class == "Early Blight":
        st.write("⚠️ Early Blight detected. Consider using fungicides and practicing crop rotation.")
    elif predicted_class == "Late Blight":
        st.write("⚠️ Late Blight detected. Immediate attention is required; use disease-resistant potato varieties.")
    elif predicted_class == "Healthy":
        st.write("✅ Your leaf is healthy! Keep up the good farming practices.")

# Sidebar information
st.sidebar.title("About the Disease Classifier")
st.sidebar.info("This tool uses AI to detect common diseases in potato leaves. It's designed to help farmers identify potential issues early.")

st.sidebar.subheader("Disease Types")
st.sidebar.write("🌱 **Early Blight:** A common potato disease caused by a fungus.")
st.sidebar.write("🌱 **Late Blight:** A serious disease that can devastate potato crops.")
st.sidebar.write("🌱 **Healthy:** No signs of disease detected.")

st.sidebar.subheader("How It Works")
st.sidebar.write("Upload a clear image of your potato leaf, and our AI will predict its health based on trained models.")
