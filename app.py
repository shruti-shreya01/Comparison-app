import streamlit as st
import pickle
import tensorflow as tf
import os
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from PIL import Image, ImageOps
import numpy as np

IMAGE_SIZE = 256

# Function to load and preprocess the uploaded image
def load_and_preprocess_image(image):
    img = load_img(image, target_size=(IMAGE_SIZE, IMAGE_SIZE))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = img_array / 255.0  # Normalize to [0, 1]
    return img_array

# Path to the pickle file
file_path = "potato_pickle_final (1).pkl"

# Check if file exists and load the model
if os.path.exists(file_path):
    with open(file_path, "rb") as f:
        data = pickle.load(f)

    # Reconstruct the model from the architecture
    model = tf.keras.models.model_from_json(data["architecture"])
else:
    st.error(f"Model file not found at path: {file_path}")
    st.stop()

class_names = {0: "Early Blight", 1: "Late Blight", 2: "Healthy"}

# Initialize session state variables
if "prediction" not in st.session_state:
    st.session_state["prediction"] = None
if "confidence" not in st.session_state:
    st.session_state["confidence"] = None

# Enhanced UI
st.markdown("<h1 style='text-align: center; color: green;'>🍃 Potato Leaf Health Check 🍃</h1>", unsafe_allow_html=True)
st.write("Welcome farmers! Upload a photo of your potato leaf and let our AI help you diagnose its health.")

# File uploader for image input
uploaded_file = st.file_uploader("Select an image of a potato leaf...", type=["jpg", "jpeg", "png"], key="uploaded_file")
def predict(model, img):
    # Preprocess the image to be compatible with the model
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)  # Add batch dimension
    
    # Predict the class probabilities
    predictions = model.predict(img_array)
    
    # Get the predicted class and confidence
    predicted_class = class_names[np.argmax(predictions[0])]
    confidence = round(100 * (np.max(predictions[0])), 2)  # Convert confidence to percentage and round
    
    return predicted_class, confidence

if uploaded_file is not None:
    # Load and preprocess the uploaded image
    img_array = load_and_preprocess_image(uploaded_file)

    # # Predict the class of the leaf disease
    # prediction = model.predict(img_array)
    # predicted_class = np.argmax(prediction, axis=1)[0]
    # confidence = np.max(prediction) * 100  # Confidence score in percentage
    # Predict the class of the leaf disease
# Function to predict the class and confidence score


# Streamlit app interface

    # Load and preprocess the uploaded image
    img = Image.open(uploaded_file).convert('RGB')
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Predict the class of the leaf disease using the `predict` function
    predicted_class, confidence = predict(model, img)
    disease_name = class_names.get(predicted_class, "Unknown")
    # Display the predicted class and confidence score
    st.markdown(f"### 🌿 Predicted Disease: **{disease_name}**")
    # st.write(f"Confidence Score: **{confidence:.2f}%**")
    st.write("Confidence Score:", confidence)

    # Store results in session state
    st.session_state["prediction"] = predicted_class
    st.session_state["confidence"] = confidence

    # Display the uploaded image
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

    # Map predicted class to the disease name
    disease_name = class_names.get(predicted_class, "Unknown")

    # # Show prediction results
    # st.markdown(f"### 🌿 Predicted Disease: **{disease_name}**")
    # st.markdown(f"### 🔍 Confidence Score: **{confidence:.2f}%**")

    # Additional Tips for Farmers
    st.markdown("#### 🛠 Tips:")
    if predicted_class == 0:  # Early Blight
        st.write("⚠️ Early Blight detected. Consider using fungicides and practicing crop rotation.")
    elif predicted_class == 1:  # Late Blight
        st.write("⚠️ Late Blight detected. Immediate attention is required, use disease-resistant potato varieties.")
    elif predicted_class == 2:  # Healthy
        st.write("✅ Your leaf is healthy! Keep up the good farming practices.")

    # Optional: Display a pie chart with prediction probabilities
    st.write("### 🔢 Prediction Probabilities:")
    prob_df = {class_names[i]: float(prediction[0][i]) * 100 for i in range(len(class_names))}
    st.bar_chart(prob_df)
    if st.button("🔄 Rerun"):
        st.rerun()

# Sidebar enhancements
st.sidebar.title("About the Disease Classifier")
st.sidebar.info("This tool uses AI to detect common diseases in potato leaves. It's designed to help farmers identify potential issues early.")

st.sidebar.subheader("Disease Types")
st.sidebar.write("🌱 **Early Blight:** A common potato disease caused by a fungus.")
st.sidebar.write("🌱 **Late Blight:** A serious disease that can devastate potato crops.")
st.sidebar.write("🌱 **Healthy:** No signs of disease detected.")

st.sidebar.subheader("How It Works")
st.sidebar.write("Upload a clear image of your potato leaf, and our AI will predict its health based on trained models.")





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







# knn, cnn and svm but same result for all

# import streamlit as st
# import pickle
# import tensorflow as tf
# import numpy as np
# from tensorflow.keras.preprocessing.image import img_to_array, load_img
# from PIL import Image
# from skimage.transform import resize

# # Set the IMAGE_SIZE based on your model's training
# IMAGE_SIZE_CNN = 256  # Size for CNN model
# IMAGE_SIZE_OTHER = 90  # Size for KNN and SVM models

# # Class names for the three conditions
# class_names = {0: "Early Blight", 1: "Late Blight", 2: "Healthy"}

# def load_and_preprocess_image(image, model_type, expected_features):
#     if model_type == "CNN":
#         img = load_img(image, target_size=(IMAGE_SIZE_CNN, IMAGE_SIZE_CNN))
#         img_array = img_to_array(img)
#         img_array = np.expand_dims(img_array, axis=0)
#         img_array = img_array / 255.0
#     else:
#         img = Image.open(image).convert('RGB')
#         img_array = np.array(img)
#         img_array = resize(img_array, (IMAGE_SIZE_OTHER, IMAGE_SIZE_OTHER, 3))
#         img_array = img_array.flatten().reshape(1, -1)
        
#         # Always prepare 8100 features for KNN and SVM
#         if img_array.shape[1] > 8100:
#             img_array = img_array[:, :8100]
#         elif img_array.shape[1] < 8100:
#             img_array = np.pad(img_array, ((0, 0), (0, 8100 - img_array.shape[1])), 'constant')

#     return img_array

# def predict(model, img_array, model_type):
#     if model_type == "CNN":
#         predictions = model.predict(img_array)
#     else:
#         predictions = model.predict_proba(img_array)

#     predicted_class = class_names[np.argmax(predictions[0])]
#     confidence = round(100 * np.max(predictions[0]), 2)
#     return predicted_class, confidence

# def load_model(model_choice):
#     if model_choice == "KNN":
#         with open("knn_model.pkl", "rb") as f:
#             model = pickle.load(f)
#         with open("knn_scaler.pkl", "rb") as f:
#             scaler = pickle.load(f)
#     elif model_choice == "SVM":
#         with open("svm_model.pkl", "rb") as f:
#             model = pickle.load(f)
#         with open("svm_scaler.pkl", "rb") as f:
#             scaler = pickle.load(f)
#     elif model_choice == "CNN":
#         with open("potato_pickle_final (1).pkl", "rb") as f:
#             data = pickle.load(f)
#             model = tf.keras.models.model_from_json(data["architecture"])
#             model.load_weights(data["weights"])
#         scaler = None
#     return model, scaler

# st.markdown("<h1 style='text-align: center; color: green;'>🍃 Potato Leaf Health Check 🍃</h1>", unsafe_allow_html=True)
# st.write("Welcome farmers! Upload a photo of your potato leaf and let our AI help you diagnose its health.")

# uploaded_file = st.file_uploader("Select an image of a potato leaf...", type=["jpg", "jpeg", "png"])
# model_choice = st.selectbox("Choose a model for prediction", ["KNN", "SVM", "CNN"])

# if uploaded_file is not None and model_choice is not None:
#     model, scaler = load_model(model_choice)

#     img = Image.open(uploaded_file).convert('RGB')
#     st.image(img, caption="Uploaded Image", use_column_width=True)
    
#     img_array = load_and_preprocess_image(uploaded_file, model_choice, None)

#     if model_choice in ["KNN", "SVM"]:
#         img_array = scaler.transform(img_array)
#         if model_choice == "SVM":
#             # Adjust features for SVM model if necessary
#             if model.n_features_in_ < img_array.shape[1]:
#                 img_array = img_array[:, :model.n_features_in_]
#             elif model.n_features_in_ > img_array.shape[1]:
#                 img_array = np.pad(img_array, ((0, 0), (0, model.n_features_in_ - img_array.shape[1])), 'constant')

#     predicted_class, confidence = predict(model, img_array, model_choice)

#     st.markdown(f"### 🌿 Predicted Disease: **{predicted_class}**")
#     st.write(f"Confidence Score: **{confidence:.2f}%**")

#     st.markdown("#### 🛠 Tips:")
#     if predicted_class == "Early Blight":
#         st.write("⚠️ Early Blight detected. Consider using fungicides and practicing crop rotation.")
#     elif predicted_class == "Late Blight":
#         st.write("⚠️ Late Blight detected. Immediate attention is required; use disease-resistant potato varieties.")
#     elif predicted_class == "Healthy":
#         st.write("✅ Your leaf is healthy! Keep up the good farming practices.")

# st.sidebar.title("About the Disease Classifier")
# st.sidebar.info("This tool uses AI to detect common diseases in potato leaves. It's designed to help farmers identify potential issues early.")

# st.sidebar.subheader("Disease Types")
# st.sidebar.write("🌱 **Early Blight:** A common potato disease caused by a fungus.")
# st.sidebar.write("🌱 **Late Blight:** A serious disease that can devastate potato crops.")
# st.sidebar.write("🌱 **Healthy:** No signs of disease detected.")

# st.sidebar.subheader("How It Works")
# st.sidebar.write("Upload a clear image of your potato leaf, and our AI will predict its health based on trained models.")










