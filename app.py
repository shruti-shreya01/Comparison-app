# import streamlit as st
# import pickle
# import tensorflow as tf
# import os
# from tensorflow.keras.preprocessing.image import img_to_array, load_img
# from PIL import Image, ImageOps
# import numpy as np

# IMAGE_SIZE = 256

# # Function to load and preprocess the uploaded image
# def load_and_preprocess_image(image):
#     img = load_img(image, target_size=(IMAGE_SIZE, IMAGE_SIZE))
#     img_array = img_to_array(img)
#     img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
#     img_array = img_array / 255.0  # Normalize to [0, 1]
#     return img_array

# # Path to the pickle file
# file_path = "potato_pickle_final (1).pkl"

# # Check if file exists and load the model
# if os.path.exists(file_path):
#     with open(file_path, "rb") as f:
#         data = pickle.load(f)
#     # Reconstruct the model from the architecture
#     model = tf.keras.models.model_from_json(data["architecture"])
# else:
#     st.error(f"Model file not found at path: {file_path}")
#     st.stop()  # Stop execution if file is not found

# class_names = {0: "Early Blight", 1: "Late Blight", 2: "Healthy"}

# # Initialize session state variables
# if "prediction" not in st.session_state:
#     st.session_state["prediction"] = None
# if "confidence" not in st.session_state:
#     st.session_state["confidence"] = None

# # Streamlit app interface
# st.title("Potato Leaf Disease Classification")
# st.write("Upload an image of a potato leaf to classify the disease.")

# # File uploader for image input
# uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"], key="uploaded_file")

# if uploaded_file is not None:
#     # Load and preprocess the uploaded image
#     img_array = load_and_preprocess_image(uploaded_file)
    
#     # Predict the class of the leaf disease
#     prediction = model.predict(img_array)
#     predicted_class = np.argmax(prediction[0])
#     confidence = round(100 * np.max(prediction[0]), 2)  # Updated confidence calculation
    
#     # Store results in session state
#     st.session_state["prediction"] = predicted_class
#     st.session_state["confidence"] = confidence
    
#     # Display the uploaded image
#     st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
    
#     # Map predicted class to the disease name
#     disease_name = class_names.get(predicted_class, "Unknown")
    
#     # Log raw prediction
#     print("Raw Prediction:", prediction)
    
#     # Log predicted class and confidence
#     print(f"Predicted Class: {predicted_class}, Confidence: {confidence}")
    
#     # Display the results in Streamlit
#     st.write(f"Predicted Disease: **{disease_name}**")
#     st.write(f"Confidence Score: **{confidence:.2f}%**")

# # Use a button to rerun the app conditionally
# if st.button("Rerun"):
#     # Check if necessary state is initialized before rerunning
#     if st.session_state["prediction"] is not None and st.session_state["confidence"] is not None:
#         st.rerun()
#     else:
#         st.warning("Please upload an image first.")

# st.sidebar.title("About")
# st.sidebar.info("This app is designed to help farmers and agronomists identify diseases in potato leaves using AI technology.")
# st.sidebar.subheader("About the Model")
# st.sidebar.write("This model classifies potato leaf diseases with high accuracy. The classes are:")
# st.sidebar.write("- Early Blight")
# st.sidebar.write("- Late Blight")
# st.sidebar.write("- Healthy")




#knn
import streamlit as st
import numpy as np
from sklearn.feature_extraction.image import extract_patches_2d
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from skimage.feature import hog
from skimage.color import rgb2gray
import os
import pickle
from PIL import Image

# Load dataset chunks
@st.cache_data
def load_dataset_chunks(save_dir):
    images = []
    labels = []
    for file in os.listdir(save_dir):
        if file.endswith('.npz'):
            data = np.load(os.path.join(save_dir, file))
            images.append(data['images'])
            labels.append(data['labels'])
    
    return np.concatenate(images), np.concatenate(labels)

# Extract HOG features
def extract_hog_features(images):
    hog_features = []
    for image in images:
        # Convert to grayscale
        grayscale_image = rgb2gray(image)
        
        # Extract HOG features
        features = hog(grayscale_image, 
                       pixels_per_cell=(16, 16), 
                       cells_per_block=(2, 2), 
                       visualize=False)
        hog_features.append(features)
    
    return np.array(hog_features)

# Train and save model
@st.cache_resource
def train_and_save_model():
    # Load and preprocess data
    X_original, y = load_dataset_chunks('dataset_chunks')
    
    # Extract HOG features
    X_hog = extract_hog_features(X_original)
    
    # Split dataset
    X_train, X_temp, y_train, y_temp = train_test_split(X_hog, y, test_size=0.4, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    
    # Standardize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    
    # Train KNN model
    knn_model = KNeighborsClassifier(n_neighbors=5)
    knn_model.fit(X_train, y_train)
    
    # Save model and scaler
    with open('knn_model.pkl', 'wb') as f:
        pickle.dump((knn_model, scaler), f)
    
    return knn_model, scaler

# Load model and scaler
@st.cache_resource
def load_model():
    with open('knn_model.pkl', 'rb') as f:
        knn_model, scaler = pickle.load(f)
    return knn_model, scaler

# Preprocess and predict
def preprocess_and_predict(image, model, scaler):
    # Convert to numpy array
    img_array = np.array(image)
    
    # Extract HOG features
    hog_features = extract_hog_features([img_array])
    
    # Standardize features
    X = scaler.transform(hog_features)
    
    # Predict
    prediction = model.predict(X)
    probabilities = model.predict_proba(X)
    
    return prediction[0], probabilities[0]

# Streamlit app
st.title('Potato Leaf Disease Classification (KNN)')

# Check if model exists, if not, train and save it
if not os.path.exists('knn_model.pkl'):
    with st.spinner('Training model... This may take a while.'):
        knn_model, scaler = train_and_save_model()
    st.success('Model trained and saved!')
else:
    knn_model, scaler = load_model()

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    
    # Make prediction
    prediction, probabilities = preprocess_and_predict(image, knn_model, scaler)
    
    # Display results
    class_names = ["Early Blight", "Late Blight", "Healthy"]
    st.write(f"Prediction: {class_names[prediction]}")
    st.write(f"Confidence: {probabilities[prediction]*100:.2f}%")
    
    # Display probabilities for all classes
    st.write("Class Probabilities:")
    for i, class_name in enumerate(class_names):
        st.write(f"{class_name}: {probabilities[i]*100:.2f}%")

# Add information about the model
st.sidebar.title("About")
st.sidebar.info("This app uses a K-Nearest Neighbors (KNN) model to classify potato leaf diseases.")
st.sidebar.subheader("Classes")
st.sidebar.write("- Early Blight")
st.sidebar.write("- Late Blight")
st.sidebar.write("- Healthy")



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










