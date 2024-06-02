import streamlit as st
from transformers import BertTokenizer, TFBertForSequenceClassification
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
import cv2
import numpy as np

# Load models
@st.cache(allow_output_mutation=True)
def load_models():
    text_model = TFBertForSequenceClassification.from_pretrained('bert-base-uncased')
    image_model = EfficientNetB0(weights='imagenet')
    video_model = tf.keras.models.load_model('video_model.h5')  # Assume you have trained this model
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    return text_model, image_model, video_model, tokenizer

text_model, image_model, video_model, tokenizer = load_models()

# Text-based Fake News Detection
def predict_text(text):
    inputs = tokenizer(text, return_tensors='tf', truncation=True, padding=True)
    outputs = text_model(inputs)
    predictions = tf.nn.softmax(outputs.logits, axis=-1)
    return predictions.numpy()[0]

# Image-based Fake News Detection
def predict_image(image):
    image = cv2.resize(image, (224, 224))
    image = np.expand_dims(image, axis=0)
    image = image / 255.0
    predictions = image_model.predict(image)
    return predictions

# Video-based Fake News Detection
def predict_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (64, 64))
        frames.append(frame)
    cap.release()
    frames = np.array(frames) / 255.0
    frames = np.expand_dims(frames, axis=0)
    predictions = video_model.predict(frames)
    return predictions

# Streamlit Interface
st.title('Fake News Detection App')

st.header('Text-based Detection')
text_input = st.text_area('Enter the news text')
if st.button('Predict Text'):
    text_predictions = predict_text(text_input)
    st.write(f'Real: {text_predictions[0]}, Fake: {text_predictions[1]}')

st.header('Image-based Detection')
uploaded_image = st.file_uploader("Upload an image", type=["jpg", "png"])
if uploaded_image is not None:
    image = cv2.imdecode(np.fromstring(uploaded_image.read(), np.uint8), 1)
    image_predictions = predict_image(image)
    st.image(image, channels="BGR")
    st.write(f'Prediction: {np.argmax(image_predictions)}')

st.header('Video-based Detection')
uploaded_video = st.file_uploader("Upload a video", type=["mp4", "avi"])
if uploaded_video is not None:
    with open('temp_video.mp4', 'wb') as f:
        f.write(uploaded_video.read())
    video_predictions = predict_video('temp_video.mp4')
    st.write(f'Prediction: {np.argmax(video_predictions)}')