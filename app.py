import os
import tensorflow as tf
import gradio as gr

# Adjust the path to where the model file is located in the Hugging Face environment
model_path = r'C:\Users\anwes\OneDrive\Desktop\AI_Music_Generator\scripts\AI_Music_Generator.h5'

# Load the model
model = tf.keras.models.load_model(model_path)