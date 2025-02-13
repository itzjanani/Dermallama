from flask import Flask, request, render_template, jsonify
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import numpy as np
import openai
import os
from rag import ask_llama3

# Disable OneDNN optimization
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

app = Flask(__name__)

# Load the skin disease classification model
model = load_model("skin_disease_model.h5")

# Define class names
class_names = ["Cellulitis", "Impetigo", "Athlete's Foot", "Nail Fungus", "Ringworm",
               "Cutaneous Larva Migrans", "Chickenpox", "Shingles"]

# Function to preprocess the image before feeding it to the model
def preprocess_image(img, target_size):
    img = img.resize(target_size)
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = img / 255.0  # Normalize pixel values
    return img

# Home route
@app.route("/", methods=["GET", "POST"])
def predict():
    if request.method == "POST":
        file = request.files["file"]

        if file:
            image = Image.open(file.stream)
            image = preprocess_image(image, target_size=(150, 150))
            prediction = model.predict(image)
            predicted_class = class_names[np.argmax(prediction)]

            return render_template("index.html", prediction=predicted_class)

    return render_template("index.html")

# Chatbot route
@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    user_message = data.get("message", "")
    detected_disease = data.get("disease", "")

    if not user_message:
        return jsonify({"response": "Please enter a message."})

    if not detected_disease:
        return jsonify({"response": "Disease detection required before chat."})

    # Define the chat prompt
    prompt = f"The detected skin disease is {detected_disease}. {user_message}"

    try:
        answer = ask_llama3(prompt)
        print(answer)
        return jsonify({"response": str(answer)})

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"response":"Sorry, there was an error processing your request."})

if __name__ == "__main__":
    app.run(debug=True, port=5001)
