import os
import cv2
import numpy as np
import pickle
import os
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()  # Load environment variables from .env file

from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
import markdown 

# Initialize Flask app
app = Flask(__name__)

# Define upload folder
UPLOAD_FOLDER = "static/uploads"
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)  # Create folder if it doesn't exist

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Load trained MobileNetV2 model
model = load_model("model.h5", compile=False)  # Load model without re-compiling

# Load class labels
with open("class_labels.pkl", "rb") as f:
    class_labels = pickle.load(f)

# Configure Gemini API
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))  # Use the API key from the environment variable


def predict_image(image_path):
    """Processes an image and makes a prediction using the trained model."""
    try:
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
        img_resized = cv2.resize(img, (224, 224))  # Resize for model input

        # Normalize and expand dimensions
        img_array = np.array(img_resized) / 255.0
        img_array = np.expand_dims(img_array, axis=0)  # Shape (1, 224, 224, 3)

        # Make prediction
        predictions = model.predict(img_array)
        predicted_class = np.argmax(predictions)  # Get class index
        confidence = np.max(predictions) * 100  # Get confidence score

        # Get class label from loaded labels
        predicted_label = class_labels[predicted_class]

        return predicted_label, confidence
    except Exception as e:
        return str(e), 0
# available_models = genai.list_models()
# for model in available_models:
#     print(f"Model: {model.name}")
#     print(f"  Description: {model.description}")
#     print(f"  Supported Generation Methods: {model.supported_generation_methods}")
#     print("-" * 20)
# import google.generativeai as genai
import os

def get_pest_info(pest_name):
    """Fetches information about a pest using Gemini AI."""
    try:
        model = genai.GenerativeModel("models/gemini-1.5-pro-latest") # or "models/gemini-1.5-flash-latest"
        response = model.generate_content(
            f"Explain the pest '{pest_name}', its damage to crops, and effective control measures."
        )
        return response.text
    except Exception as e:
        print(f"Error fetching information: {e}")
        return f"Error fetching information: {str(e)}"
@app.route("/", methods=["GET", "POST"])
def upload_file():
    """Handles image uploads and displays prediction results."""
    if request.method == "POST":
        if "file" not in request.files:
            return render_template("index.html", error="No file uploaded")

        file = request.files["file"]
        if file.filename == "":
            return render_template("index.html", error="No selected file")

        if file:
            # Save uploaded image
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
            file.save(filepath)

            # Predict the uploaded image
            predicted_label, confidence = predict_image(filepath)

           
             # Get pest information from Gemini AI
            pest_info = get_pest_info(predicted_label)
            pest_info_html = markdown.markdown(pest_info) # convert markdown to html


            return render_template(
                "index.html",
                uploaded_image=filepath,
                result=predicted_label,
                confidence=confidence,
                pest_info=pest_info_html
            )

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
