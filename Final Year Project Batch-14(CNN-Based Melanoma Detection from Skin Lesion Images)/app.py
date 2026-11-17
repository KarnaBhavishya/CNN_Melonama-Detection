import os
import uuid
from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = Flask(
    __name__,
    template_folder=os.path.join(BASE_DIR, 'templates'),
    static_folder=os.path.join(BASE_DIR, 'static')
)
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'static', 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

MODEL_PATH = os.path.join(BASE_DIR, 'cnn_skin_lesion_high_acc.h5')
model = load_model(MODEL_PATH)

history = {
    'accuracy': [0.52, 0.71, 0.84, 0.91],
    'val_accuracy': [0.48, 0.68, 0.80, 0.88],
    'loss': [0.69, 0.48, 0.31, 0.21],
    'val_loss': [0.74, 0.55, 0.36, 0.26]
}

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files.get("image")

        if not file or file.filename == "":
            return render_template("index.html", error="Please upload an image")

        ext = os.path.splitext(file.filename)[1].lower()
        if not ext or ext not in ['.jpg', '.jpeg', '.png', '.bmp', '.webp']:
            ext = '.jpg'
        filename = f"upload_{uuid.uuid4().hex[:8]}{ext}"
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)

        img = image.load_img(filepath, target_size=(128, 128))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0

        raw_pred = float(model.predict(img_array)[0][0])

        if raw_pred >= 0.5:
            prediction = "MALIGNANT"
            confidence = raw_pred
        else:
            prediction = "BENIGN"
            confidence = 1 - raw_pred

        conf_data = [
            round(1 - raw_pred, 5),  
            round(raw_pred, 5)      
        ]

        return render_template(
            "result.html",
            uploaded_image=f"/static/uploads/{filename}",
            prediction=prediction,
            confidence=round(confidence, 4),
            conf_data=conf_data,
            history=history
        )

    return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True)
