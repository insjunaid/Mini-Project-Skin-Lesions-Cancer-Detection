from flask import Flask, request, render_template, redirect, url_for
from PIL import Image
import torch
from torchvision import transforms
import os
import numpy as np

app = Flask(__name__)

# Define the class names and their malignancy status
class_names = ["Actinic keratoses and intraepithelial carcinoma(akiec)", "Basal cell carcinoma(bcc)", "Benign keratosis-like lesions(bkl)", "Dermatofibroma(df)", "Melanoma(mel)", "Melanocytic nevi(nv)", "Vascular lesions(vasc)"]
malignancy_status = {
    "Actinic keratoses and intraepithelial carcinoma(akiec)": "Malignant",
    "Basal cell carcinoma(bcc)": "Malignant",
    "Benign keratosis-like lesions(bkl)": "Benign",
    "Dermatofibroma(df)": "Benign",
    "Melanoma(mel)": "Malignant",
    "Melanocytic nevi(nv)": "Benign",
    "Vascular lesions(vasc)": "Benign"
}

recommendations = {
    "Malignant": "Visit a dermatologist immediately for further diagnosis and treatment.",
    "Benign": "The lesion appears benign. However, monitor for changes and consult a dermatologist if concerned."
}

# Load the trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torch.load("final_skin_lesion_detection_model2.pth", map_location=device)
model.eval()

# Define the image transformations
data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Route for uploading and predicting
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # Handle file upload
        if "file" not in request.files:
            return redirect(request.url)
        file = request.files["file"]
        if file.filename == "":
            return redirect(request.url)

        # Save the uploaded file in the static/uploads folder
        if file:
            upload_folder = "static/uploads"
            os.makedirs(upload_folder, exist_ok=True)
            file_path = os.path.join(upload_folder, file.filename)
            file.save(file_path)

            # Process and predict
            image = Image.open(file_path).convert("RGB")
            input_tensor = data_transforms(image).unsqueeze(0).to(device)
            with torch.no_grad():
                outputs = model(input_tensor)
                _, predicted_idx = torch.max(outputs, 1)
                predicted_class = class_names[predicted_idx.item()]
                malignancy = malignancy_status[predicted_class]
                recommendation = recommendations[malignancy]

            # Pass the image path relative to the static folder
            return render_template(
                "index.html",
                image_path=f"uploads/{file.filename}",
                predicted_class=predicted_class,
                malignancy=malignancy,
                recommendation=recommendation
            )
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=False, host='0.0.0.0', port=os.environ.get('PORT', 5000))
