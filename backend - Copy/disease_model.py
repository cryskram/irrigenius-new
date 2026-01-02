import numpy as np
import tensorflow as tf
import cv2
import base64
from gradcam import generate_gradcam


class DiseaseModel:
    def __init__(self, model_path="plant_disease_model.h5"):
        self.model = tf.keras.models.load_model(model_path)

        self.class_names = ["Healthy", "Early Blight", "Late Blight", "Leaf Mold"]

    def preprocess(self, image_bytes):
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img_resized = cv2.resize(img, (224, 224))

        img_array = img_resized.astype("float32") / 255.0

        return img_array, img_resized

    def predict(self, image_bytes):
        img_array, orig = self.preprocess(image_bytes)
        input_tensor = np.expand_dims(img_array, axis=0)

        preds = self.model.predict(input_tensor)[0]

        label_idx = np.argmax(preds)
        confidence = float(preds[label_idx])
        label = self.class_names[label_idx]

        heatmap = generate_gradcam(self.model, input_tensor)

        return label, confidence, orig, heatmap
