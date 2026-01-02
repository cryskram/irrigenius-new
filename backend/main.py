import os

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "1"

import base64
import cv2
import random
import uvicorn
from typing import Optional, List
from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware


import numpy as np
import tensorflow as tf
import time
import logging

logging.basicConfig(level=logging.INFO)


class _StubDiseaseModel:
    def __init__(self, *a, **k):
        pass

    def predict(self, image_bytes):
        dummy = np.zeros((224, 224, 3), dtype=np.uint8)
        return "healthy", 0.99, dummy, ""


class _StubIrrigationAI:
    def predict(self, temp, humidity, moisture):
        return ("WATER" if moisture < 30 else "NO_WATER", 0.85, 6.5)


try:
    from disease_model import DiseaseModel

    disease_model = DiseaseModel("plant_disease_model.h5")
except Exception:
    disease_model = _StubDiseaseModel()

try:
    from model import IrrigationAI

    ai = IrrigationAI()
except Exception:
    ai = _StubIrrigationAI()


app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


RL_MODEL_PATHS = ["rl_treatment_optimized.h5", "rl_treatment_fast.h5"]
RL_MODEL = None
RL_ACTION_SIZE = None
RL_INPUT_SIZE = None

saved_plants = []


def load_rl_model_try(paths):
    for p in paths:
        if os.path.exists(p):
            model = tf.keras.models.load_model(p, compile=False)
            out_shape = model.output_shape
            in_shape = model.input_shape
            if out_shape is None or in_shape is None:
                continue
            action_dim = out_shape[-1]
            input_dim = in_shape[-1]
            return model, input_dim, action_dim
    return None, None, None


RL_MODEL, RL_INPUT_SIZE, RL_ACTION_SIZE = load_rl_model_try(RL_MODEL_PATHS)

if RL_MODEL is None:
    logging.warning("RL model not found. Searched:", RL_MODEL_PATHS)
else:
    logging.info(
        f"RL model loaded. input_size={RL_INPUT_SIZE}, action_size={RL_ACTION_SIZE}"
    )


class ModeRequest(BaseModel):
    mode: str


class Plant(BaseModel):
    x: float
    y: float
    severity: float


class TreatmentRequest(BaseModel):
    plants: Optional[List[Plant]] = None
    num_plants: Optional[int] = None
    field_size: Optional[float] = 50.0


def build_state_vector(drone_pos: np.ndarray, plants_arr: np.ndarray):
    return np.concatenate([drone_pos, plants_arr.flatten()]).astype(np.float32)


def rl_act_from_model(state: np.ndarray):
    global RL_MODEL
    if RL_MODEL is None:
        raise RuntimeError("RL model not loaded on server.")

    start = time.time()
    q = RL_MODEL.predict(np.array([state]), verbose=0)[0]
    latency = (time.time() - start) * 1000
    logging.info(f"RL inference time: {latency:.2f} ms")

    return int(np.argmax(q))


@app.post("/save-plants")
def save_plants(req: TreatmentRequest):
    global saved_plants

    if not req.plants:
        raise HTTPException(status_code=400, detail="No plants provided")

    saved_plants = [
        {"x": float(p.x), "y": float(p.y), "severity": float(p.severity)}
        for p in req.plants
    ]

    return {"status": "saved", "count": len(saved_plants)}


@app.get("/get-saved-plants")
def get_saved_plants():
    global saved_plants
    return {"plants": saved_plants}


@app.post("/predict/drone-path")
def predict_drone_path(req: TreatmentRequest):
    if RL_MODEL is None:
        raise HTTPException(
            status_code=500,
            detail="RL model not loaded. Upload 'rl_treatment_optimized.h5' to backend folder.",
        )

    if req.plants and len(req.plants) > 0:
        plants = np.array(
            [[p.x, p.y, p.severity] for p in req.plants],
            dtype=np.float32,
        )
    else:
        num = int(req.num_plants or RL_ACTION_SIZE)
        field = float(req.field_size or 50.0)
        plants = np.random.rand(num, 3).astype(np.float32)
        plants[:, 0:2] *= field
        plants[:, 2] *= 10.0

    num_plants = len(plants)

    if num_plants != RL_ACTION_SIZE:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid plant count {num_plants}. RL model expects {RL_ACTION_SIZE} plants.",
        )

    if np.any(plants[:, 2] < 0):
        raise HTTPException(
            status_code=400,
            detail="Severity values must be >= 0.",
        )

    drone_pos = np.array([0.0, 0.0], dtype=np.float32)
    treated = [False] * num_plants
    path = []
    order = []

    for step in range(num_plants):
        state = build_state_vector(drone_pos, plants)

        try:
            action = rl_act_from_model(state)
        except RuntimeError as e:
            raise HTTPException(status_code=500, detail=str(e))

        retries = 0
        while treated[action]:
            action = random.randint(0, num_plants - 1)
            retries += 1
            if retries > 20:
                for i in range(num_plants):
                    if not treated[i]:
                        action = i
                        break
                break

        px, py, sev = plants[action]
        path.append({"x": float(px), "y": float(py), "severity": float(sev)})
        order.append(int(action))

        drone_pos = np.array([px, py], dtype=np.float32)
        treated[action] = True

    return {"path": path, "order": order}


@app.post("/generate/random-plants")
def generate_random_plants(
    num_plants: Optional[int] = None, field_size: Optional[float] = 50.0
):
    if num_plants is None:
        num_plants = RL_ACTION_SIZE or 5

    plants = np.random.rand(num_plants, 3).astype(np.float32)
    plants[:, 0:2] *= float(field_size)
    plants[:, 2] *= 10.0

    return {
        "plants": [
            {"x": float(p[0]), "y": float(p[1]), "severity": float(p[2])}
            for p in plants
        ]
    }


@app.post("/mode")
def change_mode(req: ModeRequest):
    global MODE
    MODE = req.mode
    return {"status": "updated", "mode": MODE}


@app.get("/drone/position")
def drone_position():
    try:
        from airsim_stream import get_drone_position
    except Exception as e:
        return {"status": "error", "details": f"airsim_stream import failed: {e}"}

    try:
        pos = get_drone_position()
        return {"status": "ok", "pos": pos}
    except Exception as e:
        return {"status": "error", "details": str(e)}


@app.get("/simulate")
def simulate(mode: str = "normal"):
    base_temp = 28
    base_humidity = 45
    base_moist = 20

    if mode == "dry":
        base_temp += 5
        base_humidity -= 10
        base_moist -= 15

    if mode == "humid":
        base_temp -= 2
        base_humidity += 20
        base_moist += 10

    if mode == "rainy":
        base_temp -= 4
        base_humidity += 30
        base_moist += 25

    temp = base_temp + random.uniform(-1, 1)
    humidity = base_humidity + random.uniform(-3, 3)
    moisture = base_moist + random.uniform(-5, 5)

    decision, conf, ph = ai.predict(temp, humidity, moisture)

    explanation = get_explanation(
        round(temp, 2),
        round(humidity, 2),
        round(moisture, 2),
        decision,
    )

    return {
        "temp": round(temp, 2),
        "humidity": round(humidity, 2),
        "moisture": round(moisture, 2),
        "decision": decision,
        "confidence": conf,
        "ph": ph,
        "explanation": explanation,
    }


def get_explanation(temp, humidity, moisture, decision):
    if decision == "WATER":
        return f"Soil moisture is low ({moisture}%), and temperature is {temp}C — irrigation needed."
    return f"Soil moisture ({moisture}%) is sufficient — irrigation not required."


@app.post("/detect-disease")
async def detect_disease(file: UploadFile = File(...)):
    image_bytes = await file.read()
    label, confidence, orig_img, heatmap = disease_model.predict(image_bytes)

    _, orig_buf = cv2.imencode(".jpg", orig_img)
    orig_base64 = base64.b64encode(orig_buf).decode()

    return {
        "label": label,
        "confidence": float(confidence),
        "heatmap": heatmap,
        "original": orig_base64,
    }


@app.get("/")
def home():
    return {"status": "running"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
