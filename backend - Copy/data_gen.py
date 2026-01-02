import random
import csv
import math


DISEASE_TYPES = [
    "healthy",
    "early_blight",
    "late_blight",
    "mold",
    "nutrient_deficiency",
]


def generate_row(id_idx, grid_x, grid_y, mode="random"):
    temp = random.uniform(15, 38)
    humidity = random.uniform(30, 95)
    moisture = random.uniform(5, 70)
    ph = random.uniform(5.5, 8.0)
    disease = random.choices(DISEASE_TYPES, weights=[0.7, 0.1, 0.07, 0.08, 0.05])[0]
    severity = 0.0 if disease == "healthy" else round(random.uniform(0.2, 1.0), 3)

    urgency = 0.0
    if disease != "healthy":
        urgency = severity * (1 + max(0, (30 - moisture) / 30))
    if disease == "late_blight":
        urgency *= 1.2
    if disease == "nutrient_deficiency":
        urgency *= 0.9

    return {
        "id": f"plant-{id_idx}",
        "x": grid_x,
        "y": grid_y,
        "disease": disease,
        "severity": severity,
        "temperature": round(temp, 2),
        "humidity": round(humidity, 2),
        "moisture": round(moisture, 2),
        "ph": round(ph, 2),
        "priority_score": round(min(1.0, urgency), 3),
    }


def generate_csv(path="treatment_training_data.csv", n=5000, grid_w=50, grid_h=50):
    keys = [
        "id",
        "x",
        "y",
        "disease",
        "severity",
        "temperature",
        "humidity",
        "moisture",
        "ph",
        "priority_score",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for i in range(n):
            gx = random.randint(0, grid_w - 1)
            gy = random.randint(0, grid_h - 1)
            writer.writerow(generate_row(i, gx, gy))


if __name__ == "__main__":
    generate_csv()
    print("Training CSV generated: treatment_training_data.csv")
