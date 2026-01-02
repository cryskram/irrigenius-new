import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor


class IrrigationAI:
    def __init__(self):
        self.clf = GradientBoostingClassifier()
        self.ph_model = GradientBoostingRegressor()
        self.train_models()

    def train_models(self):
        data = []
        labels = []
        ph_values = []

        for _ in range(4000):
            temp = np.random.uniform(20, 35)
            humidity = np.random.uniform(20, 90)
            moisture = np.random.uniform(0, 100)

            if moisture < 45 and temp > 28:
                y = 1
            else:
                y = 0

            ph = 6.8

            if temp > 32:
                ph -= 0.4
            if temp < 20:
                ph += 0.2

            if moisture < 40:
                ph -= 0.3
            if moisture > 70:
                ph += 0.3

            if humidity > 70:
                ph += 0.3
            if humidity < 30:
                ph -= 0.2

            ph += np.random.uniform(-0.05, 0.05)

            data.append([temp, humidity, moisture])
            labels.append(y)
            ph_values.append(ph)

        X = np.array(data)
        y = np.array(labels)
        ph_vals = np.array(ph_values)

        self.clf.fit(X, y)
        self.ph_model.fit(X, ph_vals)

    def predict(self, temp, humidity, moisture):
        inp = np.array([[temp, humidity, moisture]])

        pred = self.clf.predict(inp)[0]
        prob = self.clf.predict_proba(inp)[0][1] * 100

        ph_pred = round(float(self.ph_model.predict(inp)[0]), 2)
        print("DEBUG ->", temp, humidity, moisture)

        return ("WATER" if pred == 1 else "STOP", round(prob, 2), ph_pred)
