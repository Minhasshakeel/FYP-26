from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import pandas as pd
import numpy as np
import joblib
import tensorflow
from tensorflow.keras.models import load_model

# APp
app = Flask(__name__)


CORS(app)

# DATA
df = pd.read_csv("final_data.csv", low_memory=False)

# CLEAN
df["player_name"] = (
    df["player_name"]
    .astype(str)
    .str.lower()
    .str.strip()
)

# MODEL
model = load_model("best_transformer.keras", compile=False)


# SCALER
scaler = joblib.load("scaler.pkl")

# ================= FEATURES =================

train_columns = [
    "offensive_skill",
    "defensive_skill",
    "efficiency",
    "aggression",
    "mistake_penalty",
    "experience",
    "physical_index"
]

@app.route("/predict")
def predict():

    try:

        # INPUT
        name = request.args.get("name", "").strip().lower()

        print("PLAYER:", name)

        # FIND PLAYER
        player = df[
            df["player_name"]
            .astype(str)
            .str.strip()
            .str.lower() == name
        ]

        print("PLAYER FOUND:", player.shape)

        if player.empty:
            return jsonify({
                "status": "error",
                "message": "Player not found"
            })

        # FEATURES
        features = player[train_columns]

        print("FEATURES:")
        print(features.head())

        # CLEAN
        features = features.apply(
            pd.to_numeric,
            errors="coerce"
        ).fillna(0)

        # SCALE
        features_scaled = scaler.transform(features)

        print("SCALED:")
        print(features_scaled)

        # RESHAPE FOR TRANSFORMER
        features_scaled = np.expand_dims(
            features_scaled,
            axis=1
        )

        print("RESHAPED:")
        print(features_scaled.shape)

        # PREDICT
        prediction_prob = model.predict(
            features_scaled
        )

        print("RAW PREDICTION:")
        print(prediction_prob)

        prediction_prob = prediction_prob[0][0]

        result = (
            "Selected"
            if prediction_prob > 0.5
            else "Not Selected"
        )

        return jsonify({
            "status": "success",
            "prediction": result,
            "confidence": float(prediction_prob)
        })

    except Exception as e:

        print("ERROR:")
        print(str(e))

        return jsonify({
            "status": "error",
            "message": str(e)
        })
    

# ================= RUN =================

if __name__ == "__main__":

    app.run(debug=True)