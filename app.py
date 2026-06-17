from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np


app = Flask(__name__, static_folder="static")

# Load model
rf         = joblib.load("model.joblib")
le_dict    = joblib.load("label_encoders.joblib")
feat_cols  = joblib.load("feature_cols.joblib")

def encode(col, val):
    le = le_dict.get(col)
    if le is None:
        return val
    if val in le.classes_:
        return le.transform([val])[0]
    return 0  # unknown value

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    X = np.array([[
        encode("B_Ion",        data["B_Ion"]),
        encode("A_Ion",        data["A_Ion"]),
        encode("C_Ion",        data["C_Ion"]),
        float(data["Bandgap"]),
        float(data["Cell_Area"]),
        encode("HTL",          data["HTL"]),
        encode("ETL",          data["ETL"]),
        encode("Backcontact",  data["Backcontact"]),
        encode("Solvents",     data["Solvents"]),
        int(data["AntiSolvent"]),
        encode("Synth_Atm",    data["Synth_Atm"]),
        encode("Dep_Procedure",data["Dep_Procedure"]),
        float(data["Anneal_Temp"]),
        int(data["Is_Module"]),
        int(data["Solv_Anneal"]),
    ]])
    pred = rf.predict(X)[0]
    return jsonify({"pce": round(float(pred), 2)})

import os

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)