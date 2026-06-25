from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
import os

app = Flask(__name__, static_folder="static")

rf        = joblib.load("model.joblib")
le_dict   = joblib.load("label_encoders.joblib")
feat_cols = joblib.load("feature_cols.joblib")

def encode(col, val):
    le = le_dict.get(col)
    if le is None:
        return val
    if val in le.classes_:
        return le.transform([val])[0]
    return 0

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json

    a_parts = []
    if float(data["FA_frac"]) > 0: a_parts.append("FA")
    if float(data["MA_frac"]) > 0: a_parts.append("MA")
    if float(data["Cs_frac"]) > 0: a_parts.append("Cs")
    a_ion = "; ".join(sorted(a_parts)) if a_parts else "MA"

    b_parts = []
    if float(data["Pb_frac"]) > 0: b_parts.append("Pb")
    if float(data["Sn_frac"]) > 0: b_parts.append("Sn")
    b_ion = "; ".join(b_parts) if b_parts else "Pb"

    c_parts = []
    if float(data["Br_frac"]) > 0: c_parts.append("Br")
    if float(data["I_frac"])  > 0: c_parts.append("I")
    c_ion = "; ".join(c_parts) if c_parts else "I"

    X = np.array([[
        float(data["FA_frac"]),
        float(data["MA_frac"]),
        float(data["Cs_frac"]),
        float(data["Pb_frac"]),
        float(data["Sn_frac"]),
        float(data["I_frac"]),
        float(data["Br_frac"]),
        float(data["Bandgap"]),
        float(data["Cell_Area"]),
        float(data["Anneal_Temp"]),
        encode("HTL",          data["HTL"]),
        encode("ETL",          data["ETL"]),
        encode("Backcontact",  data["Backcontact"]),
        encode("Solvents",     data["Solvents"]),
        encode("Synth_Atm",    data["Synth_Atm"]),
        encode("Dep_Procedure",data["Dep_Procedure"]),
        int(data["AntiSolvent"]),
        int(data["Is_Module"]),
        int(data["Solv_Anneal"]),
    ]])

    pred = rf.predict(X)[0]
    return jsonify({
        "pce"  : round(float(pred), 2),
        "a_ion": a_ion,
        "b_ion": b_ion,
        "c_ion": c_ion
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    app.run(host="0.0.0.0", port=port)
