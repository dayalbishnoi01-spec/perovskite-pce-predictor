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

def build_ion_string(ions, fracs):
    # ions = ["FA", "Cs", "MA"], fracs = [0.85, 0.10, 0.05]
    # returns "FA; Cs; MA" for non-zero fractions
    parts = [ion for ion, f in zip(ions, fracs) if float(f) > 0]
    if len(parts) == 1:
        return parts[0]
    return "; ".join(parts)

def build_c_ion_string(i_frac, br_frac):
    i_frac  = round(float(i_frac), 2)
    br_frac = round(float(br_frac), 2)
    if br_frac == 0:
        return "I"
    if i_frac == 0:
        return "Br"
    return "Br; I"

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json

    # Build ion strings from fractions
    a_ion = build_ion_string(
        ["FA", "Cs", "MA"],
        [data["FA_frac"], data["Cs_frac"], data["MA_frac"]]
    )
    b_ion = build_ion_string(
        ["Pb", "Sn"],
        [data["Pb_frac"], data["Sn_frac"]]
    )
    c_ion = build_c_ion_string(data["I_frac"], data["Br_frac"])

    X = np.array([[
        encode("B_Ion",        b_ion),
        encode("A_Ion",        a_ion),
        encode("C_Ion",        c_ion),
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
    return jsonify({
        "pce"   : round(float(pred), 2),
        "a_ion" : a_ion,
        "b_ion" : b_ion,
        "c_ion" : c_ion
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
