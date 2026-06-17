from pathlib import Path
import json

import joblib
import pandas as pd
from flask import Flask, render_template, request

from train_model import train_and_save


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model.pkl"
INFO_PATH = BASE_DIR / "feature_info.json"

if not MODEL_PATH.exists() or not INFO_PATH.exists():
    train_and_save(BASE_DIR)

model = joblib.load(MODEL_PATH)
feature_info = json.loads(INFO_PATH.read_text(encoding="utf-8"))

app = Flask(__name__)


@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    error = None
    form_values = {}

    if request.method == "POST":
        try:
            row = {}
            for feature in feature_info["feature_metadata"]:
                name = feature["name"]
                value = request.form.get(name, "")
                form_values[name] = value
                row[name] = float(value)

            input_data = pd.DataFrame([row])
            prediction = float(model.predict(input_data)[0])
        except ValueError:
            error = "Revisa los datos: todos los campos deben tener numeros."
        except Exception as exc:
            error = f"No se pudo hacer la prediccion: {exc}"

    return render_template(
        "index.html",
        feature_info=feature_info,
        prediction=prediction,
        error=error,
        form_values=form_values,
    )


if __name__ == "__main__":
    app.run(debug=True)
