from pathlib import Path
import json

import joblib
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data"

FEATURE_LABELS = {
    "cylinders": "Cilindros",
    "displacement": "Desplazamiento del motor",
    "horsepower": "Caballos de fuerza",
    "weight": "Peso del auto",
    "acceleration": "Aceleracion",
    "model_year": "Anio del modelo",
    "origin": "Origen",
}

FEATURE_HELP = {
    "cylinders": "Numero de cilindros del motor.",
    "displacement": "Tamanio del motor.",
    "horsepower": "Potencia del motor.",
    "weight": "Peso del automovil.",
    "acceleration": "Tiempo de aceleracion.",
    "model_year": "Anio del modelo, por ejemplo 70, 76 u 82.",
    "origin": "1 = USA, 2 = Europa, 3 = Japon.",
}


def load_data():
    columns = [
        "mpg",
        "cylinders",
        "displacement",
        "horsepower",
        "weight",
        "acceleration",
        "model_year",
        "origin",
        "car_name",
    ]

    data = pd.read_csv(
        DATA_URL,
        sep=r"\s+",
        names=columns,
        na_values="?",
    )
    return data


def train_and_save(output_dir="."):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    df = load_data()

    features = [
        "cylinders",
        "displacement",
        "horsepower",
        "weight",
        "acceleration",
        "model_year",
        "origin",
    ]

    X = df[features]
    y = df["mpg"]

    selector_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("selector", SelectKBest(score_func=f_regression, k=5)),
        ]
    )

    selector_pipeline.fit(X, y)
    selector = selector_pipeline.named_steps["selector"]
    selected_features = X.columns[selector.get_support()].tolist()

    ranking = pd.DataFrame(
        {
            "feature": X.columns,
            "score": selector.scores_,
            "selected": selector.get_support(),
        }
    ).sort_values("score", ascending=False)

    X_selected = X[selected_features]

    X_train, X_test, y_train, y_test = train_test_split(
        X_selected,
        y,
        test_size=0.2,
        random_state=42,
    )

    model = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("regression", LinearRegression()),
        ]
    )

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    metrics = {
        "mae": float(mean_absolute_error(y_test, predictions)),
        "r2": float(r2_score(y_test, predictions)),
    }

    feature_metadata = []
    for feature in selected_features:
        values = X[feature].dropna()
        feature_metadata.append(
            {
                "name": feature,
                "label": FEATURE_LABELS[feature],
                "help": FEATURE_HELP[feature],
                "min": float(values.min()),
                "max": float(values.max()),
                "mean": float(values.mean()),
            }
        )

    metadata = {
        "dataset": "Auto MPG",
        "target": "mpg",
        "model": "LinearRegression",
        "feature_selection": "SelectKBest con f_regression",
        "selected_features": selected_features,
        "feature_metadata": feature_metadata,
        "ranking": ranking.to_dict(orient="records"),
        "metrics": metrics,
    }

    joblib.dump(model, output_path / "model.pkl")
    (output_path / "feature_info.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    return metadata


if __name__ == "__main__":
    info = train_and_save()
    print("Modelo guardado en model.pkl")
    print("Caracteristicas seleccionadas:", ", ".join(info["selected_features"]))
