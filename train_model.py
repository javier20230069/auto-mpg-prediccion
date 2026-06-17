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

# Nombres que se muestran en el formulario de la pagina.
FEATURE_LABELS = {
    "cylinders": "Cilindros",
    "displacement": "Desplazamiento del motor",
    "horsepower": "Caballos de fuerza",
    "weight": "Peso del auto",
    "acceleration": "Aceleracion",
    "model_year": "Anio del modelo",
    "origin": "Origen",
}

# Explicacion breve de cada campo del formulario.
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
    # El dataset original no trae encabezados, por eso aqui se definen los nombres de las columnas.
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

    # Se carga el dataset directamente desde la URL de UCI.
    # na_values="?" convierte los signos de interrogacion en valores nulos.
    data = pd.read_csv(
        DATA_URL,
        sep=r"\s+",
        names=columns,
        na_values="?",
    )
    return data


def train_and_save(output_dir="."):
    # Define la carpeta donde se guardaran el modelo y la informacion de las variables.
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Carga los datos desde internet.
    df = load_data()

    # Variables disponibles para intentar predecir el consumo mpg.
    # No se usa car_name porque es texto libre y no aporta directamente al formulario.
    features = [
        "cylinders",
        "displacement",
        "horsepower",
        "weight",
        "acceleration",
        "model_year",
        "origin",
    ]

    # X contiene las variables de entrada.
    # y contiene la variable objetivo que se quiere predecir: mpg.
    X = df[features]
    y = df["mpg"]

    # Pipeline para seleccionar las mejores 5 caracteristicas:
    # 1. Imputer: rellena valores nulos con la mediana.
    # 2. Scaler: escala las variables para que esten en una escala similar.
    # 3. SelectKBest: elige las 5 variables mas relacionadas con mpg.
    selector_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("selector", SelectKBest(score_func=f_regression, k=5)),
        ]
    )

    # Entrena el selector usando X y y.
    selector_pipeline.fit(X, y)

    # Obtiene el paso llamado "selector" para saber que columnas fueron elegidas.
    selector = selector_pipeline.named_steps["selector"]
    selected_features = X.columns[selector.get_support()].tolist()

    # Guarda una tabla con la puntuacion de cada variable.
    # Esta tabla sirve para saber cuales variables tuvieron mayor relacion con mpg.
    ranking = pd.DataFrame(
        {
            "feature": X.columns,
            "score": selector.scores_,
            "selected": selector.get_support(),
        }
    ).sort_values("score", ascending=False)

    # Se deja X solamente con las caracteristicas seleccionadas.
    X_selected = X[selected_features]

    # Se divide el dataset en entrenamiento y prueba.
    # 80% se usa para entrenar y 20% para evaluar.
    X_train, X_test, y_train, y_test = train_test_split(
        X_selected,
        y,
        test_size=0.2,
        random_state=42,
    )

    # Modelo final:
    # 1. Imputer: vuelve a asegurar que no haya nulos.
    # 2. Scaler: escala las variables.
    # 3. LinearRegression: modelo de regresion que predice el valor de mpg.
    model = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("regression", LinearRegression()),
        ]
    )

    # Entrena el modelo con los datos de entrenamiento.
    model.fit(X_train, y_train)

    # Hace predicciones con los datos de prueba.
    predictions = model.predict(X_test)

    # Calcula metricas para saber que tan bien predice el modelo.
    # MAE mide el error promedio.
    # R2 mide que tanto explica el modelo la variacion de mpg.
    metrics = {
        "mae": float(mean_absolute_error(y_test, predictions)),
        "r2": float(r2_score(y_test, predictions)),
    }

    # Crea informacion para mostrar los campos del formulario:
    # nombre, etiqueta, ayuda, minimo, maximo y promedio.
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

    # Guarda toda la informacion del proyecto en un diccionario.
    # Esto se usa despues en app.py para construir el formulario.
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

    # Guarda el modelo entrenado en model.pkl.
    joblib.dump(model, output_path / "model.pkl")

    # Guarda la informacion de variables y metricas en feature_info.json.
    (output_path / "feature_info.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    # Regresa la informacion para poder imprimirla cuando se ejecute este archivo.
    return metadata


if __name__ == "__main__":
    # Si este archivo se ejecuta directamente, entrena y guarda el modelo.
    info = train_and_save()
    print("Modelo guardado en model.pkl")
    print("Caracteristicas seleccionadas:", ", ".join(info["selected_features"]))
