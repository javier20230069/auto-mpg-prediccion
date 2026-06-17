# App de regresion Auto MPG para Render

Esta aplicacion web usa Flask para publicar un modelo de Machine Learning.

## Dataset

Se usa el dataset Auto MPG de UCI Machine Learning Repository.

El objetivo es predecir `mpg`, que significa millas por galon. Es decir, el modelo estima que tan eficiente es un automovil con gasolina.

## Proceso realizado

1. Se carga el dataset Auto MPG desde una URL.
2. Se separan las variables independientes `X` y la variable objetivo `y`.
3. Se tratan valores faltantes con la mediana.
4. Se escalan las variables con `StandardScaler`.
5. Se aplica seleccion de caracteristicas con `SelectKBest` y `f_regression`.
6. Se entrenan solo las 5 caracteristicas seleccionadas.
7. Se guarda el modelo con `joblib`.
8. La pagina muestra las caracteristicas seleccionadas y permite hacer una prediccion.

## Archivos principales

- `app.py`: aplicacion Flask.
- `train_model.py`: entrenamiento, seleccion de caracteristicas y guardado del modelo.
- `requirements.txt`: librerias necesarias.
- `Procfile`: comando para iniciar en Render.
- `templates/index.html`: pagina del formulario.
- `static/style.css`: estilos de la pagina.

## Como ejecutarlo localmente

```bash
pip install -r requirements.txt
python train_model.py
python app.py
```

Despues abre:

```text
http://127.0.0.1:5000/
```

## Como publicarlo en Render

1. Sube esta carpeta a un repositorio de GitHub.
2. En Render crea un nuevo Web Service.
3. Conecta el repositorio de GitHub.
4. Usa estas configuraciones:

```text
Build Command: pip install -r requirements.txt && python train_model.py
Start Command: gunicorn app:app
```

El build command instala librerias y entrena el modelo para generar `model.pkl`.

Tambien funciona porque el proyecto incluye un `Procfile` con:

```text
web: gunicorn app:app
```

## Explicacion para el profesor

En esta actividad publique una aplicacion web en Flask. La aplicacion usa un modelo de regresion para predecir el rendimiento de gasolina de un automovil medido en millas por galon.

Antes de entrenar el modelo se hizo seleccion de caracteristicas con `SelectKBest`, usando `f_regression`. Este metodo revisa que variables tienen mas relacion con la variable objetivo `mpg`. Despues el formulario de la pagina muestra solamente esas variables seleccionadas.

Cuando el usuario llena el formulario, Flask recibe los valores, los manda al modelo entrenado y devuelve una prediccion estimada de `mpg`.
