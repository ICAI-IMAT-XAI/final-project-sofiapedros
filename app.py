import torch
from flask import Flask, request, jsonify, Response
from prometheus_client import Counter, generate_latest, CONTENT_TYPE_LATEST
import numpy as np
from torch.jit import RecursiveScriptModule

device = torch.device("cpu")
PREDICTION_COUNTER = Counter(
    'tumor_prediction_count', 
    'contador de predicciones por tipo de tumor',
    ['tumor'])

def load_model(name: str) -> RecursiveScriptModule:
    """
    Esta función carga un modelo desde la carpeta 'models'.

    Args:
    - name (str): nombre del modelo a cargar.

    Returns:
        modelo en torchscript.
    """
    try:
        model: RecursiveScriptModule = torch.jit.load(
            f"models/{name}.pt",
            map_location=device
        )
    except FileNotFoundError:
        try:
            model: RecursiveScriptModule = torch.jit.load(
                f"../models/{name}.pt",
                map_location=device
            )
        except FileNotFoundError:
            raise FileNotFoundError(f"No se encontró el modelo en 'models/{name}.pt' ni en '../models/{name}.pt'")
    
    model.eval()  # Poner el modelo en modo evaluación
    return model

# Configuración del modelo
MODEL_NAME = "Tab_model_lr_0.0001_bs_8_hd_128x256_dropout_0.2"

try:
    model = load_model(MODEL_NAME)
    print(f"Modelo '{MODEL_NAME}' cargado exitosamente")
except Exception as e:
    print(f"Error al cargar el modelo: {e}")
    model = None

app = Flask(__name__)

@app.route('/metrics')
def metrics():
    return Response(generate_latest(), mimetype=CONTENT_TYPE_LATEST)

@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"error": "Modelo no cargado. Verifica que el archivo .pt existe."}), 500
    
    try:
        # Obtener los datos de la petición JSON
        data = request.get_json(force=True)
        features = np.array(data["features"], dtype=np.float32).reshape(1, -1)

        # Convertir a tensor y enviar al dispositivo
        features_tensor = torch.from_numpy(features).to(device)

        # Realizar la predicción
        with torch.no_grad():
            outputs = model(features_tensor)
            predicted_class = torch.argmax(outputs, dim=1).item()
        
        tumor_map = {0: "bening", 1: 'malignant', 2: 'normal'}
        predicted_tumor = tumor_map.get(predicted_class, 'unknown')
        
        PREDICTION_COUNTER.labels(tumor=predicted_tumor).inc()
        
        # Devolver la predicción
        return jsonify({
            "prediction": int(predicted_class), 
            'tumor': predicted_tumor
        })
    
    except KeyError:
        return jsonify({"error": "Falta el campo 'features' en el JSON"}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route("/health", methods=["GET"])
def health():
    """Endpoint para verificar el estado del servicio"""
    status = "healthy" if model is not None else "unhealthy"
    return jsonify({"status": status}), 200 if model is not None else 503

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)