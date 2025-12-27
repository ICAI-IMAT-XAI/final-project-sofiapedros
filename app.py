import torch
from flask import Flask, request, jsonify, Response
from prometheus_client import Counter, generate_latest, CONTENT_TYPE_LATEST
import numpy as np
from torch.jit import RecursiveScriptModule

from src.explain_sample_tabular import explain_single_sample, ModelWrapper
from src.data import DatasetBaseBreast, get_dataloaders
from src.explain_tabular import get_feature_names

import os
import base64
from PIL import Image

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
MODEL_NAME = "Tab_model_lr_0.0001_bs_8_hd_(128, 256)_dropout_0.2_encode_True_drop_True_cw_True_1.5"
INFO_FILENAME = "data/BrEaST-Lesions-USG-clinical-data-Dec-15-2023.xlsx"
IMAGES_FOLDER = "data/BrEaST-Lesions_USG-images_and_masks/"
SEED = 42

try:
    model = load_model(MODEL_NAME)
    print(f"Modelo '{MODEL_NAME}' cargado exitosamente")
    
    # Cargar datos de entrenamiento para SHAP
    print("Cargando datos de entrenamiento para SHAP...")
    train_loader, _, _ = get_dataloaders(
        INFO_FILENAME,
        IMAGES_FOLDER,
        batch_size=32,
        seed=SEED,
        type="tabular",
        use_label_encoding=True,
        drop_specific_columns=True,
    )
    
    # Preparar X_train para SHAP
    X_train_list = []
    for X, _ in train_loader:
        X_train_list.append(X)
    X_train = torch.cat(X_train_list)
    print(f"Datos de entrenamiento cargados: {X_train.shape}")
    
    # Obtener nombres de features
    base_dataset = DatasetBaseBreast(
        INFO_FILENAME,
        IMAGES_FOLDER,
        use_label_encoding=True,
        drop_specific_columns=True,
    )
    feature_names = get_feature_names(base_dataset)
    print(f"Nombres de features obtenidos: {len(feature_names)} features")
    
    # Crear ModelWrapper
    model_wrapper = ModelWrapper(model, device)
    print("ModelWrapper creado exitosamente")
    
except Exception as e:
    print(f"Error al cargar el modelo o datos: {e}")
    model = None
    model_wrapper = None
    X_train = None
    feature_names = None

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
            probabilities = torch.softmax(outputs, dim=1)
            predicted_class = torch.argmax(outputs, dim=1).item()
        
        tumor_map = {0: "benign", 1: 'malignant', 2: 'normal'}
        predicted_tumor = tumor_map.get(predicted_class, 'unknown')
        
        PREDICTION_COUNTER.labels(tumor=predicted_tumor).inc()
        
        # Devolver la predicción con probabilidades
        return jsonify({
            "prediction": int(predicted_class), 
            'tumor': predicted_tumor,
            'confidence': float(probabilities[0][predicted_class])
        })
    
    except KeyError:
        return jsonify({"error": "Falta el campo 'features' en el JSON"}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route("/explain", methods=["POST"])
def explain():
    """
    Endpoint para generar explicaciones SHAP de una predicción
    """
    if model is None or model_wrapper is None or X_train is None:
        return jsonify({"error": "Modelo o datos no cargados correctamente"}), 500
    
    try:
        # Obtener los datos de la petición JSON
        data = request.get_json(force=True)
        features = np.array(data["features"], dtype=np.float32).reshape(1, -1)
        
        # Convertir a tensor
        features_tensor = torch.from_numpy(features).to(device)
        
        # Realizar la predicción primero
        with torch.no_grad():
            outputs = model(features_tensor)
            predicted_class = torch.argmax(outputs, dim=1).item()
        
        # Obtener true_label si está disponible, si no usar -1
        true_label = data.get("true_label", -1)
        
        # Generar explicación SHAP
        print(f"Generando explicación SHAP para muestra...")
        explanation_result = explain_single_sample(
            model_wrapper=model_wrapper,
            X_train=X_train,
            sample=features_tensor,
            true_label=true_label if true_label != -1 else predicted_class,
            feature_names=feature_names,
            sample_idx=data.get("sample_idx", 0),
            model_name=MODEL_NAME,
            class_names=["Benign", "Malignant", "Normal"]
        )
        
        # Cargar las imágenes generadas y convertirlas a base64
        sample_idx = data.get("sample_idx", 0)
        prediction_img_path = f"results/explainability_s_tabular/{MODEL_NAME}_sample_{sample_idx}_prediction.png"
        features_img_path = f"results/explainability_s_tabular/{MODEL_NAME}_sample_{sample_idx}_feature_values.png"
        
        images_base64 = {}
        
        if os.path.exists(prediction_img_path):
            with open(prediction_img_path, "rb") as img_file:
                images_base64["prediction_plot"] = base64.b64encode(img_file.read()).decode('utf-8')
        
        if os.path.exists(features_img_path):
            with open(features_img_path, "rb") as img_file:
                images_base64["feature_values_plot"] = base64.b64encode(img_file.read()).decode('utf-8')
        
        # Preparar respuesta - CONVERTIR TODOS LOS VALORES A TIPOS NATIVOS DE PYTHON
        response = {
            "sample_idx": int(explanation_result["sample_idx"]),  # Convertir a int nativo
            "predicted_label": int(explanation_result["predicted_label"]),  # Convertir a int nativo
            "confidence": float(explanation_result["confidence"]),
            "probabilities": {
                "0": float(explanation_result["probabilities"][0]),  # Keys como string
                "1": float(explanation_result["probabilities"][1]),
                "2": float(explanation_result["probabilities"][2])
            },
            "images": images_base64,
            "message": "Explicación generada exitosamente"
        }
        
        if true_label != -1:
            response["true_label"] = int(explanation_result["true_label"])  # Convertir a int nativo
        
        return jsonify(response)
    
    except KeyError as e:
        return jsonify({"error": f"Falta el campo requerido: {str(e)}"}), 400
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        print(f"Error en /explain: {error_trace}")
        return jsonify({"error": str(e), "trace": error_trace}), 500

@app.route("/health", methods=["GET"])
def health():
    """Endpoint para verificar el estado del servicio"""
    status = "healthy" if model is not None else "unhealthy"
    model_loaded = model is not None
    shap_ready = model_wrapper is not None and X_train is not None
    
    return jsonify({
        "status": status,
        "model_loaded": model_loaded,
        "shap_ready": shap_ready,
        "features_count": len(feature_names) if feature_names else 0
    }), 200 if model is not None else 503

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)