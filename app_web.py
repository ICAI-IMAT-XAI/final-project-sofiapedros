import streamlit as st
import requests
import json
import numpy as np
import pandas as pd
import os
import base64
from io import BytesIO

BASE_API_URL = os.environ.get("API_URL", "http://tumor-api:5000")

st.set_page_config(page_title="Predicción de Lesiones Mamarias", layout="wide")

st.title("Predicción de Lesiones Mamarias")
st.write(
    "Ingresa las características clínicas y de ultrasonido para obtener una predicción."
)

# Configuración con columnas reducidas (7 features totales con label encoding)
ALL_EXPECTED_COLUMNS = [
    "Pixel_size",
    "Halo",
    "Signs",
    "Shape",
    "Margin",
    "Echogenicity",
    "Calcifications",
]

NORMALIZATION_STATS = {
    "Pixel_size": {"mean": 0.007615044713020325, "std": 0.0016304438468068838},
}

# Mapeos de label encoding según el LabelEncoder del entrenamiento
LABEL_ENCODINGS = {
    "Signs": {
        "no": 0,
        "palpable": 1,
        "nipple retraction": 2,
        "breast scar": 3,
        "skin retraction&palpable": 4,
        "redness&warmth": 5,
        "warmth&palpable": 6,
        "redness&warmth&palpable": 7,
        "nipple retraction&palpable": 8,
        "palpable&breast scar": 9,
        "breast scar&skin retraction": 10,
        "peau d`orange&palpable": 11,
        "not available": 12,
    },
    "Shape": {"oval": 0, "round": 1, "irregular": 2, "not applicable": 3},
    "Margin": {
        "circumscribed": 0,
        "not applicable": 1,
        "not circumscribed - angular": 2,
        "not circumscribed - indistinct": 3,
        "not circumscribed - microlobulated": 4,
        "not circumscribed - spiculated": 5,
        "not circumscribed - angular&indistinct": 6,
        "not circumscribed - angular&microlobulated": 7,
        "not circumscribed - angular&microlobulated&indistinct": 8,
        "not circumscribed - microlobulated&indistinct": 9,
        "not circumscribed - spiculated&angular": 10,
        "not circumscribed - spiculated&indistinct": 11,
        "not circumscribed - spiculated&angular&indistinct": 12,
        "not circumscribed - spiculated&microlobulated&indistinct": 13,
        "not circumscribed - spiculated&angular&microlobulated&indistinct": 14,
    },
    "Echogenicity": {
        "hypoechoic": 0,
        "isoechoic": 1,
        "hyperechoic": 2,
        "anechoic": 3,
        "complex cystic/solid": 4,
        "heterogeneous": 5,
        "not applicable": 6,
    },
    "Calcifications": {
        "no": 0,
        "in a mass": 1,
        "intraductal": 2,
        "indefinable": 3,
        "not applicable": 4,
    },
}

# Información del modelo
MODEL_NAME = "Tab_model_lr_0.0001_bs_8_hd_(128, 256)_dropout_0.2_encode_True_drop_True_cw_True_1.5"

# Crear dos columnas para organizar mejor los inputs
col1, col2 = st.columns(2)

with col1:
    st.subheader("Datos del Paciente")
    pixel_size = st.number_input(
        "Tamaño de píxel", min_value=0.0, value=0.0076, step=0.001, format="%.4f"
    )

    st.subheader("Signos Clínicos")
    signs = st.selectbox(
        "Signos",
        [
            "no",
            "palpable",
            "nipple retraction",
            "breast scar",
            "skin retraction&palpable",
            "redness&warmth",
            "warmth&palpable",
            "redness&warmth&palpable",
            "nipple retraction&palpable",
            "palpable&breast scar",
            "breast scar&skin retraction",
            "peau d`orange&palpable",
            "not available",
        ],
    )

with col2:
    st.subheader("Características de la Lesión")
    shape = st.selectbox("Forma", ["oval", "round", "irregular", "not applicable"])

    margin = st.selectbox(
        "Margen",
        [
            "circumscribed",
            "not applicable",
            "not circumscribed - angular",
            "not circumscribed - indistinct",
            "not circumscribed - microlobulated",
            "not circumscribed - spiculated",
            "not circumscribed - angular&indistinct",
            "not circumscribed - angular&microlobulated",
            "not circumscribed - angular&microlobulated&indistinct",
            "not circumscribed - microlobulated&indistinct",
            "not circumscribed - spiculated&angular",
            "not circumscribed - spiculated&indistinct",
            "not circumscribed - spiculated&angular&indistinct",
            "not circumscribed - spiculated&microlobulated&indistinct",
            "not circumscribed - spiculated&angular&microlobulated&indistinct",
        ],
    )

st.subheader("Características Adicionales")

col3, col4 = st.columns(2)

with col3:
    echogenicity = st.selectbox(
        "Ecogenicidad",
        [
            "hypoechoic",
            "isoechoic",
            "hyperechoic",
            "anechoic",
            "complex cystic/solid",
            "heterogeneous",
            "not applicable",
        ],
    )

    halo = st.selectbox("Halo", ["no (0)", "yes (1)", "not applicable (2)"])

with col4:
    calcifications = st.selectbox(
        "Calcificaciones",
        ["no", "in a mass", "intraductal", "indefinable", "not applicable"],
    )


def process_input_data(data_dict):
    """Procesa los datos de entrada con label encoding (7 features)"""

    # Crear array con las 7 features
    features = np.zeros(7, dtype=np.float32)

    # 1. Pixel_size (normalizado) - índice 0
    features[0] = (
        data_dict["Pixel_size"] - NORMALIZATION_STATS["Pixel_size"]["mean"]
    ) / (NORMALIZATION_STATS["Pixel_size"]["std"] + 1e-6)

    # 2. Halo (valores 0, 1, 2) - índice 1
    features[1] = data_dict["Halo"]

    # 3. Signs (label encoding) - índice 2
    features[2] = LABEL_ENCODINGS["Signs"].get(data_dict["Signs"], -1)

    # 4. Shape (label encoding) - índice 3
    features[3] = LABEL_ENCODINGS["Shape"].get(data_dict["Shape"], -1)

    # 5. Margin (label encoding) - índice 4
    features[4] = LABEL_ENCODINGS["Margin"].get(data_dict["Margin"], -1)

    # 6. Echogenicity (label encoding) - índice 5
    features[5] = LABEL_ENCODINGS["Echogenicity"].get(data_dict["Echogenicity"], -1)

    # 7. Calcifications (label encoding) - índice 6
    features[6] = LABEL_ENCODINGS["Calcifications"].get(data_dict["Calcifications"], -1)

    return features


# Botones de acción
col_btn1, col_btn2 = st.columns(2)

with col_btn1:
    predict_button = st.button(
        "Obtener Predicción", type="primary", use_container_width=True
    )

with col_btn2:
    explain_button = st.button(
        "Explicar Predicción (SHAP)", type="secondary", use_container_width=True
    )

# Variable para almacenar la predicción
if "last_prediction" not in st.session_state:
    st.session_state.last_prediction = None

# Extraer valor de halo
halo_value = 0 if "no" in halo else (1 if "yes" in halo else 2)

# Crear diccionario con los datos de entrada
input_data = {
    "Pixel_size": pixel_size,
    "Halo": halo_value,
    "Signs": signs,
    "Shape": shape,
    "Margin": margin,
    "Echogenicity": echogenicity,
    "Calcifications": calcifications,
}

# Procesar features
try:
    features = process_input_data(input_data)
    features_list = features.tolist()
except Exception as e:
    st.error(f"Error al procesar los datos: {str(e)}")
    st.stop()

# PREDICCIÓN
if predict_button:
    api_url = f"{BASE_API_URL}/predict"
    st.info(f" Número de features: {len(features_list)}")

    # Mostrar valores de las features para debugging
    with st.expander("Ver valores de features procesadas"):
        feature_names = ALL_EXPECTED_COLUMNS
        df_features = pd.DataFrame(
            {"Feature": feature_names, "Valor": [f"{v:.4f}" for v in features_list]}
        )
        st.dataframe(df_features, use_container_width=True)

    payload = {"features": features_list, "model_name": MODEL_NAME}

    with st.spinner("Realizando predicción..."):
        try:
            response = requests.post(
                api_url,
                data=json.dumps(payload),
                headers={"Content-Type": "application/json"},
                timeout=10,
            )

            if response.status_code == 200:
                result = response.json()
                prediction_result = result.get("prediction")

                # Guardar predicción en session_state
                st.session_state.last_prediction = result

                classification_map = {0: "Benigno", 1: "Maligno", 2: "Normal"}
                predicted_class = classification_map.get(
                    prediction_result, "Desconocido"
                )

                # Mostrar resultado con estilo
                st.markdown("---")
                st.markdown("### Resultado de la Predicción")

                if prediction_result == 0:
                    st.success(f"**{predicted_class}**")
                elif prediction_result == 1:
                    st.error(f"**{predicted_class}**")
                else:
                    st.info(f"**{predicted_class}**")

                if "confidence" in result:
                    st.metric(
                        "Confianza de la predicción", f"{result['confidence']*100:.1f}%"
                    )

                st.markdown("---")
                st.warning(
                    "**Nota Importante:** Esta predicción es solo una herramienta de apoyo diagnóstico. Siempre consulte con un profesional médico cualificado."
                )

            else:
                st.error(f"Error en la petición: {response.status_code}")
                with st.expander("Ver detalles del error"):
                    st.code(response.text)

        except requests.exceptions.Timeout:
            st.error("Tiempo de espera agotado. La API no respondió a tiempo.")
        except requests.exceptions.ConnectionError:
            st.error(
                "No se pudo conectar con la API. Verifica que el servidor esté en ejecución."
            )
            st.info(f"URL intentada: `{api_url}`")
        except requests.exceptions.RequestException as e:
            st.error(f"Error de conexión con la API")
            with st.expander("Ver detalles del error"):
                st.code(str(e))

# EXPLICACIÓN SHAP
if explain_button:
    api_url = f"{BASE_API_URL}/explain"
    st.markdown("---")
    st.markdown("### Generando Explicación SHAP...")

    payload = {
        "features": features_list,
        "sample_idx": 999,  # Índice arbitrario para la muestra del usuario
    }

    with st.spinner("Calculando valores SHAP (esto puede tomar unos segundos)..."):
        try:
            response = requests.post(
                api_url,
                data=json.dumps(payload),
                headers={"Content-Type": "application/json"},
                timeout=60,  # Mayor timeout para SHAP
            )

            if response.status_code == 200:
                result = response.json()

                st.success("Explicación generada exitosamente")

                # Mostrar información básica
                col1_exp, col2_exp = st.columns(2)
                with col1_exp:
                    st.metric(
                        "Clase Predicha",
                        ["Benigno", "Maligno", "Normal"][result["predicted_label"]],
                    )
                with col2_exp:
                    st.metric("Confianza", f"{result['confidence']*100:.1f}%")

                # Mostrar probabilidades
                st.write("**Probabilidades:**")
                probs = result["probabilities"]
                col_a, col_b, col_c = st.columns(3)
                with col_a:
                    st.metric("🟢 Benigno", f"{probs.get('0', 0)*100:.1f}%")
                with col_b:
                    st.metric("🔴 Maligno", f"{probs.get('1', 0)*100:.1f}%")
                with col_c:
                    st.metric("🔵 Normal", f"{probs.get('2', 0)*100:.1f}%")

                # Mostrar imágenes de explicación
                if "images" in result:
                    images = result["images"]

                    st.markdown("---")
                    st.markdown("### Visualizaciones SHAP")

                    if "prediction_plot" in images:
                        st.markdown("#### Distribución de Probabilidades")
                        img_data = base64.b64decode(images["prediction_plot"])
                        st.image(img_data, use_container_width=True)

                    if "feature_values_plot" in images:
                        st.markdown("#### Contribución de Features (SHAP Values)")
                        st.markdown(
                            """
                        Este gráfico muestra las **15 características más importantes** que influyeron en la predicción:
                        - 🔴 **Barras rojas**: Features que aumentan la probabilidad de la clase predicha
                        - 🔵 **Barras azules**: Features que disminuyen la probabilidad de la clase predicha
                        - Los valores SHAP indican la magnitud del impacto de cada feature
                        """
                        )
                        img_data = base64.b64decode(images["feature_values_plot"])
                        st.image(img_data, use_container_width=True)

                st.markdown("---")
                st.info(
                    " **Interpretación:** Los valores SHAP muestran cuánto contribuye cada característica a la predicción final. Valores positivos empujan hacia la clase predicha, valores negativos la alejan."
                )

            else:
                st.error(f" Error al generar explicación: {response.status_code}")
                with st.expander("Ver detalles del error"):
                    st.code(response.text)

        except requests.exceptions.Timeout:
            st.error(
                "Tiempo de espera agotado. El cálculo de SHAP tomó demasiado tiempo."
            )
        except requests.exceptions.ConnectionError:
            st.error(
                "No se pudo conectar con la API. Verifica que el servidor esté en ejecución."
            )
            st.info(f"URL intentada: `{api_url}`")
        except Exception as e:
            st.error(f" Error al generar explicación")
            with st.expander("Ver detalles del error"):
                st.exception(e)

# Sidebar con información
with st.sidebar:
    st.header("Información del Modelo")

    st.markdown(
        f"""
    **Modelo actual:**
    ```
    {MODEL_NAME[:50]}...
    ```
    
    **Configuración:**
    - Label Encoding: Activado
    - Columnas eliminadas: Sí
    - Features totales: **{len(ALL_EXPECTED_COLUMNS)}**
    """
    )

    st.markdown("---")

    st.header("Variables Utilizadas")
    st.markdown(
        """
    1. **Pixel_size** (normalizado)
    2. **Halo** (0: no, 1: yes, 2: n/a)
    3. **Signs** (codificado)
    4. **Shape** (codificado)
    5. **Margin** (codificado)
    6. **Echogenicity** (codificado)
    7. **Calcifications** (codificado)
    
    **Columnas excluidas:**
    - Posterior_features
    - Tissue_composition
    - Skin_thickening
    - Symptoms
    - Age
    """
    )

    st.markdown("---")

    st.header("Clasificaciones")
    st.markdown(
        """
    - 🟢 **Benigno** (0)
    - 🔴 **Maligno** (1)
    - 🔵 **Normal** (2)
    """
    )

    st.markdown("---")

    st.header("Explicabilidad")
    st.markdown(
        """
    El botón **"Explicar Predicción"** utiliza:
    - **SHAP (SHapley Additive exPlanations)**
    - Muestra las features más importantes
    - Visualiza su impacto en la predicción
    """
    )

    st.markdown("---")

    st.header("Configuración API")
    st.code(
        f"URL: {os.environ.get('API_URL', 'http://127.0.0.1:5000')}", language="text"
    )

    if st.button("Probar Conexión"):
        api_url = f"{BASE_API_URL}/health"
        try:
            response = requests.get(api_url, timeout=5)
            if response.status_code == 200:
                health_data = response.json()
                st.success("Conexión exitosa")
                st.json(health_data)
            else:
                st.error(f"Error {response.status_code}")
                st.code(response.text)
        except Exception as e:
            st.error(f"Error: {str(e)}")

    st.markdown("---")
    st.caption("Sistema de apoyo al diagnóstico de lesiones mamarias")
