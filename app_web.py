import streamlit as st
import requests
import json
import numpy as np
import pandas as pd
import os

st.title('Predicción de Lesiones Mamarias')
st.write('Ingresa las características clínicas y de ultrasonido para obtener una predicción.')

# Configuración con columnas reducidas (7 features totales con label encoding)
ALL_EXPECTED_COLUMNS = [
    'Pixel_size',
    'Halo',
    'Signs',
    'Shape',
    'Margin',
    'Echogenicity',
    'Calcifications'
]

NORMALIZATION_STATS = {
    'Pixel_size': {'mean': 0.007615044713020325, 'std': 0.0016304438468068838},
}

# Mapeos de label encoding (debes ajustar estos valores según tu dataset entrenado)
# Estos valores deben coincidir exactamente con los que generó LabelEncoder durante el entrenamiento
LABEL_ENCODINGS = {
    'Signs': {
        'no': 0, 'palpable': 1, 'nipple retraction': 2, 'breast scar': 3, 
        'skin retraction&palpable': 4, 'redness&warmth': 5, 'warmth&palpable': 6,
        'redness&warmth&palpable': 7, 'nipple retraction&palpable': 8, 
        'palpable&breast scar': 9, 'breast scar&skin retraction': 10,
        'peau d`orange&palpable': 11, 'not available': 12
    },
    'Shape': {
        'oval': 0, 'round': 1, 'irregular': 2, 'not applicable': 3
    },
    'Margin': {
        'circumscribed': 0, 'not applicable': 1, 'not circumscribed - angular': 2,
        'not circumscribed - indistinct': 3, 'not circumscribed - microlobulated': 4,
        'not circumscribed - spiculated': 5, 'not circumscribed - angular&indistinct': 6,
        'not circumscribed - angular&microlobulated': 7,
        'not circumscribed - angular&microlobulated&indistinct': 8,
        'not circumscribed - microlobulated&indistinct': 9,
        'not circumscribed - spiculated&angular': 10,
        'not circumscribed - spiculated&indistinct': 11,
        'not circumscribed - spiculated&angular&indistinct': 12,
        'not circumscribed - spiculated&microlobulated&indistinct': 13,
        'not circumscribed - spiculated&angular&microlobulated&indistinct': 14
    },
    'Echogenicity': {
        'hypoechoic': 0, 'isoechoic': 1, 'hyperechoic': 2, 'anechoic': 3,
        'complex cystic/solid': 4, 'heterogeneous': 5, 'not applicable': 6
    },
    'Calcifications': {
        'no': 0, 'in a mass': 1, 'intraductal': 2, 'indefinable': 3, 'not applicable': 4
    }
}

# Crear dos columnas para organizar mejor los inputs
col1, col2 = st.columns(2)

with col1:
    st.subheader("Datos del Paciente")
    pixel_size = st.number_input('Tamaño de píxel', min_value=0.0, value=0.0076, step=0.001, format="%.4f")
    
    st.subheader("Signos")
    signs = st.selectbox('Signos', [
        'no', 'palpable', 'nipple retraction', 'breast scar', 'skin retraction&palpable',
        'redness&warmth', 'warmth&palpable', 'redness&warmth&palpable', 
        'nipple retraction&palpable', 'palpable&breast scar', 'breast scar&skin retraction',
        'peau d`orange&palpable', 'not available'
    ])

with col2:
    st.subheader("Características de la Lesión")
    shape = st.selectbox('Forma', ['oval', 'round', 'irregular', 'not applicable'])
    
    margin = st.selectbox('Margen', [
        'circumscribed', 'not circumscribed - angular', 'not circumscribed - indistinct',
        'not circumscribed - microlobulated', 'not circumscribed - spiculated',
        'not circumscribed - angular&indistinct', 'not circumscribed - angular&microlobulated',
        'not circumscribed - angular&microlobulated&indistinct',
        'not circumscribed - microlobulated&indistinct',
        'not circumscribed - spiculated&angular', 'not circumscribed - spiculated&indistinct',
        'not circumscribed - spiculated&angular&indistinct',
        'not circumscribed - spiculated&microlobulated&indistinct',
        'not circumscribed - spiculated&angular&microlobulated&indistinct',
        'not applicable'
    ])

st.subheader("Características Adicionales")

col3, col4 = st.columns(2)

with col3:
    echogenicity = st.selectbox('Ecogenicidad', [
        'hypoechoic', 'isoechoic', 'hyperechoic', 'anechoic', 
        'complex cystic/solid', 'heterogeneous', 'not applicable'
    ])
    
    halo = st.selectbox('Halo', ['no (0)', 'yes (1)', 'not applicable (2)'])

with col4:
    calcifications = st.selectbox('Calcificaciones', [
        'no', 'in a mass', 'intraductal', 'indefinable', 'not applicable'
    ])

def process_input_data(data_dict):
    """Procesa los datos de entrada con label encoding (7 features)"""
    
    # Crear array con las 7 features
    features = np.zeros(7, dtype=np.float32)
    
    # 1. Pixel_size (normalizado) - índice 0
    features[0] = (data_dict['Pixel_size'] - NORMALIZATION_STATS['Pixel_size']['mean']) / \
                  (NORMALIZATION_STATS['Pixel_size']['std'] + 1e-6)
    
    # 2. Halo (valores 0, 1, 2) - índice 1
    features[1] = data_dict['Halo']
    
    # 3. Signs (label encoding) - índice 2
    features[2] = LABEL_ENCODINGS['Signs'].get(data_dict['Signs'], -1)
    
    # 4. Shape (label encoding) - índice 3
    features[3] = LABEL_ENCODINGS['Shape'].get(data_dict['Shape'], -1)
    
    # 5. Margin (label encoding) - índice 4
    features[4] = LABEL_ENCODINGS['Margin'].get(data_dict['Margin'], -1)
    
    # 6. Echogenicity (label encoding) - índice 5
    features[5] = LABEL_ENCODINGS['Echogenicity'].get(data_dict['Echogenicity'], -1)
    
    # 7. Calcifications (label encoding) - índice 6
    features[6] = LABEL_ENCODINGS['Calcifications'].get(data_dict['Calcifications'], -1)
    
    return features

# Botón para hacer la predicción
if st.button('🔍 Obtener Predicción', type='primary'):
    # Extraer valor de halo
    halo_value = 0 if 'no' in halo else (1 if 'yes' in halo else 2)
    
    # Crear diccionario con los datos de entrada
    input_data = {
        'Pixel_size': pixel_size,
        'Halo': halo_value,
        'Signs': signs,
        'Shape': shape,
        'Margin': margin,
        'Echogenicity': echogenicity,
        'Calcifications': calcifications,
    }
    
    try:
        features = process_input_data(input_data)
        features_list = features.tolist()
        
        st.info(f"Número de features enviadas: {len(features_list)}")
        
        # Mostrar valores de las features para debugging
        with st.expander("Ver valores de features"):
            feature_names = ALL_EXPECTED_COLUMNS
            for name, value in zip(feature_names, features_list):
                st.write(f"**{name}**: {value:.4f}")
        
        payload = {'features': features_list}
        api_url = os.environ.get('API_URL', 'http://127.0.0.1:5000/predict')
        
        with st.spinner('Realizando predicción...'):
            try:
                response = requests.post(api_url, data=json.dumps(payload),
                                       headers={'Content-Type': 'application/json'}, timeout=10)
                
                if response.status_code == 200:
                    prediction_result = response.json().get('prediction')
                    classification_map = {0: 'Benigno', 1: 'Maligno', 2: 'Normal'}
                    predicted_class = classification_map.get(prediction_result, 'Desconocido')
                    
                    if prediction_result == 0:
                        st.success(f"La predicción es: **{predicted_class}**")
                    elif prediction_result == 1:
                        st.error(f"La predicción es: **{predicted_class}**")
                    else:
                        st.info(f"La predicción es: **{predicted_class}**")
                    
                    st.info("**Nota:** Esta predicción es solo una herramienta de apoyo. Siempre consulte con un profesional médico.")
                else:
                    st.error(f"Error en la petición: {response.status_code} - {response.text}")
                    
            except requests.exceptions.RequestException as e:
                st.error(f"No se pudo conectar con la API.")
                st.error(f"Error: {e}")
                st.info(f"Intentando conectar a: {api_url}")
                
    except Exception as e:
        st.error(f"Error al procesar los datos: {str(e)}")
        st.exception(e)

# Sidebar
with st.sidebar:
    st.header("Información")
    st.write("""
    Esta aplicación utiliza un modelo de aprendizaje automático 
    para predecir la clasificación de lesiones mamarias basándose 
    en características de ultrasonido y datos clínicos.
    
    **Clasificaciones:**
    - 🟢 Benigno
    - 🔴 Maligno
    - 🔵 Normal
    
    **Variables utilizadas (7 features):**
    1. Tamaño de píxel (normalizado)
    2. Halo (0, 1, 2)
    3. Signos clínicos (codificado)
    4. Forma de la lesión (codificado)
    5. Margen de la lesión (codificado)
    6. Ecogenicidad (codificado)
    7. Calcificaciones (codificado)
    """)
    
    st.header("⚙️ Configuración")
    st.write(f"API URL: `{os.environ.get('API_URL', 'http://127.0.0.1:5000/predict')}`")
    st.write(f"Features esperadas: **{len(ALL_EXPECTED_COLUMNS)}**")
    
    if st.button("🔌 Probar Conexión API"):
        api_url = os.environ.get('API_URL', 'http://127.0.0.1:5000/predict')
        try:
            test_payload = {'features': [0.0] * len(ALL_EXPECTED_COLUMNS)}
            response = requests.post(api_url, json=test_payload, timeout=5)
            if response.status_code == 200:
                st.success("Conexión exitosa")
                st.write(f"Predicción de prueba: {response.json()}")
            else:
                st.error(f"Error: {response.status_code}")
        except Exception as e:
            st.error(f"Error: {str(e)}")
    
    st.markdown("---")
    st.caption("Modelo entrenado con drop_specific_columns=True y use_label_encoding=True")
    
    st.markdown("---")