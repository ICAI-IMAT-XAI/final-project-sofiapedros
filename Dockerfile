FROM python:3.11-slim
# Establecer el directorio de trabajo en el contenedor
WORKDIR /app
# Copiar el archivo de requisitos e instalar las dependencias
COPY requirments_docker.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirments_docker.txt

# Copiar los scripts de la aplicación y el modelo
COPY app.py .
COPY models/Tab_model_lr_0.0001_bs_8_hd_128x256_dropout_0.2.pt models/
COPY src/ ./src/
# Exponer el puerto en el que se ejecutará la aplicación
EXPOSE 5000
# Comando para ejecutar la aplicación cuando se inicie el contenedor
CMD ["python", "app.py"]