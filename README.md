[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/d89f4r04)

In this work, I developed a breast-lesion classifier based on features extracted from ultrasound images. The goal of this model is to assist in the early identification of potentially malignant findings without always requiring the immediate intervention of a radiologist, helping reduce workload and optimize medical resources. Additionally, the model’s explanations provide transparency that can support doctors in confirming whether the automated decisions are reasonable or not. This system is motivated by the need for more accessible diagnostic support tools and benefits stakeholders such as radiologists, medical institutions, and ultimately patients seeking timely and reliable assessments.

# Use
- Install requirements
```bash
pip install -r requirements.txt
```

- Train: 
```bash
python -m src.train_tabular
```
- Eval:
```bash
python -m src.eval_tabular
```
- Explain global
```bash
python -m src.explain_tabulars
```
- Explain sample
```bash
python -m src.explain_sample_tabular
```
- Sanity checks:
```bash
python -m src.sanity_checks_tabular
```

# Dataset
Available in: https://www.cancerimagingarchive.net/collection/breast-lesions-usg/

# Explanations
In notebooks/tabular_model.ipynb

Report available in Report.pdf

# Deploy web
This web allows the user to classify samples based on tabular data.

- Deploy:
```bash
docker compose up -d
```
In a web browser open: http://localhost:8501

Note: The Docker images are already uploaded to Docker Hub, so building them locally is usually not necessary.
- If you want to modify or rebuild the images, you can run:
```bash
docker build -t spedros/tumor-api-explain:latest -f Dockerfile .
docker build -t spedros/mlops-tumor-web-explain:latest -f Dockerfile.web .
```