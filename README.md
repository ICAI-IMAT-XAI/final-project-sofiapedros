[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/d89f4r04)


# Use
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

# Deploy web
- Build images:
```bash
docker build -t spedros/tumor-api-explain:latest -f Dockerfile .
docker build -t spedros/mlops-tumor-web-explain:latest -f Dockerfile.web .
```
- Deploy:
```bash
docker-compose up -d
```