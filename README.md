# 🫀 ECG Classification – CNN 1D + Self-Training

Este proyecto entrena un modelo para **clasificar señales ECG** (187 puntos) en **5 tipos de latidos**, usando:

* **PyTorch**
* **CNN 1D**
* **Self-training** con pseudo-labels
* **Class weights** para manejar desbalance

Funciona con el dataset semi-supervisado del desafío de **Kaggle**.

---

## Que hace el proyecto

1. Carga los datos etiquetados y no etiquetados.
2. Entrena una **CNN 1D** con los datos etiquetados.
3. Usa el modelo para predecir los labels de los datos sin etiqueta.
4. Selecciona las predicciones de alta confianza (≥ 0.9).
5. Reentrena el modelo sumando esos pseudo-labels.
6. Genera `submission.csv` listo para subir a Kaggle.

---

## Arquitectura del modelo

* **3 capas Conv1D** con BatchNorm + ReLU
* **2 MaxPool** para reducir la secuencia (187 → ~46)
* **Capa densa de 256 neuronas**
* **Dropout 0.3**
* SALIDA: 5 clases

La CNN aprende patrones comunes del ECG como QRS, ondas T, etc.

---

## 🚀 Cómo ejecutarlo

Instala dependencias:

```bash
pip install torch pandas numpy scikit-learn
```

Entrena el modelo:

```bash
python ecg_model.py
```

Genera el archivo de envío:

```bash
python ecg_model.py --generate_submission
```

Obtendrás:

```
submission.csv
```

---

## Resultado

Modelo final:
**F1-macro ≈ 0.87 en Kaggle**

---
