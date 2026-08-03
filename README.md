# Market Research Platform — NLP con BERT + LoRA Adapters

**Proyecto Fin de Grado — Universidad Europea de Madrid, 2025**  
**Autor:** Liam Wittels Beneish · [LinkedIn](https://www.linkedin.com/in/liam-wittels/) · wittelsliam@gmail.com

---

## Qué hace este proyecto

Plataforma de análisis de texto end-to-end basada en **BERT + LoRA Adapters (PEFT)**. Dado un texto, detecta automáticamente si contiene indicios de depresión, lenguaje de odio o si fue generado por IA.

La arquitectura usa un único modelo base (BERT) con tres adaptadores intercambiables, evitando reentrenar el modelo completo para cada tarea.

---

## Resultados

| Tarea | Precision | Recall | F1-Score |
|---|---|---|---|
| Detección de texto generado por IA | 0.87 | 0.86 | 0.86 |
| Hate Speech (Lenguaje de odio) | 0.88 | 0.87 | 0.86 |
| Patrones de depresión | 0.96 | 0.96 | **0.96** |

---

## Arquitectura

```
bert-base-uncased (frozen)
        │
        ├── LoRA Adapter (r=8, α=16) ──► Clasificador: Texto generado por IA
        ├── LoRA Adapter (r=8, α=16) ──► Clasificador: Hate Speech
        └── LoRA Adapter (r=8, α=16) ──► Clasificador: Depresión
```

**Por qué LoRA:** Fine-tuning completo de BERT requiere actualizar ~110M parámetros. Con LoRA solo se entrenan las matrices de bajo rango en las capas de atención (`query` y `value`), reduciendo drásticamente el coste computacional sin pérdida significativa de rendimiento.

---

## Estructura del repositorio

```
├── adapters/
│   ├── depression/
│   │   ├── adapter_config.json
│   │   └── adapter_model.safetensors
│   ├── hate/
│   │   ├── adapter_config.json
│   │   └── adapter_model.safetensors
│   └── generated/
│       ├── adapter_config.json
│       └── adapter_model.safetensors
├── main.py        # Interfaz Streamlit
└── README.md
```

---

## Instalación y uso

```bash
pip install streamlit transformers peft torch pandas
streamlit run main.py
```

La interfaz permite seleccionar el clasificador, introducir un texto y obtener la predicción con nivel de confianza en tiempo real.

---

## Uso de los modelos por código

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel

model = AutoModelForSequenceClassification.from_pretrained(
    "bert-base-uncased", num_labels=2
)
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# Cambia la ruta al adapter que quieras usar
model = PeftModel.from_pretrained(model, "./adapters/depression/")
model.eval()

inputs = tokenizer("I feel like nothing matters anymore",
                   return_tensors="pt", truncation=True, max_length=512)
pred = model(**inputs).logits.argmax(-1).item()
print(pred)  # 0 = non-depression, 1 = depressed
```

---

## Stack tecnológico

- **Modelo:** BERT (bert-base-uncased) + LoRA via PEFT 0.15.2
- **Framework:** PyTorch + HuggingFace Transformers
- **Datos:** Pandas, NumPy, scikit-learn
- **Interfaz:** Streamlit
- **Entrenamiento:** Google Colab
