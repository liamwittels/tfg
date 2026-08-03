# Market Research Platform — NLP con BERT + LoRA Adapters

**Proyecto Fin de Grado (TFG) — Universidad Europea de Madrid, 2025**  
**Autor:** Liam Wittels Beneish

---

## ¿Qué es esto?

Plataforma end-to-end de análisis de texto basada en **BERT (bert-base-uncased)** + **LoRA Adapters** (PEFT). Permite clasificar texto en 3 tareas distintas usando el mismo modelo base con adaptadores intercambiables, sin necesidad de reentrenar BERT completo.

## Demo

![Interfaz Streamlit](https://i.imgur.com/placeholder.png)

Selecciona una tarea, introduce un texto y obtén la predicción con nivel de confianza en tiempo real.

---

## Resultados

| Clasificador | Precision | Recall | F1-Score |
|---|---|---|---|
| Texto generado por IA | 0.87 | 0.86 | 0.86 |
| Hate Speech (Discurso de odio) | 0.88 | 0.87 | 0.86 |
| Patrones de depresión | 0.96 | 0.96 | **0.96** |

---

## Arquitectura

```
bert-base-uncased (base model, frozen)
        │
        ├── LoRA Adapter (r=8, α=16) → Clasificador: Texto generado por IA
        ├── LoRA Adapter (r=8, α=16) → Clasificador: Hate Speech  
        └── LoRA Adapter (r=8, α=16) → Clasificador: Depresión
```

**¿Por qué LoRA?** Fine-tuning completo de BERT = ~440MB por tarea. Con LoRA cada adapter pesa pocos MB, comparten el mismo base model y no hay interferencia entre tareas.

**Configuración LoRA:**
- Rank (r): 8 | Alpha: 16 | Dropout: 0.1
- Target modules: `query`, `value` (capas de atención)
- Task: Sequence Classification (SEQ_CLS)
- Librería: PEFT 0.15.2

---

## Estructura del proyecto

```
market-research-nlp/
│
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
│
├── main2.py          # Interfaz Streamlit
└── README.md
```

---

## Instalación y uso

### 1. Instalar dependencias

```bash
pip install streamlit transformers peft torch pandas
```

### 2. Lanzar la interfaz

```bash
streamlit run main2.py
```

### 3. Usar los modelos directamente

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel

# Cargar modelo base
model = AutoModelForSequenceClassification.from_pretrained(
    "bert-base-uncased", num_labels=2
)
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# Cargar adapter (depression / hate / generated)
model = PeftModel.from_pretrained(model, "./adapters/depression/")
model.eval()

# Inferencia
text = "I feel like nothing matters anymore"
inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
outputs = model(**inputs)
pred = outputs.logits.argmax(-1).item()
print("Predicción:", pred)  # 0=non-depression, 1=depressed
```

---

## Stack tecnológico

| Categoría | Tecnología |
|---|---|
| Modelo base | BERT (bert-base-uncased, HuggingFace) |
| Fine-tuning | LoRA via PEFT |
| Framework | PyTorch + HuggingFace Transformers |
| Procesado de datos | Pandas, NumPy, scikit-learn |
| Interfaz | Streamlit |
| Entorno de entrenamiento | Google Colab |

---

## Contacto

**Liam Wittels Beneish**  
Ingeniero Informático — Universidad Europea de Madrid  
[linkedin.com/in/liam-wittels](https://www.linkedin.com/in/liam-wittels/) | wittelsliam@gmail.com
