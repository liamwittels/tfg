import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from peft import PeftModel, PeftConfig

# Función para cargar modelo + tokenizer
@st.cache_resource
def load_adapter(model_name):
    # Identifica el tipo (nombre del folder)
    task_key = model_name.split("/")[-1]

    # Mapeo de etiquetas correcto por tipo de tarea
    label_map = {
        "depression": {0: "non-depression", 1: "depressed"},
        "hate": {0: "non-hateful", 1: "hateful"},
        "generated": {0: "human", 1: "machine"}
    }

    id2label = label_map[task_key]
    label2id = {v: k for k, v in id2label.items()}

    # Carga modelo base con los labels correctos
    base_model = AutoModelForSequenceClassification.from_pretrained(
        "bert-base-uncased",
        num_labels=2,
        id2label=id2label,
        label2id=label2id
    )

    # Aplica el adapter
    model = PeftModel.from_pretrained(base_model, model_name)
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    return model, tokenizer

# Diccionario con rutas de adapters
ADAPTERS = {
    "depression": "adapters/depression",
    "hate": "adapters/hate",
    "generated": "adapters/generated"
}

# Interfaz
st.title("Clasificador de texto con Adapters 🤖")
text_input = st.text_area("Escribe un texto para analizar:", height=150)
option = st.selectbox("Selecciona el modelo a usar:", list(ADAPTERS.keys()))
submit = st.button("Analizar")

if submit:
    if not text_input.strip():
        st.warning("Por favor escribe un texto.")
    else:
        st.info(f"Cargando modelo {option}...")
        model_path = ADAPTERS[option]
        model, tokenizer = load_adapter(model_path)
        pipe = pipeline("text-classification", model=model, tokenizer=tokenizer, return_all_scores=True)
        result = pipe(text_input)[0]

        # Interpretar salida
        pred_label = max(result, key=lambda x: x['score'])['label']
        score = max(result, key=lambda x: x['score'])['score']


        if option == "depression":
            mensaje = "⚠️ Texto depresivo." if pred_label == "depressed" else "✅ No parece depresivo."
        elif option == "hate":
            mensaje = "🚫 Contenido de odio detectado." if pred_label == "hateful" else "✅ Sin hate speech."
        elif option == "generated":
            mensaje = "🤖 Texto generado por IA." if pred_label == "machine" else "🧠 Texto humano."

        st.success(mensaje)
        st.caption(f"Confianza: {score:.2f}")