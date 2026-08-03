import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from peft import PeftModel
import pandas as pd

###############################
# CONFIGURACIÓN DE LA PÁGINA #
###############################

st.set_page_config(
    page_title="Clasificador de Texto con Adapters",
    page_icon="🤖",
    layout="centered",
)

# Paleta de colores personalizada (opcional, Streamlit 1.32+)
# Puedes adaptar estos colores a la guía de estilo de tu marca
CUSTOM_THEME = {
    "primary": "#4F8BF9",
    "secondaryBackground": "#F5F7FA",
    "textColor": "#0F1116",
    "success": "#2BC48A",
    "warning": "#FFAF38",
    "error": "#E04F5F",
}

for k, v in CUSTOM_THEME.items():
    st._config.set_option(f"theme.{k}", v)

#################################
# FUNCIÓN PARA CARGAR EL ADAPTER #
#################################

@st.cache_resource(show_spinner=False)
def load_adapter(model_name: str):
    """Carga un modelo + tokenizer y aplica el adapter correspondiente."""

    task_key = model_name.split("/")[-1]

    label_map = {
        "depression": {0: "non-depression", 1: "depressed"},
        "hate": {0: "non-hateful", 1: "hateful"},
        "generated": {0: "human", 1: "machine"},
    }
    id2label = label_map[task_key]
    label2id = {v: k for k, v in id2label.items()}

    base_model = AutoModelForSequenceClassification.from_pretrained(
        "bert-base-uncased",
        num_labels=2,
        id2label=id2label,
        label2id=label2id,
    )

    model = PeftModel.from_pretrained(base_model, model_name)
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    return model, tokenizer

###############################
# RUTAS DE LOS MODELOS/ADAPTERS
###############################

ADAPTERS = {
    "Depresión": "adapters/depression",
    "Discurso de odio": "adapters/hate",
    "Texto generado": "adapters/generated",
}

#################################
# SIDEBAR: controles y ayuda     #
#################################

with st.sidebar:
    st.header("⚙️ Configuración")
    option = st.selectbox("Modelo a utilizar", ADAPTERS.keys())
    st.markdown("---")
    st.write("### ℹ️ Sobre la app")
    st.write(
        "Esta demo muestra cómo usar **PEFT Adapters** para clasificar texto en diferentes tareas. "
        "Escribe una frase, selecciona la tarea y obtén una predicción al instante."
    )

###############################
# CUERPO PRINCIPAL            #
###############################

st.markdown("## 📝 Analiza tu texto")
text_input = st.text_area("Introduce el texto a analizar", height=180, placeholder="Escribe aquí…")

col1, col2 = st.columns([1, 1])
with col1:
    analyze_btn = st.button("🔍 Analizar", type="primary")
with col2:
    clear_btn = st.button("🗑️ Limpiar")

if clear_btn:
    st.experimental_rerun()

if analyze_btn:
    if not text_input.strip():
        st.warning("⚠️ Por favor introduce un texto antes de analizar.")
        st.stop()

    model_path = ADAPTERS[option]

    with st.spinner(f"Cargando modelo **{option}**…"):
        model, tokenizer = load_adapter(model_path)
        pipe = pipeline(
            "text-classification",
            model=model,
            tokenizer=tokenizer,
            return_all_scores=True,
        )

    result = pipe(text_input)[0]
    df_scores = pd.DataFrame(result)
    df_scores["percentage"] = df_scores["score"].apply(lambda x: round(x * 100, 2))

    pred = df_scores.iloc[df_scores["score"].idxmax()]
    pred_label = pred["label"]
    score = pred["percentage"]

    ###############################
    # VISUALIZACIÓN DE RESULTADOS #
    ###############################

    # Cabecera con icono según tarea
    task_icons = {
        "Depresión": "🩺",
        "Discurso de odio": "🚫",
        "Texto generado": "🤖",
    }

    st.markdown(
        f"### {task_icons[option]} Resultado: **{pred_label.replace('-', ' ').title()}**"
    )

    # Métrica de confianza
    st.metric("Confianza", f"{score}%")

    # Barra de progreso estilizada
    st.progress(score / 100)

    # Gráfico de barras con las puntuaciones
    st.bar_chart(df_scores.set_index("label")["score"])

    # Mensaje de advertencia / info contextual
    messages = {
        "Depresión": {
            "depressed": "⚠️ El texto presenta indicios de depresión. Recomendamos precaución al interpretarlo.",
            "non-depression": "✅ No se detectan señales de depresión en el texto.",
        },
        "Discurso de odio": {
            "hateful": "🚫 Se ha detectado contenido de odio en el texto.",
            "non-hateful": "✅ No se ha detectado lenguaje de odio.",
        },
        "Texto generado": {
            "machine": "🤖 El texto parece haber sido generado por IA.",
            "human": "🧠 El texto parece haber sido escrito por una persona.",
        },
    }

    st.info(messages[option][pred_label])

    with st.expander("ℹ️ Detalles de la clasificación"):
        st.write(df_scores)

###############################
# FOOTER                      #
###############################

st.markdown("---")

st.caption(
    "Plataforma de market research para la ayuda a la toma de decisiones"
)