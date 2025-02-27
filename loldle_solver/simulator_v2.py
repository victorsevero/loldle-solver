import numpy as np
import polars as pl
import streamlit as st

from loldle_solver.solver import Solver

TRANSLATION_MAP = {
    "name": "Nome",
    "gender": "Gênero",
    "position": "Rota",
    "species": "Espécie",
    "resource": "Recurso",
    "range": "Alcance",
    "region": "Região",
    "release": "Lançamento",
}


st.set_page_config(
    page_title="LoLdle Solver",
    page_icon="🧩",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Inicializar solver
if "solver" not in st.session_state:
    st.session_state.solver = Solver()
    st.session_state.displayed_entries = []
    st.session_state.n_possib = len(st.session_state.solver.df)
    st.session_state.n_possib_entries = []
    st.session_state.guesses = st.session_state.solver.get_best_guesses(
        pandas=True
    )

st.title("LoLdle Solver")

# Caixa de entrada
nome_campeao = st.text_input("Digite o nome do campeão:")
if st.button("Adicionar Entrada") and nome_campeao:
    df = st.session_state.solver.df.filter(pl.col("name") == nome_campeao)
    if not df.is_empty():
        row = df.to_pandas().iloc[0].to_dict()
        row = {
            k: "\n".join(v) if isinstance(v, np.ndarray) else v
            for k, v in row.items()
        }
        row = {k: {"value": v, "result": None} for k, v in row.items()}
        st.session_state.displayed_entries.append(row)
        st.session_state.n_possib_entries.append(st.session_state.n_possib)
    else:
        st.warning("Campeão não encontrado!")


# Exibir histórico de entradas
if st.session_state.displayed_entries:
    st.write("### Histórico de entradas")
    for idx, entry in enumerate(st.session_state.displayed_entries):
        cols = st.columns(len(entry))
        for i, (key, value_dict) in enumerate(entry.items()):
            if key == "release":
                display_options = ["⬆️", "✅", "⬇️"]
                result_map = {-1: "⬆️", 0: "✅", 1: "⬇️"}
                reverse_map = {v: k for k, v in result_map.items()}
            else:
                display_options = ["❌", "✅"]
                result_map = {0: "❌", 1: "✅"}
                reverse_map = {v: k for k, v in result_map.items()}

            current_value = result_map.get(
                value_dict["result"], display_options[0]
            )

            selected_value = cols[i].selectbox(
                TRANSLATION_MAP.get(key, key),
                display_options,
                index=display_options.index(current_value),
                key=f"{key}_{idx}",
            )

            st.session_state.displayed_entries[idx][key]["result"] = (
                reverse_map[selected_value]
            )

    if st.button("Enviar Resultados"):
        ultima_entrada = st.session_state.displayed_entries[-1]
        resultado = [
            v["result"] for k, v in ultima_entrada.items() if k != "name"
        ]
        st.session_state.solver.update_df_with_guess(
            ultima_entrada["name"]["value"], resultado
        )
        st.session_state.n_possib = len(st.session_state.solver.df)
        st.session_state.guesses = st.session_state.solver.get_best_guesses(
            pandas=True
        )
        st.rerun()

st.write(f"### Número de possibilidades: {st.session_state.n_possib}")

# Exibir melhores chutes
st.write("### Melhores chutes:")
st.table(st.session_state.guesses)
