
import os

import joblib
import pandas as pd
import streamlit as st

ARQUIVO_MODELO = "modelo_desempenho.pkl"


@st.cache_resource
def carregar_modelo(caminho_modelo):
    """Carregar o modelo do arquivo .pkl"""

    if not os.path.exists(caminho_modelo):
        st.error("Arquivo do modelo não encontrado!")
        return None, None

    try:
        modelo = joblib.load(caminho_modelo)
        classes_modelo = modelo.classes_
        return modelo, classes_modelo

    except Exception as e:
        st.error(f"Erro ao carregar o modelo: {e}")
        return None, None


def main():
    """Função principal que executa o app Streamlit"""

    modelo, classes_modelo = carregar_modelo(ARQUIVO_MODELO)

    st.title("Projeto 1 - Classificação Késimo Vizinho")
    st.subheader("Estudo de caso")

    if modelo is None:
        st.error(f"Arquivo do modelo ('{ARQUIVO_MODELO}') não encontrado.")
        st.warning("Execute o script 'python treinamento.py' no terminal para criar o modelo.")
        st.stop()

    st.sidebar.header("Insira os dados do aluno:")

    faltas = st.sidebar.slider(
        "Informe a quantidade de faltas:",
        min_value=0,
        max_value=10,
        value=0
    )

    nota_1 = st.sidebar.number_input(
        "Nota da primeira prova (0-10)",
        min_value=0.0,
        max_value=10.0,
        value=0.0
    )

    nota_2 = st.sidebar.number_input(
        "Nota da segunda prova (0-10)",
        min_value=0.0,
        max_value=10.0,
        value=0.0
    )

    if st.sidebar.button('Situação do aluno'):

        dados_entrada = pd.DataFrame(
            [[faltas, nota_1, nota_2]],
            columns=["faltas", "nota_1", "nota_2"]
        )

        st.write("Dados de Entrada:")
        st.dataframe(dados_entrada)

        probabilidades = modelo.predict_proba(dados_entrada)
        previsao = modelo.predict(dados_entrada)[0]

        st.header(f"Resultado da previsão: {previsao}")

        if previsao == "aprovado":
            st.success("Este aluno tem alta probabilidade de ser aprovado.")
        else:
            st.error("Este aluno está em zona de risco.")

        st.write("---Análise de confiança de IA---")

        prob_aprovado = probabilidades[0][list(classes_modelo).index("aprovado")]
        prob_reprovado = probabilidades[0][list(classes_modelo).index("reprovado")]

        col1, col2 = st.columns(2)

        col1.metric("Confiança em 'aprovado'", f"{prob_aprovado*100:.2f}%")
        col2.metric("Confiança em 'reprovado'", f"{prob_reprovado*100:.2f}%")


if __name__ == "__main__":
    main()
