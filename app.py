# -*- coding: utf-8 -*-
"""
Aplicación de Muestreo y Diseño de Experimentos con Streamlit
"""

import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
import scikit_posthocs as sp

# Configuración general
st.set_page_config(page_title="AgroStat - Muestreo y Diseño de Experimentos", layout="wide")
st.title("🌱 AgroStat: Muestreo y Diseño de Experimentos")

# --- Sidebar: Carga de Datos ---
st.sidebar.header("Opciones de Datos")
uploaded_file = st.sidebar.file_uploader("Sube tu archivo CSV de datos experimentales", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.sidebar.success("Archivo cargado correctamente")

    st.subheader("📌 Vista previa de los datos")
    st.dataframe(df.head())

    # Detectar variables
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    cat_cols = df.select_dtypes(include=['object', 'category']).columns

    st.markdown("**Variables numéricas:** " + ", ".join(numeric_cols))
    st.markdown("**Variables categóricas:** " + ", ".join(cat_cols))

    # Tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🔍 EDA Experimental",
        "📦 Muestreo",
        "🧪 ANOVA",
        "⚖️ Validación de Supuestos",
        "📊 Post-Hoc y Resultados"
    ])

    # ============================
    # TAB 1: ANÁLISIS EXPLORATORIO
    # ============================
    with tab1:
        st.subheader("Análisis Exploratorio de Datos (EDA)")
        st.write(df.describe())

        # Distribución por tratamiento
        if "Tratamiento" in df.columns:
            st.write("Distribución de observaciones por tratamiento:")
            st.bar_chart(df["Tratamiento"].value_counts())

            st.write("Boxplot del peso por tratamiento:")
            fig, ax = plt.subplots(figsize=(8, 5))
            sns.boxplot(data=df, x="Tratamiento", y="Peso (g)", ax=ax, palette="Set2")
            st.pyplot(fig)

            st.write("Media y desviación estándar por tratamiento:")
            resumen = df.groupby("Tratamiento")["Peso (g)"].agg(["mean", "std", "count"])
            st.dataframe(resumen.style.format({"mean": "{:.2f}", "std": "{:.2f}"}))
        else:
            st.warning("No se encontró la columna 'Tratamiento' en los datos.")

    # ============================
    # TAB 2: MUESTREO
    # ============================
    with tab2:
        st.subheader("Muestreo Estadístico")

        st.write("### Tipo de muestreo")
        metodo = st.radio("Selecciona el método de muestreo:", 
                          ["Aleatorio Simple", "Estratificado (por Tratamiento)", "Sistemático"])

        n_muestra = st.slider("Tamaño de la muestra:", 5, len(df), 20)

        if st.button("Generar muestra"):
            if metodo == "Aleatorio Simple":
                muestra = df.sample(n=n_muestra, random_state=42)

            elif metodo == "Estratificado (por Tratamiento)" and "Tratamiento" in df.columns:
                muestra = df.groupby("Tratamiento", group_keys=False).apply(
                    lambda x: x.sample(frac=n_muestra/len(df), random_state=42)
                )
            else:  # sistemático
                paso = len(df)//n_muestra
                muestra = df.iloc[::paso, :]

            st.write(f"Se generó una muestra de {len(muestra)} observaciones:")
            st.dataframe(muestra.head())

            st.write("Estadísticas de la muestra:")
            st.write(muestra.describe())

            # Exportar muestra
            csv = muestra.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Descargar muestra (CSV)",
                data=csv,
                file_name="muestra_experimental.csv",
                mime="text/csv"
            )

    # ============================
    # TAB 3: ANOVA
    # ============================
    with tab3:
        st.subheader("Análisis de Varianza (ANOVA)")

        if "Tratamiento" in df.columns and "Peso (g)" in df.columns:
            modelo = ols('Q("Peso (g)") ~ C(Tratamiento)', data=df).fit()
            anova_results = anova_lm(modelo)
            st.write("### Resultados del ANOVA")
            st.dataframe(anova_results)

            fig, ax = plt.subplots()
            sns.boxplot(x="Tratamiento", y="Peso (g)", data=df, palette="pastel", ax=ax)
            ax.set_title("Distribución de Peso por Tratamiento")
            st.pyplot(fig)
        else:
            st.warning("Debe existir la variable 'Tratamiento' y 'Peso (g)' en el dataset.")

    # ============================
    # TAB 4: VALIDACIÓN DE SUPUESTOS
    # ============================
    with tab4:
        st.subheader("Validación de Supuestos del ANOVA")

        modelo = ols('Q("Peso (g)") ~ C(Tratamiento)', data=df).fit()
        residuos = modelo.resid

        # Normalidad
        st.write("### Prueba de Normalidad (Shapiro-Wilk)")
        shapiro = stats.shapiro(residuos)
        st.write(f"Estadístico W = {shapiro.statistic:.4f}, p-valor = {shapiro.pvalue:.4f}")
        if shapiro.pvalue > 0.05:
            st.success("No se rechaza la normalidad (p > 0.05)")
        else:
            st.warning("Se rechaza la normalidad (p < 0.05)")

        # Homocedasticidad
        st.write("### Prueba de Homogeneidad de Varianzas (Levene)")
        grupos = [g["Peso (g)"].values for _, g in df.groupby("Tratamiento")]
        levene = stats.levene(*grupos)
        st.write(f"Estadístico = {levene.statistic:.4f}, p-valor = {levene.pvalue:.4f}")
        if levene.pvalue > 0.05:
            st.success("Varianzas homogéneas (p > 0.05)")
        else:
            st.warning("Varianzas heterogéneas (p < 0.05)")

        # Q-Q plot
        st.write("### Q-Q Plot de residuos")
        fig = sm.qqplot(residuos, line="s")
        st.pyplot(fig)

    # ============================
    # TAB 5: POST-HOC
    # ============================
    with tab5:
        st.subheader("Pruebas Post-Hoc (Tukey)")
        if "Tratamiento" in df.columns:
            tukey = sp.posthoc_tukey(df, val_col="Peso (g)", group_col="Tratamiento")
            st.write("### Resultados del Test de Tukey")
            st.dataframe(tukey.round(4))

            st.write("### Mapa de Calor de Comparaciones")
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.heatmap(tukey, annot=True, cmap="coolwarm", fmt=".3f", ax=ax)
            st.pyplot(fig)
        else:
            st.warning("No se encontró la columna 'Tratamiento'.")

else:
    st.info("Por favor, sube un archivo CSV para comenzar.")
