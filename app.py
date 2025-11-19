# -*- coding: utf-8 -*-
"""
Aplicación de Muestreo y Diseño de Experimentos - Streamlit
Versión compatible con Streamlit Cloud
"""

import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy.stats import shapiro, levene, f_oneway, kruskal
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.power import FTestAnovaPower
from statsmodels.formula.api import ols
import statsmodels.api as sm
import warnings
warnings.filterwarnings('ignore')

# Configuración de la página
st.set_page_config(
    page_title="Diseño de Experimentos - Aguacates",
    page_icon="🥑",
    layout="wide"
)

# CSS personalizado
st.markdown("""
    <style>
    .stApp { background-color: #f5f5f5; }
    h1 { color: #2E8B57; text-align: center; }
    .design-card { 
        border-radius: 10px; 
        padding: 15px; 
        margin: 10px 0; 
        border-left: 5px solid #2E8B57; 
        background-color: white;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .assumption-check { 
        padding: 10px; 
        margin: 5px 0; 
        border-radius: 5px;
    }
    .valid { background-color: #d4edda; border-left: 4px solid #28a745; }
    .invalid { background-color: #f8d7da; border-left: 4px solid #dc3545; }
    </style>
    """, unsafe_allow_html=True)

# Título principal
st.title("🥑 Diseño de Experimentos - Bioestimulantes en Aguacates")
st.markdown("---")

# Función para generar datos de ejemplo
def generate_sample_data():
    """Genera datos de ejemplo para la demostración"""
    np.random.seed(42)
    
    tratamientos = ['Control', 'Aminoacidos', 'Algas', 'Humicos']
    n_arboles_por_tratamiento = 10
    n_frutas_por_arbol = 5
    
    datos = []
    arbol_id = 1
    
    # Medias para cada tratamiento
    medias = {'Control': 180, 'Aminoacidos': 185, 'Algas': 195, 'Humicos': 190}
    stds = {'Control': 8, 'Aminoacidos': 8, 'Algas': 9, 'Humicos': 8}
    
    for tratamiento in tratamientos:
        for _ in range(n_arboles_por_tratamiento):
            # Efecto aleatorio del árbol
            efecto_arbol = np.random.normal(0, 3)
            for fruta_id in range(1, n_frutas_por_arbol + 1):
                peso = np.random.normal(medias[tratamiento] + efecto_arbol, stds[tratamiento])
                datos.append({
                    'Arbol_ID': f'Arbol_{arbol_id}',
                    'Tratamiento': tratamiento,
                    'Fruta_ID': fruta_id,
                    'Peso_g': round(max(peso, 150), 2)  # Mínimo realista
                })
            arbol_id += 1
    
    return pd.DataFrame(datos)

# Cargar datos
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('datos_aguacate_masivos.csv')
        st.sidebar.success("✅ Datos cargados desde archivo")
        return df
    except:
        df = generate_sample_data()
        st.sidebar.info("📊 Usando datos de ejemplo. Sube tu archivo 'datos_aguacate_masivos.csv'")
        return df

# Sidebar para navegación
st.sidebar.header("Módulos de Análisis")
app_mode = st.sidebar.selectbox(
    "Selecciona el módulo",
    [
        "📋 Contexto & Datos", 
        "🔍 Análisis Exploratorio", 
        "📊 ANOVA & Comparaciones",
        "✅ Validación Supuestos",
        "⚡ Potencia Estadística"
    ]
)

# Upload de archivo en sidebar
st.sidebar.header("📁 Cargar Datos")
uploaded_file = st.sidebar.file_uploader("Sube tu archivo CSV", type=["csv"])

if uploaded_file is not None:
    try:
        df_uploaded = pd.read_csv(uploaded_file)
        st.session_state.df = df_uploaded
        st.sidebar.success("✅ Archivo cargado exitosamente!")
    except Exception as e:
        st.sidebar.error(f"❌ Error al cargar archivo: {e}")

# Cargar datos (usar session_state para mantener entre reruns)
if 'df' not in st.session_state:
    st.session_state.df = load_data()

df = st.session_state.df

# Módulo 1: Contexto & Datos
if app_mode == "📋 Contexto & Datos":
    st.header("📋 Contexto del Proyecto")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ### 🥑 Evaluación de Bioestimulantes en Aguacates Hass
        
        **Objetivo del Experimento:**
        Evaluar el efecto de 4 formulaciones de bioestimulantes sobre el peso del aguacate Hass.
        
        **Diseño Experimental:**
        - Completamente aleatorizado (CRD)
        - 4 tratamientos
        - 10 árboles por tratamiento
        - 5 frutas medidas por árbol
        
        **Hipótesis:**
        - H₀: No hay diferencia en el peso promedio entre tratamientos
        - H₁: Al menos un tratamiento difiere en el peso promedio
        """)
    
    with col2:
        st.markdown("""
        ### 🎯 Características del Diseño
        
        | Parámetro | Valor |
        |-----------|-------|
        | Tratamientos | 4 |
        | Árboles/Tratamiento | 10 |
        | Frutas/Árbol | 5 |
        | Total Observaciones | 200 |
        """)
        
        st.metric("📊 Total de Árboles", df['Arbol_ID'].nunique())
        st.metric("🔢 Observaciones", len(df))
        st.metric("📈 Tratamientos", df['Tratamiento'].nunique())
    
    st.markdown("---")
    
    # Descripción de tratamientos
    st.subheader("🧪 Descripción de Tratamientos")
    
    tratamientos_info = {
        'Tratamiento': ['Control', 'Aminoacidos', 'Algas', 'Humicos'],
        'Descripción': [
            'Manejo estándar sin bioestimulantes',
            'Hidrolizado de proteína para síntesis celular eficiente',
            'Ascophyllum nodosum para división celular acelerada',
            'Mejoradores de suelo y absorción de nutrientes'
        ]
    }
    
    df_tratamientos = pd.DataFrame(tratamientos_info)
    st.dataframe(df_tratamientos, use_container_width=True)
    
    # Mostrar datos
    st.markdown("---")
    st.subheader("📁 Datos del Experimento")
    
    # Estadísticas rápidas
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Muestra Total", len(df))
    
    with col2:
        st.metric("Árboles Únicos", df['Arbol_ID'].nunique())
    
    with col3:
        st.metric("Tratamientos", df['Tratamiento'].nunique())
    
    with col4:
        st.metric("Peso Promedio", f"{df['Peso_g'].mean():.1f} g")
    
    # Vista de datos
    st.subheader("👀 Vista Previa de Datos")
    st.dataframe(df.head(10))
    
    # Estructura de datos
    with st.expander("📏 Estructura del Dataset"):
        st.write("**Variables:**")
        for col in df.columns:
            st.write(f"- {col}: {df[col].dtype}")
        
        st.write("**Balance del Diseño:**")
        balance = df.groupby('Tratamiento')['Arbol_ID'].nunique()
        st.write(balance)

# Módulo 2: Análisis Exploratorio
elif app_mode == "🔍 Análisis Exploratorio":
    st.header("🔍 Análisis Exploratorio de Datos")
    
    # Selección de tipo de análisis
    analisis_type = st.selectbox(
        "Tipo de análisis exploratorio",
        [
            "📈 Distribución por Tratamiento",
            "📊 Comparación de Medias", 
            "📦 Boxplots Comparativos"
        ]
    )
    
    if analisis_type == "📈 Distribución por Tratamiento":
        st.subheader("Distribución del Peso por Tratamiento")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Histogramas
            fig, ax = plt.subplots(figsize=(10, 6))
            for tratamiento in df['Tratamiento'].unique():
                data = df[df['Tratamiento'] == tratamiento]['Peso_g']
                ax.hist(data, alpha=0.6, label=tratamiento, bins=15, density=True)
            ax.set_xlabel('Peso (g)')
            ax.set_ylabel('Densidad')
            ax.set_title('Distribución de Pesos por Tratamiento')
            ax.legend()
            st.pyplot(fig)
        
        with col2:
            # Gráfico de densidad
            fig, ax = plt.subplots(figsize=(10, 6))
            for tratamiento in df['Tratamiento'].unique():
                data = df[df['Tratamiento'] == tratamiento]['Peso_g']
                sns.kdeplot(data, ax=ax, label=tratamiento, fill=True, alpha=0.5)
            ax.set_xlabel('Peso (g)')
            ax.set_ylabel('Densidad')
            ax.set_title('Densidad de Pesos por Tratamiento')
            ax.legend()
            st.pyplot(fig)
            
    elif analisis_type == "📊 Comparación de Medias":
        st.subheader("Comparación de Medias y Variabilidad")
        
        # Estadísticas descriptivas por tratamiento
        stats_tratamiento = df.groupby('Tratamiento')['Peso_g'].agg([
            'count', 'mean', 'std', 'sem', 'min', 'max', 'median'
        ]).round(2)
        
        stats_tratamiento.columns = ['N', 'Media', 'Desv. Std', 'Error Std', 'Mín', 'Máx', 'Mediana']
        st.dataframe(stats_tratamiento)
        
        # Gráfico de medias con intervalos de confianza
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.pointplot(data=df, x='Tratamiento', y='Peso_g', 
                     capsize=0.1, errwidth=1.5, ax=ax)
        ax.set_title('Medias de Peso por Tratamiento con Intervalos de Confianza (95%)')
        ax.set_ylabel('Peso (g)')
        ax.tick_params(axis='x', rotation=45)
        st.pyplot(fig)
        
    elif analisis_type == "📦 Boxplots Comparativos":
        st.subheader("Comparación Visual entre Tratamientos")
        
        fig, ax = plt.subplots(figsize=(12, 8))
        sns.boxplot(data=df, x='Tratamiento', y='Peso_g', ax=ax)
        sns.stripplot(data=df, x='Tratamiento', y='Peso_g', 
                     color='black', alpha=0.5, size=3, ax=ax)
        ax.set_title('Distribución de Pesos por Tratamiento')
        ax.set_ylabel('Peso (g)')
        ax.tick_params(axis='x', rotation=45)
        st.pyplot(fig)
        
        # ANOVA simple para preview
        st.subheader("🔍 Preview ANOVA")
        grupos = [df[df['Tratamiento'] == tr]['Peso_g'].values for tr in df['Tratamiento'].unique()]
        f_stat, p_value = f_oneway(*grupos)
        
        col1, col2 = st.columns(2)
        col1.metric("Estadístico F", f"{f_stat:.4f}")
        col2.metric("Valor p", f"{p_value:.4f}")

# Módulo 3: ANOVA & Comparaciones
elif app_mode == "📊 ANOVA & Comparaciones":
    st.header("📊 Análisis de Varianza (ANOVA) y Comparaciones Múltiples")
    
    # Realizar ANOVA
    st.subheader("🔍 Análisis de Varianza (ANOVA)")
    
    # Usando statsmodels para ANOVA detallada
    model = ols('Peso_g ~ C(Tratamiento)', data=df).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)
    
    st.dataframe(anova_table.style.format("{:.4f}"))
    
    # Interpretación de resultados
    p_value = anova_table['PR(>F)'][0]
    f_value = anova_table['F'][0]
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Estadístico F", f"{f_value:.4f}")
    col2.metric("Valor p", f"{p_value:.4f}")
    
    # Tamaño del efecto
    ss_tratamiento = anova_table['sum_sq'][0]
    ss_total = anova_table['sum_sq'].sum()
    eta_squared = ss_tratamiento / ss_total
    col3.metric("η² (Eta cuadrado)", f"{eta_squared:.4f}")
    
    # Interpretación
    if p_value < 0.05:
        st.success("""
        ✅ **Resultado Significativo:** Se rechaza H₀. 
        Existen diferencias estadísticamente significativas entre al menos dos tratamientos.
        """)
        
        # Test de Tukey HSD
        st.subheader("🔬 Comparaciones Múltiples - Test de Tukey HSD")
        
        tukey = pairwise_tukeyhsd(
            endog=df['Peso_g'],
            groups=df['Tratamiento'],
            alpha=0.05
        )
        
        # Resultados en tabla
        tukey_df = pd.DataFrame(data=tukey._results_table.data[1:], 
                              columns=tukey._results_table.data[0])
        st.dataframe(tukey_df)
        
        # Resumen de diferencias
        st.subheader("📋 Resumen de Diferencias Significativas")
        
        diferencias = tukey_df[tukey_df['reject'] == True]
        if len(diferencias) > 0:
            for _, row in diferencias.iterrows():
                st.info(f"**{row['group1']} vs {row['group2']}:** "
                       f"diferencia = {row['meandiff']:.2f}g, "
                       f"p = {row['p-adj']:.4f}")
        else:
            st.warning("No hay diferencias significativas entre pares de tratamientos")
            
    else:
        st.warning("""
        ❌ **Resultado No Significativo:** No se rechaza H₀.
        No hay evidencia suficiente de diferencias entre los tratamientos.
        """)
    
    # Gráfico de medias
    st.subheader("📈 Medias por Tratamiento")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    medias = df.groupby('Tratamiento')['Peso_g'].mean()
    errores = df.groupby('Tratamiento')['Peso_g'].sem()
    
    bars = ax.bar(medias.index, medias.values, 
                 yerr=errores.values, capsize=5, alpha=0.7,
                 color=['skyblue', 'lightcoral', 'lightgreen', 'gold'])
    
    ax.set_title('Medias de Peso por Tratamiento')
    ax.set_ylabel('Peso (g)')
    ax.tick_params(axis='x', rotation=45)
    
    # Añadir valores en las barras
    for bar, media in zip(bars, medias.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
               f'{media:.1f}g', ha='center', va='bottom')
    
    st.pyplot(fig)

# Módulo 4: Validación Supuestos
elif app_mode == "✅ Validación Supuestos":
    st.header("✅ Validación de Supuestos del ANOVA")
    
    st.markdown("""
    ### 📋 Supuestos del ANOVA
    
    1. **Normalidad:** Los residuos deben distribuirse normalmente
    2. **Homocedasticidad:** Varianzas iguales entre grupos  
    3. **Independencia:** Observaciones independientes entre sí
    """)
    
    # Preparar datos para análisis de supuestos
    grupos = [df[df['Tratamiento'] == tr]['Peso_g'].values for tr in df['Tratamiento'].unique()]
    
    # Calcular residuos
    residuos = []
    for tr in df['Tratamiento'].unique():
        media_tr = df[df['Tratamiento'] == tr]['Peso_g'].mean()
        residuos_tr = df[df['Tratamiento'] == tr]['Peso_g'] - media_tr
        residuos.extend(residuos_tr)
    
    residuos = np.array(residuos)
    
    # 1. Normalidad
    st.subheader("1. 📊 Normalidad de los Residuos")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Test de Shapiro-Wilk
        stat_sw, p_sw = shapiro(residuos)
        st.metric("Shapiro-Wilk p-value", f"{p_sw:.4f}")
        
        if p_sw > 0.05:
            st.markdown('<div class="assumption-check valid">✅ No se rechaza normalidad (p > 0.05)</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="assumption-check invalid">❌ Se rechaza normalidad (p ≤ 0.05)</div>', unsafe_allow_html=True)
    
    with col2:
        # QQ-Plot
        fig, ax = plt.subplots(figsize=(8, 6))
        stats.probplot(residuos, dist="norm", plot=ax)
        ax.set_title('Q-Q Plot para Normalidad de Residuos')
        st.pyplot(fig)
    
    # 2. Homocedasticidad
    st.subheader("2. 📏 Homocedasticidad (Igualdad de Varianzas)")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Test de Levene
        stat_lev, p_lev = levene(*grupos)
        st.metric("Levene p-value", f"{p_lev:.4f}")
        
        if p_lev > 0.05:
            st.markdown('<div class="assumption-check valid">✅ No se rechaza homocedasticidad (p > 0.05)</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="assumption-check invalid">❌ Se rechaza homocedasticidad (p ≤ 0.05)</div>', unsafe_allow_html=True)
    
    with col2:
        # Gráfico de varianzas
        varianzas = [np.var(grupo) for grupo in grupos]
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.bar(df['Tratamiento'].unique(), varianzas, alpha=0.7)
        ax.set_title('Varianzas por Tratamiento')
        ax.set_ylabel('Varianza')
        ax.tick_params(axis='x', rotation=45)
        st.pyplot(fig)
    
    # Resumen de validación
    st.subheader("📋 Resumen de Validación de Supuestos")
    
    supuestos_validos = all([p_sw > 0.05, p_lev > 0.05])
    
    if supuestos_validos:
        st.success("""
        ✅ **Todos los supuestos se cumplen.** 
        El análisis ANOVA es válido y las conclusiones son confiables.
        """)
    else:
        st.warning("""
        ⚠️ **Algunos supuestos no se cumplen.**
        Considera:
        - Transformaciones de datos (log, sqrt)
        - Tests no paramétricos (Kruskal-Wallis)
        """)

# Módulo 5: Potencia Estadística
elif app_mode == "⚡ Potencia Estadística":
    st.header("⚡ Análisis de Potencia Estadística")
    
    st.markdown("""
    ### 💪 ¿Qué es la Potencia Estadística?
    
    La potencia (1-β) es la probabilidad de detectar un efecto cuando realmente existe.
    - **Potencia alta (> 0.8):** Buena capacidad para detectar efectos
    - **Potencia baja:** Riesgo de Error Tipo II (no detectar efecto real)
    """)
    
    # Calculadora de potencia
    st.subheader("🧮 Calculadora de Potencia")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        efecto = st.selectbox(
            "Tamaño del efecto (f)",
            ["Pequeño (0.1)", "Mediano (0.25)", "Grande (0.4)", "Personalizado"],
            index=1
        )
        
        if efecto == "Pequeño (0.1)":
            f_effect = 0.1
        elif efecto == "Mediano (0.25)":
            f_effect = 0.25
        elif efecto == "Grande (0.4)":
            f_effect = 0.4
        else:
            f_effect = st.number_input("f personalizado", 0.01, 1.0, 0.25, 0.01)
    
    with col2:
        alpha = st.slider("Nivel α", 0.01, 0.10, 0.05, 0.01)
        potencia_deseada = st.slider("Potencia deseada", 0.70, 0.99, 0.80, 0.05)
    
    with col3:
        k_grupos = st.number_input("Número de grupos", 2, 10, 4)
        n_grupo = st.number_input("Tamaño por grupo", 2, 100, 10)
    
    # Cálculos de potencia
    power_analysis = FTestAnovaPower()
    
    # Potencia con parámetros actuales
    potencia_actual = power_analysis.power(f_effect, n_grupo, k_grupos, alpha)
    
    # Tamaño muestral requerido
    n_requerido = power_analysis.solve_power(
        effect_size=f_effect,
        power=potencia_deseada,
        nobs=None,
        alpha=alpha,
        k_groups=k_grupos
    )
    
    # Mostrar resultados
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Potencia Actual", f"{potencia_actual:.3f}")
        if potencia_actual >= 0.8:
            st.success("✅ Potencia adecuada")
        else:
            st.warning("⚠️ Potencia insuficiente")
    
    with col2:
        st.metric("Tamaño Muestral Requerido", f"{np.ceil(n_requerido):.0f} por grupo")
        if n_grupo >= n_requerido:
            st.success("✅ Tamaño muestral adecuado")
        else:
            st.warning("⚠️ Se necesitan más réplicas")
    
    # Gráfico de curva de potencia
    st.subheader("📈 Curva de Potencia")
    
    n_range = np.arange(2, 101, 2)
    power_curve = power_analysis.power(f_effect, n_range, k_grupos, alpha)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(n_range, power_curve, linewidth=2, label=f'f = {f_effect}')
    ax.axhline(y=potencia_deseada, color='red', linestyle='--', alpha=0.7, 
               label=f'Potencia deseada ({potencia_deseada})')
    ax.axvline(x=n_requerido, color='green', linestyle='--', alpha=0.7, 
               label=f'n requerido ({np.ceil(n_requerido):.0f})')
    
    ax.set_xlabel('Tamaño Muestral por Grupo')
    ax.set_ylabel('Potencia Estadística (1-β)')
    ax.set_title('Curva de Potencia vs Tamaño Muestral')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    st.pyplot(fig)

# Footer
st.markdown("---")
st.markdown(
    """
    **🧪 Aplicación desarrollada para Diseño de Experimentos**  
    *Métodos: ANOVA, Comparaciones Múltiples, Validación de Supuestos, Potencia*
    """
)

st.sidebar.markdown("---")
st.sidebar.markdown("🚀 Desarrollado para Análisis de Experimentos")
