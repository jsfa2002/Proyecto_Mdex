# -*- coding: utf-8 -*-
"""
Aplicación de Muestreo y Diseño de Experimentos - Streamlit
"""

import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy.stats import shapiro, levene, f_oneway, kruskal, ttest_ind
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.power import FTestAnovaPower, TTestPower
from statsmodels.formula.api import ols
import statsmodels.api as sm
from sklearn.utils import resample
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
    .sidebar .sidebar-content { background-color: #f0f2f6; }
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
    .warning { background-color: #fff3cd; border-left: 4px solid #ffc107; }
    </style>
    """, unsafe_allow_html=True)

# Título principal
st.title(" Diseño de Experimentos - Bioestimulantes en Aguacates")
st.markdown("---")

# Sidebar para navegación
st.sidebar.header("Módulos de Análisis")
app_mode = st.sidebar.selectbox(
    "Selecciona el módulo",
    [
        " Contexto & Datos", 
        " Análisis Exploratorio", 
        " Diseño Experimental",
        " ANOVA & Comparaciones",
        " Validación Supuestos",
        " Potencia Estadística",
        " Remuestreo & Bootstrap"
    ]
)

# Cargar datos
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('datos_aguacate_masivos.csv')
        return df
    except:
        return None

df = load_data()

# Módulo 1: Contexto & Datos
if app_mode == " Contexto & Datos":
    st.header(" Contexto del Proyecto")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ###  Evaluación de Bioestimulantes en Aguacates Hass
        
        **Objetivo del Experimento:**
        Evaluar el efecto de 4 formulaciones de bioestimulantes sobre el peso del aguacate Hass.
        
        **Diseño Experimental:**
        - Completamente aleatorizado (CRD)
        - 4 tratamientos
        - 10 árboles por tratamiento (unidades experimentales)
        - 5 frutas medidas por árbol
        
        **Hipótesis:**
        - H₀: No hay diferencia en el peso promedio entre tratamientos
        - H₁: Al menos un tratamiento difiere en el peso promedio
        """)
    
    with col2:
        st.markdown("""
        ###  Características del Diseño
        
        | Parámetro | Valor |
        |-----------|-------|
        | Tratamientos | 4 |
        | Árboles/Tratamiento | 10 |
        | Frutas/Árbol | 5 |
        | Total Observaciones | 200 |
        | Unidad Experimental | Árbol |
        | Variable Respuesta | Peso (g) |
        """)
        
        st.metric(" Total de Árboles", 40)
        st.metric(" Observaciones", 200)
        st.metric(" Grados de Libertad", "Trat: 3, Error: 36")
    
    st.markdown("---")
    
    # Descripción de tratamientos
    st.subheader(" Descripción de Tratamientos")
    
    tratamientos_info = {
        'Tratamiento': ['Control', 'A - Aminoácidos', 'B - Extracto de Algas', 'C - Ácidos Húmicos'],
        'Descripción': [
            'Manejo estándar sin bioestimulantes',
            'Hidrolizado de proteína para síntesis celular eficiente',
            'Ascophyllum nodosum para división celular acelerada',
            'Mejoradores de suelo y absorción de nutrientes'
        ],
        'Mecanismo Esperado': [
            'Línea base',
            'Ahorro energético en síntesis',
            'Estimulación división celular',
            'Mejora absorción nutrientes'
        ]
    }
    
    df_tratamientos = pd.DataFrame(tratamientos_info)
    st.dataframe(df_tratamientos, use_container_width=True)
    
    # Mostrar datos si están disponibles
    st.markdown("---")
    st.subheader(" Datos del Experimento")
    
    if df is not None:
        st.success(" Datos cargados exitosamente")
        
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
        st.subheader(" Vista Previa de Datos")
        st.dataframe(df.head(10))
        
        # Estructura de datos
        with st.expander(" Estructura del Dataset"):
            st.write("**Variables:**")
            for col in df.columns:
                st.write(f"- {col}: {df[col].dtype}")
            
            st.write("**Balance del Diseño:**")
            balance = df.groupby('Tratamiento')['Arbol_ID'].nunique()
            st.write(balance)
            
    else:
        st.error(" No se pudo cargar el archivo 'datos_aguacate_masivos.csv'")
        st.info(" Asegúrate de que el archivo esté en el directorio correcto")

# Módulo 2: Análisis Exploratorio
elif app_mode == " Análisis Exploratorio":
    st.header(" Análisis Exploratorio de Datos")
    
    if df is not None:
        # Selección de tipo de análisis
        analisis_type = st.selectbox(
            "Tipo de análisis exploratorio",
            [
                " Distribución por Tratamiento",
                " Comparación de Medias", 
                " Variabilidad entre Árboles",
                " Boxplots Comparativos",
                " Outliers & Valores Extremos"
            ]
        )
        
        if analisis_type == " Distribución por Tratamiento":
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
                
        elif analisis_type == " Comparación de Medias":
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
            
            # Coeficiente de variación
            st.subheader(" Medidas de Variabilidad")
            cv_data = []
            for tratamiento in df['Tratamiento'].unique():
                data = df[df['Tratamiento'] == tratamiento]['Peso_g']
                cv = (data.std() / data.mean()) * 100
                cv_data.append({'Tratamiento': tratamiento, 'CV (%)': round(cv, 2)})
            
            df_cv = pd.DataFrame(cv_data)
            st.dataframe(df_cv)
            
        elif analisis_type == " Variabilidad entre Árboles":
            st.subheader("Análisis de Variabilidad entre Árboles")
            
            # Seleccionar tratamiento para análisis detallado
            tratamiento_seleccionado = st.selectbox(
                "Seleccionar tratamiento para análisis detallado",
                df['Tratamiento'].unique()
            )
            
            df_trat = df[df['Tratamiento'] == tratamiento_seleccionado]
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Boxplot por árbol
                fig, ax = plt.subplots(figsize=(12, 6))
                sns.boxplot(data=df_trat, x='Arbol_ID', y='Peso_g', ax=ax)
                ax.set_title(f'Distribución de Pesos por Árbol - {tratamiento_seleccionado}')
                ax.tick_params(axis='x', rotation=45)
                st.pyplot(fig)
            
            with col2:
                # Media y desviación por árbol
                arbol_stats = df_trat.groupby('Arbol_ID')['Peso_g'].agg(['mean', 'std']).round(2)
                arbol_stats.columns = ['Media', 'Desv. Std']
                st.dataframe(arbol_stats)
                
                # Gráfico de medias por árbol
                fig, ax = plt.subplots(figsize=(10, 6))
                medias_arbol = df_trat.groupby('Arbol_ID')['Peso_g'].mean().sort_values()
                ax.bar(medias_arbol.index, medias_arbol.values)
                ax.set_title(f'Media de Peso por Árbol - {tratamiento_seleccionado}')
                ax.set_ylabel('Peso Promedio (g)')
                ax.tick_params(axis='x', rotation=45)
                st.pyplot(fig)
                
        elif analisis_type == " Boxplots Comparativos":
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
            st.subheader(" Preview ANOVA")
            grupos = [df[df['Tratamiento'] == tr]['Peso_g'].values for tr in df['Tratamiento'].unique()]
            f_stat, p_value = f_oneway(*grupos)
            
            col1, col2 = st.columns(2)
            col1.metric("Estadístico F", f"{f_stat:.4f}")
            col2.metric("Valor p", f"{p_value:.4f}")
            
        elif analisis_type == " Outliers & Valores Extremos":
            st.subheader("Detección de Outliers y Valores Extremos")
            
            # Método IQR
            outliers_data = []
            for tratamiento in df['Tratamiento'].unique():
                data = df[df['Tratamiento'] == tratamiento]['Peso_g']
                Q1 = data.quantile(0.25)
                Q3 = data.quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outliers = data[(data < lower_bound) | (data > upper_bound)]
                outliers_data.append({
                    'Tratamiento': tratamiento,
                    'Outliers': len(outliers),
                    'Porcentaje': f"{(len(outliers)/len(data))*100:.1f}%",
                    'Valores': list(outliers.values) if len(outliers) > 0 else "Ninguno"
                })
            
            df_outliers = pd.DataFrame(outliers_data)
            st.dataframe(df_outliers)
            
    else:
        st.error(" No hay datos cargados para análisis")

# Módulo 3: Diseño Experimental
elif app_mode == " Diseño Experimental":
    st.header(" Diseño Experimental y Consideraciones")
    
    st.markdown("""
    ###  Diseño Completamente Aleatorizado (CRD)
    
    **Estructura del Modelo:**
    """)
    
    st.latex(r"Y_{ij} = \mu + \tau_i + \epsilon_{ij}")
    
    st.markdown("""
    Donde:
    - $Y_{ij}$: Peso de la fruta j en el tratamiento i
    - $\mu$: Media general del peso
    - $\tau_i$: Efecto del tratamiento i
    - $\epsilon_{ij}$: Error experimental ∼ N(0, σ²)
    """)
    
    # Consideraciones del diseño
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ###  Fortalezas del Diseño
        
        **Ventajas:**
        - Aleatorización completa
        - Simplicidad en implementación
        - Fácil análisis estadístico
        - Igual número de réplicas
        
        **Control Experimental:**
        - Árboles homogéneos seleccionados
        - Mismo manejo agronómico
        - Mismo ambiente de crecimiento
        - Mediciones estandarizadas
        """)
    
    with col2:
        st.markdown("""
        ###  Limitaciones y Mejoras
        
        **Posibles Mejoras:**
        - Bloqueo por posición en finca
        - Covariables (edad árbol, producción previa)
        - Medidas repetidas en el tiempo
        - Mayor número de réplicas
        
        **Consideraciones:**
        - Efecto árbol como fuente de variación
        - Correlación intra-árbol
        - Tamaño del efecto detectable
        """)
    
    st.markdown("---")
    
    # Tabla ANOVA esperada
    st.subheader(" Tabla ANOVA Esperada")
    
    anova_esperada = {
        'Fuente de Variación': ['Tratamientos', 'Error', 'Total'],
        'Grados de Libertad': [3, 36, 39],
        'Suma de Cuadrados': ['SCT', 'SCE', 'SCTotal'],
        'Cuadrados Medios': ['CMT = SCT/3', 'CME = SCE/36', '-'],
        'Estadístico F': ['F = CMT/CME', '-', '-']
    }
    
    df_anova_esperada = pd.DataFrame(anova_esperada)
    st.dataframe(df_anova_esperada, use_container_width=True)
    
    # Cálculo de potencia
    st.subheader(" Cálculo de Potencia del Diseño")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        efecto_esperado = st.slider("Tamaño del efecto esperado (f)", 0.1, 1.0, 0.5, 0.1)
    
    with col2:
        alpha = st.slider("Nivel de significancia (α)", 0.01, 0.10, 0.05, 0.01)
    
    with col3:
        n_grupos = st.number_input("Número de grupos", 2, 10, 4)
    
    # Calcular potencia
    power_analysis = FTestAnovaPower()
    potencia = power_analysis.solve_power(
        effect_size=efecto_esperado,
        nobs=10,  # 10 árboles por tratamiento
        alpha=alpha,
        k_groups=n_grupos
    )
    
    st.metric("Potencia Estadística (1-β)", f"{potencia:.3f}")
    
    if potencia < 0.8:
        st.warning(" Potencia insuficiente (< 0.8). Considera aumentar el tamaño muestral.")
    else:
        st.success(" Potencia adecuada para detectar el efecto")

# Módulo 4: ANOVA & Comparaciones
elif app_mode == " ANOVA & Comparaciones":
    st.header(" Análisis de Varianza (ANOVA) y Comparaciones Múltiples")
    
    if df is not None:
        # Realizar ANOVA
        st.subheader(" Análisis de Varianza (ANOVA)")
        
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
             **Resultado Significativo:** Se rechaza H₀. 
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
            
            # Gráfico de comparaciones
            fig, ax = plt.subplots(figsize=(10, 6))
            tukey.plot_simultaneous(ax=ax)
            ax.set_title('Comparaciones Múltiples - Tukey HSD')
            ax.set_xlabel('Diferencia en Medias')
            st.pyplot(fig)
            
            # Resumen de diferencias
            st.subheader(" Resumen de Diferencias Significativas")
            
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
             **Resultado No Significativo:** No se rechaza H₀.
            No hay evidencia suficiente de diferencias entre los tratamientos.
            """)
        
        # Análisis de efectos
        st.subheader(" Análisis de Efectos y Potencia")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Interpretación del tamaño del efecto
            if eta_squared < 0.01:
                interpretacion = "Muy pequeño"
            elif eta_squared < 0.06:
                interpretacion = "Pequeño"
            elif eta_squared < 0.14:
                interpretacion = "Mediano"
            else:
                interpretacion = "Grande"
            
            st.metric("Tamaño del Efecto", interpretacion)
            
            # Potencia observada
            n_per_group = df.groupby('Tratamiento').size().min()
            k = len(df['Tratamiento'].unique())
            f_effect = np.sqrt(eta_squared / (1 - eta_squared))  # Convertir a f
            
            power_analysis = FTestAnovaPower()
            potencia_observada = power_analysis.power(f_effect, n_per_group, k, alpha=0.05)
            st.metric("Potencia Observada", f"{potencia_observada:.4f}")
        
        with col2:
            # Gráfico de medias con diferencias
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
            
    else:
        st.error(" No hay datos cargados para análisis")

# Módulo 5: Validación Supuestos
elif app_mode == " Validación Supuestos":
    st.header(" Validación de Supuestos del ANOVA")
    
    if df is not None:
        st.markdown("""
        ###  Supuestos del ANOVA
        
        1. **Normalidad:** Los residuos deben distribuirse normalmente
        2. **Homocedasticidad:** Varianzas iguales entre grupos  
        3. **Independencia:** Observaciones independientes entre sí
        4. **Aleatorización:** Asignación aleatoria de tratamientos
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
        st.subheader("1.  Normalidad de los Residuos")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Test de Shapiro-Wilk
            stat_sw, p_sw = shapiro(residuos)
            st.metric("Shapiro-Wilk p-value", f"{p_sw:.4f}")
            
            if p_sw > 0.05:
                st.markdown('<div class="assumption-check valid"> No se rechaza normalidad (p > 0.05)</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="assumption-check invalid"> Se rechaza normalidad (p ≤ 0.05)</div>', unsafe_allow_html=True)
        
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
                st.markdown('<div class="assumption-check valid"> No se rechaza homocedasticidad (p > 0.05)</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="assumption-check invalid"> Se rechaza homocedasticidad (p ≤ 0.05)</div>', unsafe_allow_html=True)
        
        with col2:
            # Gráfico de varianzas
            varianzas = [np.var(grupo) for grupo in grupos]
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.bar(df['Tratamiento'].unique(), varianzas, alpha=0.7)
            ax.set_title('Varianzas por Tratamiento')
            ax.set_ylabel('Varianza')
            ax.tick_params(axis='x', rotation=45)
            st.pyplot(fig)
        
        # 3. Independencia
        st.subheader("3. 🔄 Independencia de Observaciones")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **Evaluación del Diseño:**
            -  Aleatorización completa
            -  Unidades experimentales independientes
            -  Sin medidas repetidas en el tiempo
            -  Sin correlación espacial conocida
            """)
            st.markdown('<div class="assumption-check valid"> Diseño asegura independencia</div>', unsafe_allow_html=True)
        
        with col2:
            # Gráfico de residuos vs orden
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.scatter(range(len(residuos)), residuos, alpha=0.6)
            ax.axhline(y=0, color='red', linestyle='--')
            ax.set_xlabel('Orden de Observación')
            ax.set_ylabel('Residuos')
            ax.set_title('Residuos vs Orden (Independencia)')
            st.pyplot(fig)
        
        # Resumen de validación
        st.subheader("📋 Resumen de Validación de Supuestos")
        
        supuestos_validos = all([p_sw > 0.05, p_lev > 0.05])
        
        if supuestos_validos:
            st.success("""
             **Todos los supuestos se cumplen.** 
            El análisis ANOVA es válido y las conclusiones son confiables.
            """)
        else:
            st.warning("""
             **Algunos supuestos no se cumplen.**
            Considera:
            - Transformaciones de datos (log, sqrt)
            - Tests no paramétricos (Kruskal-Wallis)
            - Modelos de efectos mixtos
            - Análisis robustos
            """)
            
            # Alternativas cuando supuestos fallan
            st.subheader(" Alternativas cuando los Supuestos Fallan")
            
            if p_sw <= 0.05:
                st.info("**Para normalidad:** Prueba Kruskal-Wallis (no paramétrico)")
                stat_kw, p_kw = kruskal(*grupos)
                st.metric("Kruskal-Wallis p-value", f"{p_kw:.4f}")
            
            if p_lev <= 0.05:
                st.info("**Para varianzas desiguales:** Prueba Welch o transformaciones")
                
    else:
        st.error(" No hay datos cargados para validación")

# Módulo 6: Potencia Estadística
elif app_mode == " Potencia Estadística":
    st.header(" Análisis de Potencia Estadística")
    
    st.markdown("""
    ###  ¿Qué es la Potencia Estadística?
    
    La potencia (1-β) es la probabilidad de detectar un efecto cuando realmente existe.
    - **Potencia alta (> 0.8):** Buena capacidad para detectar efectos
    - **Potencia baja:** Riesgo de Error Tipo II (no detectar efecto real)
    """)
    
    # Calculadora de potencia
    st.subheader(" Calculadora de Potencia")
    
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
            st.success(" Potencia adecuada")
        else:
            st.warning(" Potencia insuficiente")
    
    with col2:
        st.metric("Tamaño Muestral Requerido", f"{np.ceil(n_requerido):.0f} por grupo")
        if n_grupo >= n_requerido:
            st.success(" Tamaño muestral adecuado")
        else:
            st.warning(" Se necesitan más réplicas")
    
    # Gráfico de curva de potencia
    st.subheader("Curva de Potencia")
    
    n_range = np.arange(2, 101, 2)
    power_curve = power_analysis.power(f_effect, n_range, k_grupos, alpha)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(n_range, power_curve, linewidth=2, label=f'f = {f_effect}')
    ax.axhline(y=potencia_deseada, color='red', linestyle='--', alpha=0.7, 
               label=f'Potencia deseada ({potencia_deseada})')
    ax.axvline(x=n_requerido, color='green', linestyle='--', alpha=0.7, 
               label=f'n requerido ({np.ceil(n_requerido):.0f})')
    ax.axvline(x=n_grupo, color='blue', linestyle='--', alpha=0.7, 
               label=f'n actual ({n_grupo})')
    
    ax.set_xlabel('Tamaño Muestral por Grupo')
    ax.set_ylabel('Potencia Estadística (1-β)')
    ax.set_title('Curva de Potencia vs Tamaño Muestral')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    st.pyplot(fig)
    
    # Análisis de potencia post-hoc si hay datos
    if df is not None:
        st.subheader(" Análisis de Potencia Post-Hoc")
        
        # Calcular tamaño del efecto observado
        model = ols('Peso_g ~ C(Tratamiento)', data=df).fit()
        anova_table = sm.stats.anova_lm(model, typ=2)
        
        ss_tratamiento = anova_table['sum_sq'][0]
        ss_total = anova_table['sum_sq'].sum()
        eta_squared = ss_tratamiento / ss_total
        f_observed = np.sqrt(eta_squared / (1 - eta_squared))
        
        n_actual = df.groupby('Tratamiento').size().min()
        k_actual = len(df['Tratamiento'].unique())
        
        potencia_observada = power_analysis.power(f_observed, n_actual, k_actual, 0.05)
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Tamaño Efecto Observado (f)", f"{f_observed:.3f}")
        col2.metric("Potencia Observada", f"{potencia_observada:.3f}")
        col3.metric("Mínimo Efecto Detectable", f"{power_analysis.solve_power(effect_size=None, power=0.8, nobs=n_actual, alpha=0.05, k_groups=k_actual):.3f}")

# Módulo 7: Remuestreo & Bootstrap
elif app_mode == " Remuestreo & Bootstrap":
    st.header(" Métodos de Remuestreo y Bootstrap")
    
    if df is not None:
        st.markdown("""
        ###  ¿Qué es el Bootstrap?
        
        Técnica de remuestreo para estimar la distribución de un estadístico
        cuando los supuestos paramétricos no se cumplen.
        """)
        
        # Configuración del bootstrap
        st.subheader(" Configuración del Análisis Bootstrap")
        
        col1, col2 = st.columns(2)
        
        with col1:
            n_bootstrap = st.slider("Número de réplicas bootstrap", 100, 5000, 1000)
            statistic = st.selectbox("Estadístico a estimar", 
                                   ["Media", "Diferencia de Medias", "Varianza"])
        
        with col2:
            confidence_level = st.slider("Nivel de confianza", 0.80, 0.99, 0.95)
            grupo1 = st.selectbox("Grupo 1 (para diferencia)", df['Tratamiento'].unique())
            grupo2 = st.selectbox("Grupo 2 (para diferencia)", 
                                df['Tratamiento'].unique(), 
                                index=1)
        
        if st.button(" Ejecutar Bootstrap"):
            st.subheader(" Resultados del Bootstrap")
            
            if statistic == "Media":
                # Bootstrap para media de un grupo
                grupo_seleccionado = st.selectbox("Grupo para análisis", df['Tratamiento'].unique())
                data = df[df['Tratamiento'] == grupo_seleccionado]['Peso_g']
                
                bootstrap_means = []
                for _ in range(n_bootstrap):
                    sample = resample(data, replace=True, random_state=42)
                    bootstrap_means.append(np.mean(sample))
                
                # Intervalo de confianza
                alpha = 1 - confidence_level
                lower = np.percentile(bootstrap_means, (alpha/2)*100)
                upper = np.percentile(bootstrap_means, (1 - alpha/2)*100)
                
                col1, col2, col3 = st.columns(3)
                col1.metric("Media Original", f"{np.mean(data):.2f}")
                col2.metric(f"IC Bootstrap {confidence_level:.0%}", f"({lower:.2f}, {upper:.2f})")
                col3.metric("Error Estándar Bootstrap", f"{np.std(bootstrap_means):.4f}")
                
                # Histograma bootstrap
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.hist(bootstrap_means, bins=30, alpha=0.7, density=True)
                ax.axvline(np.mean(data), color='red', linestyle='--', label='Media Original')
                ax.axvline(lower, color='green', linestyle='--', label=f'Límite IC {confidence_level:.0%}')
                ax.axvline(upper, color='green', linestyle='--')
                ax.set_xlabel('Media Bootstrap')
                ax.set_ylabel('Densidad')
                ax.set_title(f'Distribución Bootstrap de la Media - {grupo_seleccionado}')
                ax.legend()
                st.pyplot(fig)
                
            elif statistic == "Diferencia de Medias":
                # Bootstrap para diferencia de medias
                data1 = df[df['Tratamiento'] == grupo1]['Peso_g']
                data2 = df[df['Tratamiento'] == grupo2]['Peso_g']
                
                bootstrap_diffs = []
                for _ in range(n_bootstrap):
                    sample1 = resample(data1, replace=True, random_state=42)
                    sample2 = resample(data2, replace=True, random_state=42)
                    bootstrap_diffs.append(np.mean(sample1) - np.mean(sample2))
                
                # Intervalo de confianza
                alpha = 1 - confidence_level
                lower = np.percentile(bootstrap_diffs, (alpha/2)*100)
                upper = np.percentile(bootstrap_diffs, (1 - alpha/2)*100)
                
                diff_original = np.mean(data1) - np.mean(data2)
                
                col1, col2, col3 = st.columns(3)
                col1.metric("Diferencia Original", f"{diff_original:.2f}")
                col2.metric(f"IC Bootstrap {confidence_level:.0%}", f"({lower:.2f}, {upper:.2f})")
                
                # Test de hipótesis
                p_value = np.mean(np.array(bootstrap_diffs) <= 0) if diff_original > 0 else np.mean(np.array(bootstrap_diffs) >= 0)
                p_value = 2 * min(p_value, 1 - p_value)  # Two-tailed
                col3.metric("Valor p Bootstrap", f"{p_value:.4f}")
                
                # Histograma diferencias
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.hist(bootstrap_diffs, bins=30, alpha=0.7, density=True)
                ax.axvline(diff_original, color='red', linestyle='--', label='Diferencia Original')
                ax.axvline(lower, color='green', linestyle='--', label=f'Límite IC {confidence_level:.0%}')
                ax.axvline(upper, color='green', linestyle='--')
                ax.axvline(0, color='black', linestyle='-', alpha=0.5, label='Diferencia Nula')
                ax.set_xlabel('Diferencia de Medias Bootstrap')
                ax.set_ylabel('Densidad')
                ax.set_title(f'Distribución Bootstrap: {grupo1} - {grupo2}')
                ax.legend()
                st.pyplot(fig)
                
                # Interpretación
                if p_value < 0.05:
                    st.success(f" Diferencia estadísticamente significativa (p = {p_value:.4f})")
                else:
                    st.warning(f" Diferencia no significativa (p = {p_value:.4f})")
        
        # Simulación de Monte Carlo
        st.subheader(" Simulación de Monte Carlo")
        
        st.markdown("""
        Simulación del experimento bajo diferentes escenarios para evaluar 
        la robustez de las conclusiones.
        """)
        
        if st.button(" Ejecutar Simulación Monte Carlo"):
            st.info("Ejecutando simulación... Esto puede tomar unos segundos.")
            
            # Parámetros de simulación basados en datos observados
            n_simulations = 500
            efectos_simulados = []
            p_values_simulados = []
            
            media_global = df['Peso_g'].mean()
            std_global = df['Peso_g'].std()
            
            for _ in range(n_simulations):
                # Simular datos bajo H0 (no efecto)
                datos_simulados = []
                for tratamiento in df['Tratamiento'].unique():
                    n_arboles = df[df['Tratamiento'] == tratamiento]['Arbol_ID'].nunique()
                    n_frutas = 5  # Asumiendo 5 frutas por árbol
                    
                    for arbol in range(n_arboles):
                        # Efecto aleatorio del árbol
                        efecto_arbol = np.random.normal(0, std_global * 0.2)
                        for fruta in range(n_frutas):
                            peso = np.random.normal(media_global + efecto_arbol, std_global * 0.5)
                            datos_simulados.append({
                                'Tratamiento': tratamiento,
                                'Peso_g': max(peso, 100)  # Mínimo realista
                            })
                
                df_sim = pd.DataFrame(datos_simulados)
                
                # ANOVA en datos simulados
                try:
                    model_sim = ols('Peso_g ~ C(Tratamiento)', data=df_sim).fit()
                    anova_sim = sm.stats.anova_lm(model_sim, typ=2)
                    p_values_simulados.append(anova_sim['PR(>F)'][0])
                except:
                    p_values_simulados.append(1.0)
            
            # Calcular tasa de error tipo I
            tasa_error_I = np.mean(np.array(p_values_simulados) < 0.05)
            
            st.metric("Tasa de Error Tipo I (α observado)", f"{tasa_error_I:.3f}")
            
            if abs(tasa_error_I - 0.05) < 0.02:
                st.success(" Tasa de Error Tipo I cercana al α nominal (0.05)")
            else:
                st.warning(" Tasa de Error Tipo I diferente del α nominal")
            
            # Distribución de p-values bajo H0
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.hist(p_values_simulados, bins=20, alpha=0.7, density=True)
            ax.axvline(0.05, color='red', linestyle='--', label='α = 0.05')
            ax.set_xlabel('Valor p')
            ax.set_ylabel('Densidad')
            ax.set_title('Distribución de Valores p bajo H₀ (Simulación Monte Carlo)')
            ax.legend()
            st.pyplot(fig)
            
    else:
        st.error(" No hay datos cargados para análisis de remuestreo")

# Footer
st.markdown("---")
st.markdown(
    """
    ** Aplicación desarrollada para Diseño de Experimentos y Análisis Estadístico**  
    *Métodos: ANOVA, Comparaciones Múltiples, Validación de Supuestos, Potencia, Bootstrap*
    """
)

st.sidebar.markdown("---")
st.sidebar.markdown(" Desarrollado para Análisis de Experimentos")
