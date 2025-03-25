# streamlit run app.py

import streamlit as st
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import BertTokenizer, BertModel, get_linear_schedule_with_warmup
from torch.optim import AdamW
from textwrap import wrap
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud
import re
import torch.nn as nn
from collections import Counter
import os
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import Dataset, DataLoader

# Configuraciones de página
st.set_page_config(
    page_title="Análisis de Sentimiento - Energías en Colombia",
    page_icon="🌞",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Definir la arquitectura del modelo
class ClasificadorSentimientoBERT(nn.Module):
    """BERT model con atención mejorada para clasificación de sentimiento"""

    def __init__(self, n_clases, nombre_modelo, prob_dropout=0.5):
        super(ClasificadorSentimientoBERT, self).__init__()

        # Cargar modelo BERT pre-entrenado
        self.bert = BertModel.from_pretrained(nombre_modelo, return_dict=True)

        # Añadir dropout para regularización
        self.drop = nn.Dropout(p=prob_dropout)

        # Capa de atención
        self.attention = nn.Sequential(
            nn.Linear(self.bert.config.hidden_size, 256),
            nn.Tanh(),
            nn.Linear(256, 1),
            nn.Softmax(dim=1)
        )

        # Cabeza de clasificación con más capas
        self.fc1 = nn.Linear(self.bert.config.hidden_size, 512)
        self.act1 = nn.ReLU()
        self.fc2 = nn.Linear(512, 256)
        self.act2 = nn.ReLU()
        self.clasificador = nn.Linear(256, n_clases)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        
        # Obtener representaciones de todos los tokens
        all_hidden_states = outputs.last_hidden_state  # [batch_size, seq_len, hidden_size]
        
        # Aplicar atención
        weights = self.attention(all_hidden_states)
        context_vector = torch.sum(weights * all_hidden_states, dim=1)
        
        # Clasificación por capas
        x = self.drop(context_vector)
        x = self.fc1(x)
        x = self.act1(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.act2(x)
        x = self.drop(x)
        return self.clasificador(x)

# Clase de configuración
class Config:
    """Clase de configuración para almacenar todos los parámetros del proyecto"""
    # Parámetros de datos
    RANDOM_SEED = 42  # Semilla aleatoria para reproducibilidad
    MAX_LEN = 200     # Longitud máxima de secuencia para BERT
    BATCH_SIZE = 32   # Tamaño de lote para entrenamiento
    DATASET_PATH = '' # Se llenará desde la UI
    TEST_SIZE = 0.2   # Proporción para conjunto de prueba

    # Parámetros del modelo
    PRE_TRAINED_MODEL_NAME = 'bert-base-cased'  # Modelo BERT pre-entrenado
    DROPOUT_PROB = 0.5  # Probabilidad de dropout para regularización
    N_CLASSES = 2       # Número de clases (positivo/negativo)

    # Parámetros de entrenamiento
    EPOCHS = 5         # Número de épocas de entrenamiento
    LEARNING_RATE = 1e-5  # Tasa de aprendizaje
    WARMUP_STEPS = 0     # Pasos de calentamiento para scheduler
    MAX_GRAD_NORM = 1.0  # Norma máxima de gradiente para clipping

    # Parámetros de salida
    MODEL_SAVE_PATH = './models'  # Ruta para guardar el modelo

    # Configuración de dispositivo
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Función para cargar el modelo
@st.cache_resource
def cargar_modelo(ruta_modelo, tokenizer_name='bert-base-cased'):
    """Carga el modelo entrenado desde el archivo guardado"""
    # Configuración
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Cargar tokenizador
    tokenizer = BertTokenizer.from_pretrained(tokenizer_name)
    
    # Inicializar modelo
    modelo = ClasificadorSentimientoBERT(
        n_clases=2,
        nombre_modelo=tokenizer_name,
        prob_dropout=0.5
    )
    
    # Cargar pesos guardados
    modelo.load_state_dict(torch.load(ruta_modelo, map_location=device))
    modelo = modelo.to(device)
    modelo.eval()
    
    return modelo, tokenizer, device

# Dataset para entrenamiento
class DatasetSentimiento(Dataset):
    """Dataset de PyTorch para análisis de sentimiento con BERT"""

    def __init__(self, reviews, etiquetas, tokenizer, max_len):
        """
        Inicializa el dataset
        
        Args:
            reviews (numpy.ndarray): Array de textos de reviews
            etiquetas (numpy.ndarray): Array de etiquetas de sentimiento (0 o 1)
            tokenizer (BertTokenizer): Tokenizador BERT
            max_len (int): Longitud máxima de secuencia
        """
        self.reviews = reviews
        self.etiquetas = etiquetas
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.reviews)

    def __getitem__(self, item):
        review = str(self.reviews[item])
        etiqueta = self.etiquetas[item]

        codificacion = self.tokenizer.encode_plus(
            review,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        return {
            'texto_review': review,
            'input_ids': codificacion['input_ids'].flatten(),
            'attention_mask': codificacion['attention_mask'].flatten(),
            'label': torch.tensor(etiqueta, dtype=torch.long)
        }

# Función para clasificar sentimiento
def clasificar_sentimiento(modelo, tokenizer, texto, device, max_len=200):
    """Clasifica el sentimiento de un texto dado"""
    # Verificación de seguridad
    if modelo is None or tokenizer is None:
        st.error("Modelo o tokenizer no iniciados. Por favor carga o entrena un modelo primero.")
        return None
    
    # Tokenizar el texto de entrada
    codificacion = tokenizer.encode_plus(
        texto,
        max_length=max_len,
        add_special_tokens=True,
        return_token_type_ids=False,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt'
    )

    # Mover a device
    input_ids = codificacion['input_ids'].to(device)
    attention_mask = codificacion['attention_mask'].to(device)

    # Obtener predicción del modelo
    with torch.no_grad():
        salidas = modelo(input_ids=input_ids, attention_mask=attention_mask)
        probs = torch.nn.functional.softmax(salidas, dim=1)
        _, prediccion = torch.max(salidas, dim=1)

    # Obtener puntuación de confianza
    confianza = probs[0][prediccion.item()].item()

    return {
        'texto': texto,
        'sentimiento': 'Positivo' if prediccion.item() == 1 else 'Negativo',
        'confianza': confianza,
        'prediccion': prediccion.item()
    }

# Función para entrenar el modelo
def entrenar_modelo(config, df, progress_bar=None, status_text=None):
    """
    Entrena un nuevo modelo con los parámetros proporcionados
    
    Args:
        config: Configuración del modelo
        df: DataFrame con los datos
        progress_bar: Barra de progreso de Streamlit
        status_text: Texto de estado de Streamlit
        
    Returns:
        tuple: (modelo, tokenizer)
    """
    # Configurar semillas para reproducibilidad
    np.random.seed(config.RANDOM_SEED)
    torch.manual_seed(config.RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.RANDOM_SEED)
    
    # Inicializar tokenizador
    if status_text:
        status_text.text("Inicializando tokenizador...")
    tokenizer = BertTokenizer.from_pretrained(config.PRE_TRAINED_MODEL_NAME)
    
    # Preparar datos
    if status_text:
        status_text.text("Preparando datos...")
    
    # Convertir etiquetas si es necesario
    if 'label' not in df.columns:
        df['label'] = (df['sentiment'] == 'Positivo').astype(int)
    
    # Dividir dataset
    df_train, df_test = train_test_split(
        df,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_SEED,
        stratify=df['label']
    )
    
    # Crear datasets
    train_dataset = DatasetSentimiento(
        reviews=df_train.review.to_numpy(),
        etiquetas=df_train.label.to_numpy(),
        tokenizer=tokenizer,
        max_len=config.MAX_LEN
    )
    
    test_dataset = DatasetSentimiento(
        reviews=df_test.review.to_numpy(),
        etiquetas=df_test.label.to_numpy(),
        tokenizer=tokenizer,
        max_len=config.MAX_LEN
    )
    
    # Crear dataloaders
    train_data_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=0
    )
    
    test_data_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        num_workers=0
    )
    
    # Inicializar modelo
    if status_text:
        status_text.text("Inicializando modelo...")
    modelo = ClasificadorSentimientoBERT(
        n_clases=config.N_CLASSES,
        nombre_modelo=config.PRE_TRAINED_MODEL_NAME,
        prob_dropout=config.DROPOUT_PROB
    )
    modelo = modelo.to(config.DEVICE)
    
    # Inicializar optimizador y scheduler
    optimizer = AdamW(modelo.parameters(), 
                      lr=config.LEARNING_RATE,
                      weight_decay=0.01)  # Usar PyTorch AdamW
    total_steps = len(train_data_loader) * config.EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config.WARMUP_STEPS,
        num_training_steps=total_steps
    )
    
    # Calcular pesos de clase
    y = df_train['label'].values
    class_weights = compute_class_weight('balanced', classes=np.unique(y), y=y)
    class_weights_dict = {i: class_weights[i] for i in range(len(class_weights))}
    class_weights_tensor = torch.FloatTensor(list(class_weights_dict.values())).to(config.DEVICE)
    
    # Inicializar función de pérdida
    funcion_perdida = nn.CrossEntropyLoss(weight=class_weights_tensor).to(config.DEVICE)
    
    # Entrenamiento
    if status_text:
        status_text.text("Iniciando entrenamiento...")
    
    mejor_accuracy = 0
    
    # Función para entrenar una época
    def entrenar_epoca(modelo, data_loader, funcion_perdida, optimizer, device, scheduler, n_ejemplos):
        modelo.train()
        perdidas = []
        predicciones_correctas = 0
        
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            etiquetas = batch['label'].to(device)
            
            salidas = modelo(input_ids=input_ids, attention_mask=attention_mask)
            perdida = funcion_perdida(salidas, etiquetas)
            _, preds = torch.max(salidas, dim=1)
            
            predicciones_correctas += torch.sum(preds == etiquetas)
            perdidas.append(perdida.item())
            
            perdida.backward()
            nn.utils.clip_grad_norm_(modelo.parameters(), max_norm=config.MAX_GRAD_NORM)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        
        accuracy = predicciones_correctas.double() / n_ejemplos
        perdida_promedio = np.mean(perdidas)
        return accuracy, perdida_promedio
    
    # Función para evaluar
    def evaluar_modelo(modelo, data_loader, funcion_perdida, device, n_ejemplos):
        modelo.eval()
        perdidas = []
        predicciones_correctas = 0
        
        with torch.no_grad():
            for batch in data_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                etiquetas = batch['label'].to(device)
                
                salidas = modelo(input_ids=input_ids, attention_mask=attention_mask)
                perdida = funcion_perdida(salidas, etiquetas)
                _, preds = torch.max(salidas, dim=1)
                
                predicciones_correctas += torch.sum(preds == etiquetas)
                perdidas.append(perdida.item())
        
        accuracy = predicciones_correctas.double() / n_ejemplos
        perdida_promedio = np.mean(perdidas)
        return accuracy, perdida_promedio
    
    # Bucle de entrenamiento
    for epoca in range(config.EPOCHS):
        if progress_bar:
            progress_bar.progress((epoca) / config.EPOCHS)
        
        if status_text:
            status_text.text(f"Entrenando época {epoca + 1}/{config.EPOCHS}...")
        
        # Fase de entrenamiento
        train_acc, train_loss = entrenar_epoca(
            modelo, train_data_loader, funcion_perdida, optimizer, config.DEVICE, scheduler, len(df_train)
        )
        
        # Fase de evaluación
        val_acc, val_loss = evaluar_modelo(
            modelo, test_data_loader, funcion_perdida, config.DEVICE, len(df_test)
        )
        
        if status_text:
            status_text.text(f"Época {epoca + 1}/{config.EPOCHS} - "
                           f"Train: {train_acc:.4f} | Val: {val_acc:.4f}")
        
        # Guardar el mejor modelo
        if val_acc > mejor_accuracy:
            mejor_accuracy = val_acc
            
            # Crear directorio si no existe
            os.makedirs(config.MODEL_SAVE_PATH, exist_ok=True)
            
            # Guardar modelo
            torch.save(modelo.state_dict(), f"{config.MODEL_SAVE_PATH}/mejor_modelo.pt")
            
            # Guardar tokenizador
            tokenizer.save_pretrained(config.MODEL_SAVE_PATH)
    
    if progress_bar:
        progress_bar.progress(1.0)
    
    if status_text:
        status_text.text(f"¡Entrenamiento completado! Mejor accuracy: {mejor_accuracy:.4f}")
    
    # Cargar el mejor modelo guardado
    modelo.load_state_dict(torch.load(f"{config.MODEL_SAVE_PATH}/mejor_modelo.pt"))
    
    return modelo, tokenizer

# Función para visualizar resultados con Plotly
def visualizar_resultado_plotly(resultado_clasificacion):
    """Crea una visualización interactiva del resultado con Plotly"""
    texto = resultado_clasificacion['texto']
    confianza = resultado_clasificacion['confianza'] * 100
    sentimiento = resultado_clasificacion['sentimiento']
    
    # Crear un gráfico de barras horizontal para la confianza
    color = '#28a745' if sentimiento == 'Positivo' else '#dc3545'
    
    fig = go.Figure()
    
    # Añadir la barra de confianza
    fig.add_trace(go.Bar(
        y=['Confianza'],
        x=[confianza],
        orientation='h',
        marker=dict(color=color),
        text=[f"{confianza:.1f}%"],
        textposition='auto',
        hoverinfo='none',
    ))
    
    # Configurar layout
    fig.update_layout(
        title=f"Sentimiento: {sentimiento}",
        title_font=dict(size=20, color=color),
        xaxis=dict(
            title='Confianza (%)',
            range=[0, 100],
            showgrid=True,
            gridcolor='rgba(0,0,0,0.1)'
        ),
        yaxis=dict(showticklabels=False),
        plot_bgcolor='rgba(0,0,0,0.02)',
        margin=dict(l=20, r=20, t=60, b=20),
        height=150
    )
    
    return fig

# Función para analizar frecuencia de palabras
def analizar_palabras(textos, sentimientos, stopwords=None):
    """Analiza la frecuencia de palabras en los textos según su sentimiento"""
    if stopwords is None:
        stopwords = ['el', 'la', 'los', 'las', 'un', 'una', 'unos', 'unas', 'y', 'o', 'a', 'ante', 
                     'bajo', 'con', 'de', 'desde', 'en', 'entre', 'hacia', 'hasta', 'para', 'por',
                     'según', 'sin', 'sobre', 'tras', 'que', 'como', 'cuando', 'donde', 'quien',
                     'cuyo', 'del', 'al', 'su', 'sus', 'este', 'esta', 'estos', 'estas', 'ese',
                     'esa', 'esos', 'esas', 'aquel', 'aquella', 'aquellos', 'aquellas', 'le', 'lo',
                     'se', 'si', 'no', 'más', 'ya', 'muy', 'es', 'son', 'ser', 'estar', 'pero',
                     'mi', 'tu', 'yo', 'qué', 'quién', 'cómo', 'cuándo', 'dónde', 'por qué',
                     'porque', 'pues', 'tan', 'fue', 'ha', 'han', 'había', 'he', 'me']
    
    textos_positivos = [texto for texto, sentimiento in zip(textos, sentimientos) if sentimiento == 'Positivo']
    textos_negativos = [texto for texto, sentimiento in zip(textos, sentimientos) if sentimiento == 'Negativo']
    
    def procesar_texto(texto):
        # Convertir a minúsculas y quitar caracteres especiales
        texto = texto.lower()
        texto = re.sub(r'[^\w\s]', '', texto)
        # Dividir en palabras
        palabras = texto.split()
        # Eliminar stopwords
        palabras = [palabra for palabra in palabras if palabra not in stopwords and len(palabra) > 2]
        return palabras
    
    # Procesar todos los textos
    palabras_positivas = []
    for texto in textos_positivos:
        palabras_positivas.extend(procesar_texto(texto))
    
    palabras_negativas = []
    for texto in textos_negativos:
        palabras_negativas.extend(procesar_texto(texto))
    
    # Contar frecuencias
    contador_positivas = Counter(palabras_positivas)
    contador_negativas = Counter(palabras_negativas)
    
    return contador_positivas, contador_negativas

# Función para crear nube de palabras
def crear_nube_palabras(contador_palabras, titulo, color='viridis'):
    """Crea una nube de palabras a partir de un contador de palabras"""
    
    # Crear la nube de palabras
    wc = WordCloud(
        background_color='white',
        max_words=50,
        colormap=color,
        width=800, 
        height=400,
        contour_width=1,
        contour_color='steelblue'
    ).generate_from_frequencies(contador_palabras)
    
    # Crear figura
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wc, interpolation='bilinear')
    ax.set_title(titulo, fontsize=16)
    ax.axis('off')
    
    return fig

# Función para crear gráfico de barras de palabras más frecuentes
def grafico_palabras_frecuentes(contador_palabras, titulo, color, n=10):
    """Crea un gráfico de barras con las palabras más frecuentes"""
    palabras_top = dict(contador_palabras.most_common(n))
    
    # Crear un DataFrame con los datos
    df = pd.DataFrame({
        'palabra': list(palabras_top.keys()),
        'frecuencia': list(palabras_top.values())
    })
    
    # Crear gráfico con Plotly
    fig = px.bar(
        df,
        x='frecuencia',
        y='palabra',
        orientation='h',
        title=titulo,
        labels={'frecuencia': 'Frecuencia', 'palabra': 'Palabras'},
        color_discrete_sequence=[color]
    )
    
    fig.update_layout(
        yaxis={'categoryorder':'total ascending'},
        margin=dict(l=20, r=20, t=50, b=20),
        height=400
    )
    
    return fig

# Cargar datos de ejemplo
@st.cache_data
def cargar_datos_ejemplo():
    """Carga datos de ejemplo predefinidos"""
    ejemplos = [
        "No vamos a salvar el mundo, eso sí seguiremos pobres, que desgracia estos empobrecedores con máscara de ambientalistas.",
        "Gracias @IvanDuque porque despues de mas de 30 meses después, la secta puede mostrar algunos resultados … así sean de otros.",
        "Excelente presidente, así debe ser. Avanzamos en la transición energética. ✊🏽💯",
        "Esperemos que esto sea un éxito para el beneficio de la población,y no esperar tal vez ataques de las empresas operadoras de energía a sabotear más el proceso...",
        "¡Qué bueno ver cómo Colombia está invirtiendo en energía solar! 🌞 Poco a poco vamos sumando esfuerzos para proteger el medio ambiente. #EnergíaSostenible #FuturoVerde 🇨🇴",
        "Presidente petro,un reformador del estado colombiano, para eso lo elejimos las mayorías de colombia,camino difícil en el tercer país más desigual del mundo,cuyos causantes de tal aberracion no se resignaran e intentarán las contrarreformas si vuelven a la presidencia.",
        "La transición energética debe ser más que un eslogan político. Necesitamos inversión real y políticas sostenibles.",
        "Los paneles solares son una estafa, solo benefician a unos pocos y no resuelven el problema energético de Colombia.",
        "Cada vez que veo nuevos proyectos de energía limpia me lleno de esperanza por el futuro de nuestro país.",
        "El gas sigue siendo indispensable para nuestra economía, no podemos abandonarlo por caprichos ideológicos."
    ]
    
    # Dataset mínimo con anotaciones para entrenamiento de prueba
    data = {
        'review': ejemplos,
        'sentiment': ['Negativo', 'Positivo', 'Positivo', 'Positivo', 'Positivo', 'Negativo', 'Negativo', 'Negativo', 'Positivo', 'Negativo']
    }
    
    return pd.DataFrame(data)

# Palabras discriminativas predefinidas
palabras_positivas_discriminativas = {
    "gracias": 8.83, 
    "transición": 6.00, 
    "verdad": 6.00, 
    "nuevos": 5.00, 
    "proyectos": 5.00, 
    "dejó": 5.00, 
    "anterior": 4.00, 
    "muchas": 4.00, 
    "sea": 4.00, 
    "trabajo": 3.50
}

palabras_negativas_discriminativas = {
    "solo": 8.00, 
    "usted": 6.00, 
    "gas": 6.00, 
    "hacen": 4.00, 
    "llegar": 4.00, 
    "hacer": 4.00, 
    "dónde": 4.00, 
    "colombianos": 4.00, 
    "mar": 4.00, 
    "paneles": 4.00
}

# --- APLICACIÓN PRINCIPAL ---

# Título y descripción
st.title("🌞 Análisis de Sentimiento: Políticas Energéticas en Colombia")
st.markdown("""
Esta aplicación utiliza un modelo de Inteligencia Artificial basado en BERT para analizar el sentimiento
en comentarios sobre políticas energéticas y transición energética en Colombia.
""")

# Sidebar con opciones
st.sidebar.header("Configuración")

# Opción para elegir entre cargar modelo pre-entrenado o entrenar nuevo
modo_modelo = st.sidebar.radio(
    "Selecciona el modo de uso del modelo:",
    ["Usar modelo pre-entrenado", "Entrenar nuevo modelo"]
)

# Variables para almacenar modelo y tokenizer
if 'modelo' not in st.session_state:
    st.session_state.modelo = None
if 'tokenizer' not in st.session_state:
    st.session_state.tokenizer = None
if 'device' not in st.session_state:
    st.session_state.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Si se elige cargar modelo pre-entrenado
if modo_modelo == "Usar modelo pre-entrenado":
    st.sidebar.write("### Cargar Modelo Pre-entrenado")
    
    # Agregar uso del selector de archivos
    opcion_carga = st.sidebar.radio(
        "Método de carga:",
        ["Usar ruta predeterminada", "Subir archivo de modelo"]
    )
    
    if opcion_carga == "Usar ruta predeterminada":
        # Opción 1: Usar ruta predeterminada
        ruta_modelo = st.sidebar.text_input(
            "Ruta al archivo del modelo:", 
            "models/mejor_modelo.pt"
        )
        
        # Verificar existencia del archivo
        if os.path.isfile(ruta_modelo):
            st.sidebar.success(f"✅ Archivo encontrado: {os.path.basename(ruta_modelo)}")
        else:
            st.sidebar.error(f"❌ Archivo no encontrado en: {ruta_modelo}")
            
        # Mostrar directorio actual para depuración
        dir_actual = os.getcwd()
        st.sidebar.info(f"Directorio actual: {dir_actual}")
        
        # Explorador de directorios básico
        if st.sidebar.checkbox("Mostrar archivos disponibles"):
            # Obtener directorio sugerido
            dir_sugerido = os.path.dirname(ruta_modelo) if os.path.dirname(ruta_modelo) else "."
            
            # Si el directorio no existe, usar el actual
            if not os.path.exists(dir_sugerido):
                dir_sugerido = "."
                
            st.sidebar.write(f"Contenido de: {dir_sugerido}")
            
            try:
                # Listar todos los archivos
                archivos = os.listdir(dir_sugerido)
                archivos_pt = [f for f in archivos if f.endswith('.pt')]
                
                if archivos_pt:
                    st.sidebar.write("Modelos disponibles (.pt):")
                    for archivo in archivos_pt:
                        ruta_completa = os.path.join(dir_sugerido, archivo)
                        if st.sidebar.button(f"📁 {archivo}", key=f"archivo_{archivo}"):
                            ruta_modelo = ruta_completa
                            st.rerun()
                else:
                    st.sidebar.warning(f"No se encontraron archivos .pt en {dir_sugerido}")
                    
                # Mostrar otros directorios para navegar
                directorios = [d for d in archivos if os.path.isdir(os.path.join(dir_sugerido, d))]
                if directorios:
                    st.sidebar.write("Subdirectorios:")
                    for dir in directorios:
                        if st.sidebar.button(f"📂 {dir}/", key=f"dir_{dir}"):
                            # Actualizar la ruta sugerida
                            nueva_ruta = os.path.join(dir_sugerido, dir, "mejor_modelo.pt")
                            ruta_modelo = nueva_ruta
                            st.rerun()
            except Exception as e:
                st.sidebar.error(f"Error al explorar directorio: {str(e)}")
    else:
        # Opción 2: Subir archivo directamente
        archivo_subido = st.sidebar.file_uploader("Sube el archivo del modelo (.pt)", type=['pt'])
        if archivo_subido:
            # Crear directorio temporal si no existe
            os.makedirs("temp_models", exist_ok=True)
            # Guardar el archivo subido
            ruta_modelo = f"temp_models/{archivo_subido.name}"
            with open(ruta_modelo, "wb") as f:
                f.write(archivo_subido.getbuffer())
            st.sidebar.success(f"✅ Modelo subido: {archivo_subido.name}")
        else:
            ruta_modelo = None
            st.sidebar.warning("⚠️ Por favor sube un archivo de modelo (.pt)")
    
    # Botón para cargar el modelo
    if ruta_modelo:  # Solo habilitar si hay ruta
        cargar_btn = st.sidebar.button("Cargar Modelo", type="primary")
        
        if cargar_btn:
            if not os.path.isfile(ruta_modelo):
                st.sidebar.error(f"❌ Archivo no encontrado en: {ruta_modelo}")
            else:
                try:
                    with st.spinner("Cargando el modelo..."):
                        # Guardar el modelo en el session_state
                        modelo_cargado, tokenizer_cargado, device = cargar_modelo(ruta_modelo)
                        st.session_state['modelo'] = modelo_cargado
                        st.session_state['tokenizer'] = tokenizer_cargado
                        st.session_state['device'] = device
                        st.session_state['modelo_cargado'] = True
                    st.sidebar.success("✅ Modelo cargado correctamente")
                except Exception as e:
                    st.sidebar.error(f"❌ Error al cargar el modelo: {str(e)}")
    else:
        st.sidebar.button("Cargar Modelo", type="primary", disabled=True)


# Si se elige entrenar nuevo modelo
else:
    st.sidebar.subheader("Configuración de entrenamiento")
    
    # Parámetros de entrenamiento
    dataset_option = st.sidebar.radio(
        "Datos para entrenamiento:",
        ["Usar ejemplos predefinidos", "Cargar archivo CSV/Excel"]
    )
    
    if dataset_option == "Cargar archivo CSV/Excel":
        archivo_datos = st.sidebar.file_uploader("Sube un archivo CSV/Excel con datos", type=['csv', 'xlsx'])
        if archivo_datos:
            try:
                if archivo_datos.name.endswith('.csv'):
                    df_entrenamiento = pd.read_csv(archivo_datos)
                else:
                    df_entrenamiento = pd.read_excel(archivo_datos)
                
                # Verificar columnas
                if 'review' in df_entrenamiento.columns and 'sentiment' in df_entrenamiento.columns:
                    st.sidebar.success(f"✅ Archivo cargado: {len(df_entrenamiento)} registros")
                else:
                    st.sidebar.error("❌ El archivo debe tener columnas 'review' y 'sentiment'")
                    df_entrenamiento = None
            except Exception as e:
                st.sidebar.error(f"❌ Error al cargar el archivo: {str(e)}")
                df_entrenamiento = None
        else:
            df_entrenamiento = None
    else:
        # Usar datos de ejemplo
        df_entrenamiento = cargar_datos_ejemplo()
        st.sidebar.info(f"ℹ️ Usando {len(df_entrenamiento)} ejemplos predefinidos")
    
    # Mostrar tabla de datos si están disponibles
    if df_entrenamiento is not None and st.sidebar.checkbox("Mostrar datos de entrenamiento"):
        st.sidebar.dataframe(df_entrenamiento[['review', 'sentiment']].head())
    
    # Parámetros adicionales
    config = Config()
    config.EPOCHS = st.sidebar.slider("Número de épocas", 1, 20, 5)
    config.BATCH_SIZE = st.sidebar.selectbox("Tamaño de batch", [8, 16, 32, 64], 1)  # Default 16
    config.LEARNING_RATE = float(st.sidebar.selectbox(
        "Tasa de aprendizaje", 
        ["1e-5", "2e-5", "3e-5", "5e-5"], 
        0
    ))
    
    # Ruta para guardar el modelo
    config.MODEL_SAVE_PATH = st.sidebar.text_input("Carpeta para guardar el modelo", "models")
    
    # Botón para entrenar
    entrenar_btn = st.sidebar.button("Entrenar Modelo")
    
    if entrenar_btn and df_entrenamiento is not None:
        # Contenedor para mostrar progreso
        entrenamiento_container = st.empty()
        with entrenamiento_container.container():
            st.write("### 🚀 Entrenando el modelo...")
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            try:
                # Entrenar modelo
                modelo, tokenizer = entrenar_modelo(
                    config, 
                    df_entrenamiento, 
                    progress_bar=progress_bar,
                    status_text=status_text
                )
                
                # Actualizar UI
                st.success(f"✅ Modelo entrenado y guardado en '{config.MODEL_SAVE_PATH}/mejor_modelo.pt'")
                
            except Exception as e:
                st.error(f"❌ Error durante el entrenamiento: {str(e)}")

# Información sobre el modelo
with st.sidebar.expander("ℹ️ Información del modelo"):
    st.write("""
    - **Tipo de modelo**: BERT con atención personalizada
    - **Parámetros**: 109,032,963
    - **Métricas clases**:
        - Positivo: Precisión 58%, Recall 72%
        - Negativo: Precisión 70%, Recall 55%
    """)

# Definir callbacks primero (fuera de la UI)
def cargar_ejemplo_aleatorio():
    """Callback para cargar un ejemplo aleatorio"""
    ejemplos = cargar_datos_ejemplo()['review'].tolist()
    ejemplo_aleatorio = np.random.choice(ejemplos)
    
    # Almacenar para la próxima renderización
    st.session_state.input_text = ejemplo_aleatorio
    # También almacenar para tenerlo disponible para análisis
    st.session_state.ultimo_texto = ejemplo_aleatorio

def analizar_texto_actual():
    """Callback para analizar el texto actual"""
    if 'modelo_cargado' not in st.session_state or not st.session_state.modelo_cargado:
        st.error("Modelo no cargado")
        return
        
    # Obtener el texto actual del widget
    texto_actual = st.session_state.input_text
    
    # Guardar para referencia futura
    st.session_state.ultimo_texto = texto_actual
    
    # Realizar análisis
    resultado = clasificar_sentimiento(
        st.session_state.modelo,
        st.session_state.tokenizer,
        texto_actual,
        st.session_state.device
    )
    
    if resultado:
        st.session_state.ultimo_resultado = resultado

# Secciones de la aplicación (pestañas)
tab1, tab2, tab3 = st.tabs(["Análisis Individual", "Análisis por Lotes", "Estadísticas"])

# Obtener variables del session_state para usarlas localmente
if 'modelo' in st.session_state and 'tokenizer' in st.session_state:
    modelo = st.session_state['modelo']  
    tokenizer = st.session_state['tokenizer']
    device = st.session_state['device']
else:
    modelo = None
    tokenizer = None
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Tab 1: Análisis Individual
with tab1:
    st.header("Análisis de Comentarios Individuales")
    
    # Entrada de texto (no necesitamos configurar initial_value, 
    # el callback maneja esto directamente)
    user_input = st.text_area(
        "Escribe un comentario sobre políticas energéticas en Colombia:",
        height=100,
        placeholder="Ejemplo: Las energías renovables son el futuro de Colombia, necesitamos más inversión en este sector.",
        key="input_text"  # Esta key corresponde a st.session_state.input_text
    )
    
    # Botones con callbacks
    col1, col2 = st.columns([1, 3])
    with col1:
        # Verificar si el modelo está cargado
        modelo_listo = 'modelo_cargado' in st.session_state and st.session_state['modelo_cargado']

        if not modelo_listo:
            st.warning("⚠️ Primero debes cargar o entrenar un modelo")

        # Botón que usa el callback, sin necesidad de st.rerun()
        st.button(
            "Analizar Sentimiento", 
            type="primary", 
            disabled=not modelo_listo,
            on_click=analizar_texto_actual
        )
    
    with col2:
        # Botón de ejemplo aleatorio con callback
        st.button(
            "Cargar ejemplo aleatorio",
            on_click=cargar_ejemplo_aleatorio
        )
    
    # Mostrar resultados si hay alguno disponible
    if 'ultimo_resultado' in st.session_state:
        resultado = st.session_state.ultimo_resultado
        
        # Obtener el texto analizado (del session_state o del widget actual)
        texto_a_mostrar = st.session_state.get('ultimo_texto', user_input)
        
        # Dividir en columnas
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Mostrar texto analizado
            st.subheader("Texto analizado:")
            st.write(texto_a_mostrar)
        
        with col2:
            # Mostrar resultados
            st.subheader("Resultado:")
            color = "green" if resultado['sentimiento'] == "Positivo" else "red"
            emoji = "😊" if resultado['sentimiento'] == "Positivo" else "😠"
            st.markdown(f"<h3 style='color:{color};'>{emoji} {resultado['sentimiento']}</h3>", unsafe_allow_html=True)
            st.metric("Confianza", f"{resultado['confianza']*100:.1f}%")
        
        # Gráfico interactivo
        st.plotly_chart(visualizar_resultado_plotly(resultado), use_container_width=True)
        
        # Análisis de contexto
        with st.expander("Ver análisis contextual"):
            st.write("### Palabras clave detectadas")
            
            # Detectar palabras discriminativas en el texto
            palabras_texto = set(re.sub(r'[^\w\s]', '', texto_a_mostrar.lower()).split())
            
            positivas_encontradas = [palabra for palabra in palabras_positivas_discriminativas.keys() 
                                   if palabra in palabras_texto]
            negativas_encontradas = [palabra for palabra in palabras_negativas_discriminativas.keys() 
                                    if palabra in palabras_texto]
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Palabras asociadas a sentimiento positivo:**")
                if positivas_encontradas:
                    for palabra in positivas_encontradas:
                        st.markdown(f"- **{palabra}** ({palabras_positivas_discriminativas[palabra]:.2f}x)")
                else:
                    st.write("No se encontraron palabras clave positivas.")
            
            with col2:
                st.write("**Palabras asociadas a sentimiento negativo:**")
                if negativas_encontradas:
                    for palabra in negativas_encontradas:
                        st.markdown(f"- **{palabra}** ({palabras_negativas_discriminativas[palabra]:.2f}x)")
                else:
                    st.write("No se encontraron palabras clave negativas.")

# Tab 2: Análisis por Lotes
with tab2:
    st.header("Análisis de Múltiples Comentarios")
    
    # Opciones de entrada
    opcion_entrada = st.radio(
        "Selecciona el método de entrada:",
        ["Texto (un comentario por línea)", "Cargar archivo CSV", "Usar ejemplos predefinidos"]
    )
    
    # Verificar si el modelo está cargado
    modelo_listo = 'modelo_cargado' in st.session_state and st.session_state['modelo_cargado']

    if not modelo_listo:
        st.warning("⚠️ Primero debes cargar o entrenar un modelo")
    
    if opcion_entrada == "Texto (un comentario por línea)":
        textos_input = st.text_area(
            "Ingresa múltiples comentarios (uno por línea):",
            height=150,
            placeholder="Comentario 1\nComentario 2\nComentario 3"
        )
        if textos_input:
            textos = textos_input.strip().split('\n')
        else:
            textos = []
            
    elif opcion_entrada == "Cargar archivo CSV":
        st.info("El archivo CSV debe tener una columna llamada 'comentario' o 'texto'.")
        archivo_csv = st.file_uploader("Sube un archivo CSV con comentarios", type=['csv'])
        if archivo_csv:
            try:
                df = pd.read_csv(archivo_csv)
                column = None
                for col_name in ['comentario', 'texto', 'comment', 'text', 'review']:
                    if col_name in df.columns:
                        column = col_name
                        break
                
                if column:
                    textos = df[column].tolist()
                    st.success(f"Se cargaron {len(textos)} comentarios del archivo.")
                else:
                    st.error("No se encontró una columna válida en el archivo.")
                    textos = []
            except Exception as e:
                st.error(f"Error al cargar el archivo: {str(e)}")
                textos = []
        else:
            textos = []
            
    else:  # Usar ejemplos predefinidos
        textos = cargar_datos_ejemplo()['review'].tolist()
        st.info(f"Se cargarán {len(textos)} comentarios de ejemplo.")
    
    # Botón para analizar
    if textos and st.button("Analizar Comentarios", disabled=not modelo_listo):
        if modelo is not None and tokenizer is not None:  # Añadir esta verificación
            with st.spinner(f"Analizando {len(textos)} comentarios..."):
                resultados = []
                progress_bar = st.progress(0)

                for i, texto in enumerate(textos):
                    if texto.strip():  # Ignorar líneas vacías
                        resultado = clasificar_sentimiento(st.session_state.modelo, st.session_state.tokenizer, texto, st.session_state.device)
                        if resultado:  # Verificar que el resultado no sea None
                            resultados.append(resultado)
                    # Actualizar barra de progreso
                    progress_bar.progress((i + 1) / len(textos))

                # Solo guardar resultados si hay alguno
                if resultados:
                    st.session_state.resultados_lote = resultados
                
    
    # Mostrar resultados si hay disponibles
    if 'resultados_lote' in st.session_state and modelo_listo:
        resultados = st.session_state.resultados_lote
        
        # Crear DataFrame para análisis
        df_resultados = pd.DataFrame([
            {'texto': r['texto'], 
             'sentimiento': r['sentimiento'], 
             'confianza': r['confianza']} 
            for r in resultados
        ])
        
        # Mostrar estadísticas generales
        st.subheader("Resumen de resultados")
        
        col1, col2, col3 = st.columns(3)
        
        total = len(df_resultados)
        positivos = len(df_resultados[df_resultados['sentimiento'] == 'Positivo'])
        negativos = len(df_resultados[df_resultados['sentimiento'] == 'Negativo'])
        
        with col1:
            st.metric("Total de comentarios", total)
        
        with col2:
            st.metric("Comentarios positivos", f"{positivos} ({positivos/total*100:.1f}%)")
        
        with col3:
            st.metric("Comentarios negativos", f"{negativos} ({negativos/total*100:.1f}%)")
        
        # Gráfico de distribución de sentimientos
        fig = px.pie(
            values=[positivos, negativos],
            names=['Positivo', 'Negativo'],
            title="Distribución de Sentimientos",
            color_discrete_sequence=['#28a745', '#dc3545'],
            hole=0.4
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Tabla de resultados detallados
        with st.expander("Ver todos los resultados"):
            # Añadir boton para descargar como CSV
            st.download_button(
                label="Descargar resultados como CSV",
                data=df_resultados.to_csv(index=False).encode('utf-8'),
                file_name='resultados_sentimiento.csv',
                mime='text/csv',
            )
            
            # Mostrar tabla con resultados
            st.dataframe(
                df_resultados,
                column_config={
                    'texto': st.column_config.TextColumn("Texto"),
                    'sentimiento': st.column_config.TextColumn("Sentimiento"),
                    'confianza': st.column_config.ProgressColumn(
                        "Confianza",
                        format="%.1f%%",
                        min_value=0,
                        max_value=1,
                    ),
                },
                hide_index=True,
            )
        
        # Análisis de palabras
        st.subheader("Análisis de Palabras")
        
        palabras_pos, palabras_neg = analizar_palabras(
            df_resultados[df_resultados['sentimiento'] == 'Positivo']['texto'].tolist(),
            df_resultados[df_resultados['sentimiento'] == 'Positivo']['sentimiento'].tolist()
        )
        
        _, palabras_neg = analizar_palabras(
            df_resultados[df_resultados['sentimiento'] == 'Negativo']['texto'].tolist(),
            df_resultados[df_resultados['sentimiento'] == 'Negativo']['sentimiento'].tolist()
        )
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.plotly_chart(
                grafico_palabras_frecuentes(palabras_pos, "Palabras más frecuentes (positivas)", "#28a745"),
                use_container_width=True
            )
        
        with col2:
            st.plotly_chart(
                grafico_palabras_frecuentes(palabras_neg, "Palabras más frecuentes (negativas)", "#dc3545"),
                use_container_width=True
            )
        
        # Nubes de palabras
        st.subheader("Nubes de Palabras")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if palabras_pos:
                fig_nube_pos = crear_nube_palabras(palabras_pos, "Nube de Palabras - Comentarios Positivos", 'Greens')
                st.pyplot(fig_nube_pos)
            else:
                st.info("No hay suficientes datos para generar la nube de palabras positivas.")
        
        with col2:
            if palabras_neg:
                fig_nube_neg = crear_nube_palabras(palabras_neg, "Nube de Palabras - Comentarios Negativos", 'Reds')
                st.pyplot(fig_nube_neg)
            else:
                st.info("No hay suficientes datos para generar la nube de palabras negativas.")

# Tab 3: Estadísticas del Modelo
with tab3:
    st.header("Estadísticas del Modelo")
    
    # Verificar si el modelo está cargado
    if not modelo_listo:
        st.warning("⚠️ Primero debes cargar o entrenar un modelo para ver estadísticas")
    else:
        # Métricas de rendimiento
        st.subheader("Métricas de Rendimiento")
        
        metricas_df = pd.DataFrame([
            {"Métrica": "Accuracy general", "Valor": "62.96%"},
            {"Métrica": "Precisión clase Negativa", "Valor": "70.00%"},
            {"Métrica": "Recall clase Negativa", "Valor": "55.00%"},
            {"Métrica": "F1-score clase Negativa", "Valor": "62.00%"},
            {"Métrica": "Precisión clase Positiva", "Valor": "58.00%"},
            {"Métrica": "Recall clase Positiva", "Valor": "72.00%"},
            {"Métrica": "F1-score clase Positiva", "Valor": "64.00%"},
        ])
        
        st.table(metricas_df)
        
        # Matriz de confusión
        st.subheader("Matriz de Confusión")
        
        confusion_matrix = np.array([[16, 13], [7, 18]])
        
        fig = px.imshow(
            confusion_matrix,
            x=['Predicho Negativo', 'Predicho Positivo'],
            y=['Real Negativo', 'Real Positivo'],
            color_continuous_scale='Blues',
            labels=dict(color="Cantidad"),
            text_auto=True
        )
        
        fig.update_layout(
            xaxis_title="Predicción",
            yaxis_title="Etiqueta Real",
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Palabras discriminativas
        st.subheader("Palabras Discriminativas")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Palabras más asociadas con sentimiento positivo:**")
            pos_df = pd.DataFrame([
                {"Palabra": k, "Frecuencia relativa": f"{v:.2f}x"}
                for k, v in palabras_positivas_discriminativas.items()
            ])
            st.table(pos_df)
        
        with col2:
            st.write("**Palabras más asociadas con sentimiento negativo:**")
            neg_df = pd.DataFrame([
                {"Palabra": k, "Frecuencia relativa": f"{v:.2f}x"}
                for k, v in palabras_negativas_discriminativas.items()
            ])
            st.table(neg_df)
        
        # Análisis de confianza del modelo
        st.subheader("Análisis de Confianza")
        
        confianza_df = pd.DataFrame([
            {"Indicador": "Confianza promedio", "Valor": "85.60%"},
            {"Indicador": "Confianza mínima", "Valor": "60.52%"},
            {"Indicador": "Confianza máxima", "Valor": "91.31%"},
        ])
        
        st.table(confianza_df)
        
        # Distribución de predicciones
        st.subheader("Distribución de Predicciones")
        
        predicciones_df = pd.DataFrame([
            {"Clase": "Negativo (0)", "Cantidad": 23, "Porcentaje": "42.59%"},
            {"Clase": "Positivo (1)", "Cantidad": 31, "Porcentaje": "57.41%"},
        ])
        
        fig = px.bar(
            predicciones_df, 
            x='Clase', 
            y='Cantidad',
            color='Clase',
            text='Porcentaje',
            color_discrete_map={
                'Negativo (0)': '#dc3545',
                'Positivo (1)': '#28a745'
            },
            labels={'Cantidad': 'Número de predicciones', 'Clase': 'Clase predicha'}
        )
        
        st.plotly_chart(fig, use_container_width=True)

# Footer
st.markdown("""
---
**Proyecto: ANÁLISIS DE SENTIMIENTO SOBRE POLÍTICAS ENERGÉTICAS EN COLOMBIA CON NLP**

Desarrollado con ❤️ usando Streamlit, BERT y PyTorch | por Alexander y Robinson 2025
""")