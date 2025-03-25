# 📊 Análisis de Sentimiento sobre Políticas Energéticas en Colombia mediante BERT

![Banner](https://live.staticflickr.com/65535/54408567583_bb7a61c461_b.jpg)

## 📌 Resumen
Este proyecto analiza el sentimiento de comentarios sobre políticas energéticas en Colombia usando un modelo NLP basado en BERT. Se evaluó la percepción pública sobre la transición energética, energías renovables y políticas ambientales a partir de comentarios extraídos de redes sociales (Twitter/X).

El modelo alcanzó una precisión del **62.96%**, identificando patrones lingüísticos asociados a opiniones positivas y negativas. Se detectó un debate polarizado con una ligera tendencia hacia comentarios positivos sobre energías renovables, aunque con preocupaciones económicas y de implementación.

## 🌐 Acceso al Modelo  

🔍 **Este modelo es un archivo `.pt` que contiene una versión finetuneada de BERT**, específicamente entrenada para el análisis de comentarios sobre políticas públicas y energéticas en Colombia. 🏛️⚡  

📢 **¡El modelo ya está disponible para su uso en línea!** Puedes probarlo directamente en la aplicación de Streamlit:  
👉 [🔗 **bert-talent-tech.streamlit.app**](https://bert-talent-tech.streamlit.app/)  

📥 **Opciones de descarga y repositorio**  

🔹 **Mediafire**: [📂 Carpeta del modelo en Mediafire](https://www.mediafire.com/folder/7fytsb7tv9anz/TalentoTech2025_IA)  
🔹 **OneDrive**: [📁 Carpeta del modelo en OneDrive](https://1drv.ms/f/c/5277a5d3cfb61d6b/EtJzKYgzq75Ek07LacS1GNwBn7Ow5MZ-dYhWdsN6_wcLjw?e=Wsy3AH)  
🔹 **Repositorio en GitHub**: [📜 Código fuente en GitHub (rama master)](https://github.com/junortiz/BERT-Analisis-de-sentimiento-comentarios-x-twitter/tree/master)  

📌 **Opciones de uso:**  

1️⃣ **Prueba el modelo en línea** en la aplicación de Streamlit.  
2️⃣ **Descarga manualmente el modelo** y agrégalo a la carpeta `models/` de tu proyecto.  
3️⃣ **Descarga automática desde Hugging Face** dentro de la aplicación.  

🖼️ **Ejemplo de la opción en la aplicación:**  

![Interfaz de Streamlit](https://live.staticflickr.com/65535/54408848458_fb31148eb6_b.jpg)  

🖼️ **Pasos para cargar modelo:**  

![Interfaz de Streamlit](https://live.staticflickr.com/65535/54408973810_bd3b492f08_b.jpg)  

### 🚀 Opciones de Uso
1. **Cargar el modelo preentrenado** y obtener resultados de inmediato.
2. **Generar un nuevo modelo** usando el dataset disponible.

## 📊 Metodología
### 📥 Recopilación de Datos
Se extrajeron comentarios de **Twitter/X** con términos clave sobre energía y sostenibilidad en Colombia. Se analizaron menciones en:
- Ministerio de Medio Ambiente
- Presidente Gustavo Petro
- Ecopetrol
- Palabras clave: "energía", "clima", "paneles", "sostenibilidad", etc.

### 🔬 Arquitectura del Modelo Preentrenado
- **BERT en español (bert-base-cased)**
- **Capas densas adicionales (768→512→256→2)**
- **Activación ReLU + Dropout 0.5**
- **Balanceo de clases mediante pesos calculados**
- **109M parámetros entrenables**

# 🚀 Instrucciones para Implementar la Aplicación Streamlit

Sigue estos pasos para desplegar tu aplicación de análisis de sentimiento en Streamlit. Asegúrate de cumplir cada parte para garantizar su correcto funcionamiento. 😎

---

## 📁 1. Estructura de Carpetas

Organiza tu proyecto con la siguiente estructura para mantener todo bien ordenado:

```
analisis-sentimiento-energia/
├── app.py              # Código principal de la aplicación
├── requirements.txt    # Archivo con las dependencias necesarias
└── models/
    ├── mejor_modelo.pt  # Tu modelo pre-entrenado
    └── tokenizer/       # Archivos del tokenizador de BERT
        ├── vocab.txt
        ├── config.json
        ├── tokenizer_config.json
        └── ... # Otros archivos del tokenizador
```

Esta organización facilita la gestión de los archivos y evita errores al cargar los modelos.

---

## 🛠️ 2. Pasos de Implementación

### 🐍 Paso 1: Crea y Activa un Entorno Virtual

Para aislar las dependencias del proyecto, crea y activa un entorno virtual:

```bash
# Crear un entorno virtual llamado 'env'
python -m venv env

# Activar el entorno virtual
# En Windows:
env\Scripts\activate
# En macOS/Linux:
source env/bin/activate
```

Esto evitará conflictos de dependencias con otros proyectos.

### 📦 Paso 2: Instala las Dependencias

Instala los paquetes requeridos con el archivo `requirements.txt`:

```bash
pip install -r requirements.txt
```

Esto descargará e instalará todas las librerías necesarias.

### 📚 Paso 3: Copia tu Modelo Pre-entrenado y Tokenizador

1. Crea la carpeta `models` (si no existe):
   ```bash
   mkdir -p models
   ```
2. Copia el archivo `mejor_modelo.pt` dentro de `models/`.
3. Si tienes los archivos del tokenizador (ej. `vocab.txt`, `config.json`), colócalos en `models/tokenizer/`.

Esto garantizará que el modelo y el tokenizador sean accesibles por la aplicación.

### 🚀 Paso 4: Ejecuta la Aplicación en Streamlit

Con el entorno listo y los archivos en su lugar, inicia la aplicación con:

```bash
streamlit run app.py
```

Al ejecutar este comando, Streamlit abrirá automáticamente el navegador con la interfaz de la aplicación.

---

## 💻 3. Uso de la Aplicación

La aplicación tiene dos modos de funcionamiento:

### 🔍 Modo 1: Usar Modelo Pre-entrenado
1. Selecciona "Usar modelo pre-entrenado" en la barra lateral.
2. Ingresa la ruta al modelo (por defecto: `models/mejor_modelo.pt`).
3. Haz clic en "Cargar Modelo".
4. Analiza comentarios individuales o por lotes.

### 👩‍🎓 Modo 2: Entrenar un Nuevo Modelo
1. Selecciona "Entrenar nuevo modelo" en la barra lateral.
2. Usa ejemplos predefinidos o carga tus propios datos.
3. Configura los parámetros de entrenamiento (número de épocas, batch size, etc.).
4. Haz clic en "Entrenar Modelo".
5. Una vez finalizado el entrenamiento, usa el nuevo modelo para clasificar comentarios.

---

## 🔧 4. Solución de Problemas Comunes

### 📅 Error al Cargar el Modelo
Si ves "No se pudo cargar el modelo", verifica:
1. Que `mejor_modelo.pt` exista en la ruta correcta.
2. Que la estructura del modelo coincida con `ClasificadorSentimientoBERT`.

### 💪 Error de Memoria Durante el Entrenamiento
Si el entrenamiento falla por problemas de memoria:
1. Reduce el batch size (ej. de 32 a 16 o 8).
2. Usa un subconjunto de datos más pequeño.
3. Reduce la longitud máxima de secuencia (`MAX_LEN`).

### 📝 Error al Cargar Archivos CSV
Si hay problemas al cargar datos:
1. Verifica que el archivo tenga las columnas `review` y `sentiment`.
2. Asegúrate de que esté en formato UTF-8.
3. Comprueba que no haya valores nulos en las columnas principales.



---

Siguiendo estos pasos, tu aplicación de análisis de sentimiento en Streamlit funcionará sin problemas. ¡Éxito! 🚀😄

