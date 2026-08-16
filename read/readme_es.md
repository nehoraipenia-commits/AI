# AI - Neural Generative Core & Semantic Framework

Un sistema avanzado, ligero e híbrido de inteligencia generativa, desarrollado de forma nativa con PyTorch. El framework combina una arquitectura Transformer causal personalizada con Byte-Pair Encoding (BPE), búsqueda vectorial semántica (RAG), reconstrucción de frases en francés (FSR), seguimiento dinámico del perfil del usuario y fine-tuning cooperativo sin conexión de LLM locales mediante instancias de Ollama.

### Características principales

* Motor Transformer causal personalizado: Desarrollado completamente a partir de las primitivas de PyTorch, con codificaciones posicionales sinusoidales, atención causal multi-cabeza y normalización de capas.

* Tokenización Byte-Pair Encoding (BPE): Pipeline personalizado de tokenización por subpalabras con escalado dinámico del vocabulario y normalización completa de Unicode y acentos.

* Memoria semántica multiarchivo (RAG): Almacén vectorial basado en TF-IDF y similitud coseno, capaz de indexar bases de código Python (.py) y documentación (.txt) para una generación consciente del contexto.

* Integración con Ollama y fine-tuning dinámico: Framework cooperativo multimodelo que obtiene ampliaciones de conocimiento de LLM locales (por ejemplo, Gemma, Llama) para entrenar dinámicamente los pesos PyTorch personalizados sin conexión.

* Reconstrucción de frases en francés (FSR): Módulo neuronal Seq2Seq dedicado a corregir contracciones francesas, errores sintácticos y errores tipográficos.

* Memoria dinámica del usuario y motor de perfil: Extracción en tiempo real de información del usuario (nombre, preferencias, contexto), mantenida a través de sesiones de conversación sucesivas.

* Interfaz CLI interactiva: Interfaz de terminal completa con análisis de ejecución en tiempo real y gestión del estado del modelo.

* Arquitectura del sistema

```text
                                  +---------------------------+
                                  |   Entrada del usuario /   |
                                  |          Prompt           |
                                  +-------------+-------------+
                                                |
                                                v
   +----------------------------+   +-----------+-----------+   +----------------------------+
   | Memoria dinámica           |-->| Memoria semántica      |<--| Archivos del directorio   |
   | del usuario                |   | (RAG)                  |   | indexados (.py & .txt)    |
   | (hechos e historial)       |   | (Almacén TF-IDF)       |   |                            |
   +----------------------------+   +-----------+-----------+   +----------------------------+
                                                |
                                                v
                                 +--------------+--------------+
                                 | Tokenización Byte-Pair     |
                                 | Encoding (BPE)             |
                                 | Pipeline                   |
                                 +--------------+--------------+
                                                |
                                                v
                                 +--------------+--------------+
                                 | Transformer causal        |
                                 | personalizado              |
                                 | (Atención multi-cabeza)    |
                                 +--------------+--------------+
                                                |
                                                v
  +----------------------------+    +-----------+-------------+   +----------------------------+
  | Módulo de reconstrucción   |<--| Motor de sampling       |-->| LLM Ollama local         |
  | francesa                   |   | inteligente              |   | (Bucles de warmup         |
  | (Corrección gramatical)    |   | (Top-K/Nucleus Sampling) |   | cooperativos)             |
  +----------------------------+    +-----------+-------------+   +----------------------------+
                                                |
                                                v
                                  +-------------+-------------+
                                  |  Respuesta generada por   |
                                  |           la IA           |
                                  +---------------------------+
```

## Requisitos previos

* Antes de la instalación, asegúrate de que los siguientes programas estén instalados en tu máquina:

* Python: Versión 3.8 o superior

* PyTorch: Versión 1.12.0 o superior (soporte CUDA/MPS opcional pero recomendado)

* Ollama (Opcional): Necesario únicamente si deseas utilizar el fine-tuning cooperativo con LLM locales.

* Instalación y configuración

### Clonar el repositorio

```bash
git clone https://github.com/nehoraipenia-commits/AI.git
cd AI
```

### Crear y activar un entorno virtual (Recomendado)

## En macOS/Linux

```bash
python3 -m venv venv
source venv/bin/activate
```

## En Windows

```bash
python -m venv venv
venv\Scripts\activate
```

### Instalar las dependencias necesarias

```bash
pip install torch numpy scikit-learn requests
```

### Iniciar la interfaz CLI del asistente

```bash
python3 ai_pytorch.py
```

## Comandos de la interfaz CLI interactiva

Una vez iniciado el CLI (`AI-CLI >>>`), puedes introducir prompts de texto estándar o utilizar los comandos de control del sistema:

Comando

Descripción

```text
/status
```

Muestra las especificaciones del hardware, la asignación del dispositivo, el tamaño del vocabulario BPE activo y el estado del archivo de memoria base.

```text
/train <epochs>
```

Reoptimiza los pesos del Transformer directamente a partir de tu base de código y documentación locales.

```text
/model <name>
```

Cambia el identificador del modelo Ollama local activo (por ejemplo, `/model gemma2:2b` o `/model llama3`).

```text
/gemma <count>
```

Ejecuta bucles de generación cooperativa con Ollama para ampliar la base de conocimientos local y volver a entrenar el modelo.

```text
/train_fsr <ep>
```

Entrena el módulo Transformer de reconstrucción de frases en francés utilizando textos locales.

```text
/reconstruct <text>
```

Corrige formulaciones francesas corruptas, errores tipográficos o contracciones mediante el modelo FSR.

```text
/save
```

Exporta manualmente los checkpoints y los pesos actuales del sistema al disco (`checkpoints/`).

```text
/load
```

Recarga los estados guardados del sistema y los vocabularios del tokenizer desde los archivos de checkpoint.

```text
/clear
```

Limpia la pantalla de la interfaz de terminal.

```text
/exit
```

Finaliza los procesos de forma segura y guarda los pesos actuales.

## Estructura del repositorio

```text
AI/
├── AI.py                            # Controlador principal del pipeline del sistema y CLI interactiva
├── AdvancedPhraseReformulator.py    # Motor de reformulación gramatical basado en POS
├── ai_generation_helper.py          # Muestreo inteligente de logits, motor de Markov y optimización de texto
├── FrenchContractionHandler.py      # Reglas de contracción francesa y reconstrucción Seq2Seq
├── dynamic_user_memory.py           # Extractor dinámico de datos del usuario y gestor del historial
├── nlp_tokenization_suite.py        # Suite de tokenización Byte-Pair Encoding (BPE)
├── big_book.txt                     # Corpus principal del sistema y base de conocimientos
├── ollama_responses.txt             # Registro de las ampliaciones generadas por los LLM cooperativos
└── checkpoints/                     # Directorio de serialización de los pesos del modelo
```

Para más información: https://nexeo-ai.netlify.app/

