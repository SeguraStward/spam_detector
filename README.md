# Spam Detector — Bilingüe (Inglés + Español)

> Clasificador de mensajes y correos electrónicos que detecta **spam vs ham** con Machine Learning sobre un corpus multilingüe (~9 000 mensajes EN+ES).

[![Python](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-orange.svg)](https://scikit-learn.org/)
[![Status](https://img.shields.io/badge/status-completed-green.svg)](#)
[![F1 Score](https://img.shields.io/badge/F1-0.955-brightgreen.svg)](#-resultados)

---

## Tabla de contenido

- [Características](#-características)
- [Resultados](#-resultados)
- [Arquitectura](#-arquitectura)
- [Requisitos previos](#-requisitos-previos)
- [Instalación](#-instalación)
- [Uso rápido](#-uso-rápido)
- [Comandos disponibles](#-comandos-disponibles)
- [Datasets](#-datasets)
- [Algoritmos](#-algoritmos)
- [Estructura del proyecto](#-estructura-del-proyecto)
- [Decisiones de diseño](#-decisiones-de-diseño)
- [Extender el proyecto](#-extender-el-proyecto)
- [Solución de problemas](#-solución-de-problemas)
- [Referencias](#-referencias)

---

## Características

- **Bilingüe** — entrenado y validado en inglés y español
- **3 algoritmos intercambiables** — Naive Bayes, Regresión Logística, Logística + char n-grams
- **Pipeline end-to-end** — preprocesamiento, vectorización y modelo serializados juntos (`.joblib`)
- **CLI ergonómica** — predicciones desde texto, archivo, o evaluación masiva
- **Probabilidades calibradas** — cada predicción incluye nivel de confianza por clase
- **Métricas profesionales** — accuracy, precision, recall, F1 y matriz de confusión
- **Reproducible** — `random_state=42` y split estratificado

---

## Resultados

Sobre el test set (20 % stratified, ~1 660 mensajes EN+ES):

| Algoritmo | Accuracy | Precision (spam) | Recall (spam) | F1 (spam) |
|-----------|---------:|-----------------:|--------------:|----------:|
| Naive Bayes (BoW) | 0.919 | 0.650 | **0.945** | 0.770 |
| **Logistic + TF-IDF (1,2)** | **0.987** | **0.936** | 0.975 | **0.955** |
| Logistic + TF-IDF char_wb (3,5) | 0.984 | 0.920 | 0.971 | 0.945 |

> **Recomendación:** usa `logistic` para máxima precisión global, o `logistic_char` cuando el texto pueda venir con errores ortográficos o mezcla de idiomas.

---

## Arquitectura

```
            ┌──────────────┐
   Texto ──▶│ Preprocessor │  HTML → URLs → emails → minúsculas → stopwords EN+ES
            └──────┬───────┘
                   ▼
            ┌──────────────┐
            │  Vectorizer  │  BoW · TF-IDF (1,2) · TF-IDF char_wb (3,5)
            └──────┬───────┘
                   ▼
            ┌──────────────┐
            │  Classifier  │  MultinomialNB · LogisticRegression
            └──────┬───────┘
                   ▼
       { label, confidence, probabilities }
```

Todo va dentro de un `sklearn.Pipeline`, así que el modelo persistido en `.joblib` reaplica el mismo preprocesamiento en inferencia que en entrenamiento.

---

## Requisitos previos

- **Python 3.11+**
- **pip** y **venv** (`sudo apt install python3-venv python3-full` en Debian/Ubuntu)
- ~200 MB libres en disco (datasets + modelos)

---

## Instalación

```bash
# 1. Clonar y entrar al proyecto
git clone <url-del-repo> spam_detection
cd spam_detection

# 2. Crear entorno virtual
python3 -m venv venv
source venv/bin/activate

# 3. Instalar dependencias
pip install --upgrade pip
pip install -r requirements.txt

# 4. Descargar stopwords de NLTK
python -c "import nltk; nltk.download('stopwords')"
```

> **Tip:** verifica que estás en el venv con `which python` — debe apuntar a `…/spam_detection/venv/bin/python`.

---

## Uso rápido

Tres comandos para tener todo listo:

```bash
python -m src.download_data    # 1. descarga datasets (~5 MB)
python -m src.main --train     # 2. entrena los 3 modelos
python -m src.main -a logistic -m "WIN a free iPhone now!!!"
```

Salida esperada:

```
Algoritmo:   logistic
Predicción:  SPAM
Confianza:   91.28%
Probabilidades:
  P(ham)  = 8.72%
  P(spam) = 91.28%
```

---

## Comandos disponibles

### Entrenar todos los modelos

```bash
python -m src.main --train
```

Entrena los 3 algoritmos, evalúa cada uno y los persiste en `models/*.joblib`.

### Clasificar un mensaje desde la línea de comandos

```bash
python -m src.main -a <algoritmo> -m "<mensaje>"
```

Donde `<algoritmo>` es uno de: `naive_bayes`, `logistic`, `logistic_char`.

**Ejemplos:**

```bash
# Inglés
python -m src.main -a logistic -m "Click here to claim your free reward!!!"

# Español
python -m src.main -a logistic -m "FELICIDADES ganaste 5000 dolares, reclama en http://premio.cc"

# Ham legítimo
python -m src.main -a logistic -m "Hola, paso por ti a las 7 para ir al cine"
```

### Clasificar el contenido de un archivo

```bash
python -m src.main -a logistic -f ruta/al/correo.txt
```

### Evaluar un modelo sobre el test set

```bash
python -m src.main -a logistic --evaluate
```

Muestra accuracy, precision, recall, F1 y matriz de confusión.

### Ayuda

```bash
python -m src.main --help
```

---

## Datasets

El proyecto combina **cuatro fuentes complementarias** para cubrir email + SMS, inglés + español:

| Fuente | Tipo | Idioma | Tamaño | Origen |
|--------|------|--------|-------:|--------|
| SpamAssassin (`data/raw/`) | Email | EN | ~3 000 | [Apache SpamAssassin](https://spamassassin.apache.org/old/publiccorpus/) |
| SMS Spam Collection | SMS | EN | ~5 500 | [UCI ML Repository](https://archive.ics.uci.edu/dataset/228/sms+spam+collection) |
| Spanish Spam Seed (`data/spanish_spam.tsv`) | SMS/Email curado | ES | ~90 | Curado manualmente (phishing real de banca, telco, ecommerce) |
| Enron-Spam *(opcional)* (`data/raw/enron/`) | Email | EN | hasta ~33 000 | [Metsis et al. 2006](https://www2.aueb.gr/users/ion/data/enron-spam/) |

### Distribución de clases

- **Ham:** ~7 100 mensajes
- **Spam:** ~1 200 mensajes (~14 %)

El loader unifica todo en un DataFrame con columnas `text`, `label`, `lang` y elimina duplicados automáticamente.

---

## Algoritmos

### 1. Naive Bayes (BoW)

- **Vectorización:** Bag of Words con n-grams (1,2)
- **Suavizado:** Laplace, α = 1.0
- **Ventaja:** muy rápido, excelente recall
- **Mejor cuando:** quieres minimizar falsos negativos

### 2. Regresión Logística + TF-IDF *(recomendado)*

- **Vectorización:** TF-IDF (1,2)-grams, `sublinear_tf=True`
- **Optimizador:** liblinear, `C=1.0`
- **`class_weight=balanced`** para compensar desbalance
- **Mejor cuando:** quieres máxima precisión global en el idioma de entrenamiento

### 3. Regresión Logística + TF-IDF char_wb

- **Vectorización:** TF-IDF de n-grams de caracteres (3-5)
- **Mejor cuando:** el texto tiene errores ortográficos, mezcla de idiomas, o transferencia cross-lingual (entrenado mayormente en inglés, evaluado en español)

---

## Estructura del proyecto

```
spam_detection/
├── data/
│   ├── raw/                    # SpamAssassin / Enron (no versionado)
│   ├── sms_spam_en.tsv         # UCI SMS Spam Collection
│   └── spanish_spam.tsv        # Seed curado en español
├── models/                     # *.joblib (creado tras --train)
├── src/
│   ├── __init__.py
│   ├── preprocessor.py         # Limpieza + tokenización bilingüe
│   ├── vectorizer.py           # Factory: BoW / TF-IDF / char_wb
│   ├── models.py               # Pipelines de scikit-learn
│   ├── evaluator.py            # Métricas y matriz de confusión
│   ├── load_dataset.py         # Carga unificada multi-fuente
│   ├── download_data.py        # Descarga SMS + seed español
│   ├── train.py                # Entrenamiento y persistencia
│   └── main.py                 # CLI con argparse
├── PLANNING.md                 # Análisis PEAS y plan original
├── PROGRESS.md                 # Bitácora de avance
├── README.md                   # Este archivo
└── requirements.txt
```

---

## Decisiones de diseño

| Decisión | Motivo |
|----------|--------|
| `sklearn.Pipeline` end-to-end | Garantiza que inferencia y entrenamiento usen exactamente el mismo preprocesamiento. Un solo `.joblib` contiene todo. |
| `class_weight="balanced"` | El dataset tiene ~14 % de spam; sin esto el modelo se sesga a predecir ham. |
| `sublinear_tf=True` | Reduce el peso desproporcionado de palabras muy frecuentes en correos largos. |
| char n-grams (3,5) | Agnósticos de idioma — capturan patrones lexicales sub-palabra (`http`, `gana`, `gratis`, `$$$`) que transfieren entre idiomas. |
| Stopwords bilingües combinadas | Una sola lista (EN ∪ ES) elimina ruido en ambos idiomas con un solo vectorizador. |
| `random_state=42` + `stratify` | Resultados reproducibles y proporciones de clase preservadas en train/test. |
| `joblib` sobre `pickle` | `joblib` es más eficiente con arrays de NumPy/SciPy. |

---

## Extender el proyecto

### Mejorar el rendimiento en español

Añadir más filas a `data/spanish_spam.tsv` (formato `label\ttext`):

```tsv
label	text
spam	Tu cuenta del banco fue bloqueada, verifica en http://...
ham	Mañana nos vemos a las 5 en el restaurante
```

Luego reentrenar:

```bash
python -m src.main --train
```

### Agregar una nueva fuente de datos

Edita `src/load_dataset.py::load_all` y añade tu loader (`load_folder_dataset` o `load_tsv_dataset`).

### Probar otro algoritmo

Edita `src/models.py::build_pipeline` y añade tu rama:

```python
if algorithm == "svm":
    return Pipeline([
        ("clean", TextPreprocessor()),
        ("vec", build_vectorizer("tfidf")),
        ("clf", LinearSVC(C=1.0, class_weight="balanced")),
    ])
```

Y agrégalo a la tupla `ALGORITHMS`.

---

## Solución de problemas

### `ModuleNotFoundError: No module named 'sklearn'`

No tienes el venv activado. Ejecuta:

```bash
source venv/bin/activate
which python    # debe apuntar a venv/bin/python
```

### `error: externally-managed-environment` al hacer `pip install`

Estás usando el `pip` del sistema (PEP 668). Activa el venv primero:

```bash
source venv/bin/activate
which pip       # debe apuntar a venv/bin/pip
```

### `Command 'python' not found` con el venv activado

El venv está corrupto. Recréalo:

```bash
deactivate 2>/dev/null
rm -rf venv
sudo apt install -y python3-venv python3-full
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### `LookupError: Resource stopwords not found`

Falta descargar las stopwords de NLTK:

```bash
python -c "import nltk; nltk.download('stopwords')"
```

### `FileNotFoundError: No se encontraron datasets bajo data/`

Aún no has descargado los datasets:

```bash
python -m src.download_data
```

### `FileNotFoundError: No existe models/*.joblib`

Aún no has entrenado los modelos:

```bash
python -m src.main --train
```

---

## Referencias

- Russell, S. & Norvig, P. — *Artificial Intelligence: A Modern Approach*
- Metsis, V., Androutsopoulos, I., Paliouras, G. (2006) — *Spam Filtering with Naive Bayes — Which Naive Bayes?*
- Almeida, T. A., Hidalgo, J. M. G. (2011) — *Contributions to the Study of SMS Spam Filtering*
- [scikit-learn — Working with Text Data](https://scikit-learn.org/stable/tutorial/text_analytics/working_with_text_data.html)

---

## Licencia

Proyecto académico — curso de Inteligencia Artificial.
