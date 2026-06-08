# Detector de Spam Bilingüe (Inglés + Español)

> Clasificador de mensajes **spam vs ham** con Machine Learning sobre un corpus bilingüe (~6 290 mensajes EN + ES). Todo el proyecto vive en un único notebook reproducible.

[![Python](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-orange.svg)](https://scikit-learn.org/)
[![F1](https://img.shields.io/badge/F1-0.938-brightgreen.svg)](#resultados)
[![Recall](https://img.shields.io/badge/Recall-0.947-brightgreen.svg)](#resultados)

Curso de Inteligencia Artificial · Universidad Nacional.

---

## Tabla de contenido
- [Qué es](#qué-es)
- [Resultados](#resultados)
- [Cómo ejecutarlo](#cómo-ejecutarlo)
- [Datos](#datos)
- [Modelos](#modelos)
- [Estructura del proyecto](#estructura-del-proyecto)
- [Decisiones de diseño](#decisiones-de-diseño)
- [Entregables](#entregables)
- [Referencias](#referencias)

---

## Qué es

Un **agente que aprende** a distinguir spam de mensajes legítimos y generaliza a mensajes nuevos, en **inglés y español**. Se implementan y comparan tres modelos clásicos, se validan con cross-validation, se afinan con GridSearch y se calibra el umbral para priorizar **Recall** (contexto de seguridad: dejar pasar phishing es más caro que bloquear un mensaje bueno).

Todo el flujo —descarga de datos, preprocesamiento, entrenamiento, evaluación y una demo interactiva— está en **[`spam_detection.ipynb`](spam_detection.ipynb)**. Es autocontenido: descarga sus datasets y los cachea para correr offline.

---

## Resultados

Test set (20 % estratificado, 551 mensajes EN + ES):

| Modelo | Accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| Naive Bayes (BoW) | 0.911 | 0.954 | 0.842 | 0.895 |
| Reg. Logística — TF-IDF palabras (base) | 0.920 | 0.947 | 0.870 | 0.907 |
| Reg. Logística — char n-grams (base) | 0.935 | 0.945 | 0.907 | 0.926 |
| **Reg. Logística — char n-grams (tuneado) ← final** | **0.946** | **0.954** | 0.923 | **0.938** |

Se **tunearon ambos** candidatos (palabras y char n-grams) con `GridSearchCV` y se eligió el de mejor F1 en validación cruzada: **gana char n-grams** (`C=10, min_df=2, ngram=(3,5)`).

**Calibración del threshold:** el umbral óptimo es **0.307** (no 0.5), calibrado con probabilidades *out-of-fold del train* (no sobre el test). Bajándolo, el Recall sube de 0.923 a **0.947** a costa de algo de Precision — decisión consciente por el contexto de seguridad.

**Evaluación cross-lingual:** F1 **inglés 0.971** / **español 0.905** → el español rinde ~7 puntos menos pero se mide **solo con spam genuino** (nativo + seed), sin traducciones que inflen el número.

---

## Cómo ejecutarlo

```bash
# 1. Entorno virtual
python3 -m venv venv
source venv/bin/activate

# 2. Dependencias
pip install --upgrade pip
pip install -r requirements.txt

# 3. Abrir el notebook y ejecutar todo
jupyter notebook spam_detection.ipynb       # o ábrelo en VS Code y "Run All"
```

O ejecutarlo de principio a fin sin abrirlo:

```bash
jupyter nbconvert --to notebook --execute --inplace spam_detection.ipynb
```

La **primera ejecución** descarga los datasets y guarda `data/corpus.parquet`; las siguientes cargan de esa caché (reproducible y **offline**). Las stopwords de NLTK se descargan solas.

> **Demo interactiva:** la última parte del notebook levanta una interfaz **Gradio** donde puedes escribir un mensaje, elegir modelo y mover el threshold, además de ver las métricas y la matriz de confusión en vivo.

---

## Datos

Corpus bilingüe combinado de **~6 290 mensajes**, **19.6 % spam** (desbalanceado, fiel a la realidad):

| Fuente | Idioma | Mensajes | Tipo |
|---|---|---:|---|
| SMS Spam Collection (UCI) | Inglés | 5 574 | benchmark estándar |
| spam_ham_spanish (softecapps) | Español | 1 207 | spam **nativo** |
| Seed de phishing local | Español | 61 | curado a mano (BBVA, SAT, CFE…) |

Se descargan con una cascada de *fallback* (si una fuente falla, prueba la siguiente) y se cachean en `data/corpus.parquet`.

> **Nota.** Se descartó una cuarta fuente *traducida* (`dbarbedillo`, el UCI inglés traducido al español): al coexistir el original (EN) y su traducción (ES), el split aleatorio repartía gemelos entre *train* y *test*, provocando **fuga cross-lingual** que inflaba las métricas. Usar solo español genuino hace la evaluación honesta.

---

## Modelos

| # | Modelo | Vectorización | Tipo |
|---|---|---|---|
| 1 | Naive Bayes | Bag of Words | Generativo (Bayes + independencia) |
| 2 | Regresión Logística | TF-IDF de palabras (1,2) | Discriminativo (pesos + sigmoide) |
| 3 | Regresión Logística | TF-IDF char n-grams (3,5) | Robusto a typos / cross-lingual |

El **modelo final** es el #3 tuneado. Los n-gramas de caracteres miran trozos de letras, lo que los hace robustos a errores de escritura y a la mezcla de idiomas. El pipeline completo (vectorizador + modelo) se serializa en `models/spam_classifier.joblib`.

---

## Estructura del proyecto

```
spam_detection/
├── spam_detection.ipynb        # Notebook principal (todo el pipeline)
├── requirements.txt
├── README.md                   # Este archivo
├── PROGRESS.md                 # Bitácora de avance
├── data/
│   └── corpus.parquet          # Caché del corpus (se genera en la 1ª ejecución)
├── models/
│   └── spam_classifier.joblib  # Modelo final serializado (se genera al ejecutar)
├── docs/
│   ├── documento_tecnico.pdf   # Documento técnico (12 págs)
│   ├── documento_tecnico.md    # Fuente del documento
│   └── img/                    # Figuras del proyecto
└── presentacion/
    └── animada/                # Presentación web animada (abrir index.html)
        ├── index.html
        ├── assets/             # styles.css, app.js, slides.js
        └── img/
```

---

## Decisiones de diseño

| Decisión | Motivo |
|---|---|
| Reemplazar URLs/números por tokens (`__url__`, `__num__`) | La **presencia** de un link/número es señal de spam, aunque el valor concreto no |
| `sklearn.Pipeline` end-to-end | Mismo preprocesamiento en entrenamiento e inferencia; un solo `.joblib`. Evita *data leakage* |
| Undersampling ~1.5:1 + `class_weight="balanced"` | Doble defensa contra el desbalance (~20 % spam) sin que el modelo se sesgue a "ham" |
| Tunear **ambos** modelos y elegir por F1 (CV) | Que decida el rendimiento, no la intuición → ganó char n-grams |
| Calibrar el threshold (0.307) sobre el train | Priorizar Recall sin fuga: el umbral se elige con datos out-of-fold, no con el test |
| Español **solo genuino** (nativo + seed) | Descartar las traducciones del UCI que causaban fuga cross-lingual; evaluación honesta |
| `random_state=42` + caché parquet | Reproducible y ejecutable offline |

---

## Entregables

- **Notebook:** [`spam_detection.ipynb`](spam_detection.ipynb) — análisis completo con lecturas de cada gráfico.
- **Documento técnico:** [`docs/documento_tecnico.pdf`](docs/documento_tecnico.pdf).
- **Presentación animada:** [`presentacion/animada/index.html`](presentacion/animada/index.html) (con guion en `guion.md`).

---

## Referencias

- Almeida, T. A. & Gómez Hidalgo, J. M. (2011). *Contributions to the Study of SMS Spam Filtering.*
- Metsis, V., Androutsopoulos, I. & Paliouras, G. (2006). *Spam Filtering with Naive Bayes — Which Naive Bayes?*
- Pedregosa, F. et al. (2011). *Scikit-learn: Machine Learning in Python.*
- Russell, S. & Norvig, P. (2021). *Artificial Intelligence: A Modern Approach* (4ª ed.).

---

## Licencia
Proyecto académico — curso de Inteligencia Artificial.
