# Detector de Spam Bilingüe (Inglés + Español)

> Clasificador de mensajes **spam vs ham** con Machine Learning sobre un corpus bilingüe (~11 400 mensajes EN + ES). Todo el proyecto vive en un único notebook reproducible.

[![Python](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-orange.svg)](https://scikit-learn.org/)
[![F1](https://img.shields.io/badge/F1-0.950-brightgreen.svg)](#resultados)
[![Recall](https://img.shields.io/badge/Recall-0.981-brightgreen.svg)](#resultados)

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

Test set (20 % estratificado, 932 mensajes EN + ES):

| Modelo | Accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| Naive Bayes (BoW) | 0.924 | 0.934 | 0.871 | 0.902 |
| Reg. Logística — TF-IDF palabras (base) | 0.930 | 0.916 | 0.909 | 0.913 |
| Reg. Logística — char n-grams (base) | 0.938 | 0.911 | 0.936 | 0.923 |
| **Reg. Logística — char n-grams (tuneado) ← final** | **0.959** | 0.942 | **0.957** | **0.950** |

Se **tunearon ambos** candidatos (palabras y char n-grams) con `GridSearchCV` y se eligió el de mejor F1 en validación cruzada: **gana char n-grams** (`C=10, min_df=2, ngram=(3,5)`).

**Calibración del threshold:** el umbral óptimo es **0.253** (no 0.5). Bajándolo, el Recall sube de 0.957 a **0.981** (solo escapa el 1.9 % del spam) a costa de algo de Precision — decisión consciente por el contexto de seguridad.

**Evaluación cross-lingual:** F1 **inglés 0.965** / **español 0.941** → el modelo transfiere bien entre idiomas.

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

Corpus bilingüe combinado de **~11 400 mensajes**, **16.3 % spam** (desbalanceado, fiel a la realidad):

| Fuente | Idioma | Mensajes | Tipo |
|---|---|---:|---|
| SMS Spam Collection (UCI) | Inglés | 5 574 | benchmark estándar |
| SMS Multilingual (dbarbedillo) | Español | 5 572 | UCI traducido |
| spam_ham_spanish (softecapps) | Español | 1 207 | spam **nativo** |
| Seed de phishing local | Español | 61 | curado a mano (BBVA, SAT, CFE…) |

Se descargan con una cascada de *fallback* (si una fuente falla, prueba la siguiente) y se cachean en `data/corpus.parquet`.

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
| Undersampling 1.5:1 + `class_weight="balanced"` | Doble defensa contra el desbalance (16 % spam) sin que el modelo se sesgue a "ham" |
| Tunear **ambos** modelos y elegir por F1 (CV) | Que decida el rendimiento, no la intuición → ganó char n-grams |
| Calibrar el threshold (0.253) | Priorizar Recall por el contexto de seguridad |
| 3 fuentes de español (incl. **nativo**) | No depender solo de traducción automática; evaluación más honesta |
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
