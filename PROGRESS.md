# Progreso del Proyecto — Detector de Spam

> Actualizado automaticamente por el mentor al final de cada sesion.

---

## Estado actual

- **Fase:** Entrega final — brechas de rubrica cerradas (notebook + PDF + presentacion)
- **Ultimo paso completado:** Sistema interactivo (ipywidgets+Gradio), cache offline, tabla E/S, notebook ejecutado con outputs, documento tecnico PDF (11 pags) y esqueleto de presentacion
- **Resultado real (test set, modelo final = char n-grams tuneado):** Accuracy 0.959 / F1 0.950 / Recall(spam) 0.957; con threshold optimo 0.253 -> Recall 0.981. Cross-lingual: EN F1 0.965 / ES F1 0.941
- **Modelo final:** se tunearon AMBOS (palabras y char n-grams) con GridSearch y gano char n-grams (C=10, min_df=2, ngram (3,5); F1 CV 0.946 vs 0.939 palabras)
- **Corpus:** EN 5,574 + ES (traducido 5,572 + NATIVO softecapps 1,207 + seed 61) ~= 11,400 msgs, 16.3% spam

---

## Historial de sesiones

### Sesion 0 — Planeacion inicial (Abril 2026)

**Lo que se hizo:**
- Se definio el problema: clasificador de spam en correos electronicos
- Se completo el analisis PEAS
- Se selecciono el dataset: Enron-Spam (~33,700 correos)
- Se definieron los dos algoritmos: Naive Bayes y Regresion Logistica
- Se creo PLANNING.md con requisitos, arquitectura y plan de implementacion
- Se creo README.md con informacion del proyecto

**Conceptos vistos:** PEAS, tipos de agente, tipos de entorno, Naive Bayes (teoria), Regresion Logistica (teoria)

**Archivos creados:** `PLANNING.md`, `README.md`

---

### Sesion 1 — Entorno, Dataset y Preprocesamiento (Abril 2026)

**Lo que se hizo:**
- Entorno virtual creado con `venv` y activado
- Dependencias instaladas: `pandas`, `scikit-learn`, `nltk`, `numpy`
- Creado `requirements.txt`
- Dataset Enron-Spam descargado: 2,551 ham + 501 spam en `data/raw/`
- Explorado formato de los correos (headers + body)
- Aprendido `pathlib`: `Path`, `iterdir()`, `glob()`, operador `/`
- Creado `src/load_dataset.py` con funcion `load_dataset()` que retorna DataFrame
- Verificado DataFrame: 3,052 filas, 2 columnas (`text`, `label`)
- Aprendido `value_counts()` de pandas
- Creado `src/preprocessor.py` con clases limpias sin codigo suelto
- Aprendido patron `if __name__ == "__main__"`
- Stopwords de nltk en ingles y espanol combinadas con operador `|` de sets

**Conceptos dominados esta sesion:**
- Entornos virtuales, `venv`, `pip`, dependencias transitivas
- `pathlib`: navegacion de archivos, `iterdir()`, `glob()`
- Listas de diccionarios → DataFrame con `pd.DataFrame()`
- `value_counts()` de pandas
- Herencia en Python, cadena de clases
- Patron `if __name__ == "__main__"`
- Sets y operador `|` para union

**Archivos creados:** `requirements.txt`, `src/load_dataset.py`, `src/preprocessor.py`

---

### Sesion 2 — Cierre de brechas para entrega final (29 mayo 2026)

**Lo que se hizo (modo ejecucion):**
- A4: tabla formal de variables de entrada/salida (seccion 3.1) en el notebook
- A3: cache del corpus en `data/corpus.parquet` con carga offline-first (celdas de descarga ahora son cache-aware)
- A1: sistema interactivo de ingreso manual con **ipywidgets** (boton + barra de confianza) + interfaz **Gradio**, reutilizando `predict_message()`, `best_pipeline` y `optimal_threshold`; incluye fallback `input()`
- A2: notebook ejecutado end-to-end con `nbconvert --execute`; 32 celdas de codigo, 0 errores, outputs guardados
- B: documento tecnico `docs/documento_tecnico.pdf` (11 paginas, estructura de rubrica, numeros reales, 5 figuras, referencias APA) + fuente `.md` + script `_build_pdf.py`
- C: `docs/presentacion.md` (Marp, 14 slides + apendice Q&A con notas del orador)
- `requirements.txt`: + `gradio`, `ipywidgets`, `pyarrow`

**Archivos creados:** `docs/documento_tecnico.{md,pdf}`, `docs/presentacion.md`, `docs/_build_pdf.py`, `docs/img/*.png`, `data/corpus.parquet`
**Archivos modificados:** `spam_detection.ipynb`, `requirements.txt`, `PROGRESS.md`

**Pendiente para el usuario:** rellenar nombre(s) en portada del PDF y slides; ensayar la demo de Gradio para la defensa.

---

### Sesion 3 — Spam espanol nativo + interfaz Gradio unica (29 mayo 2026)

**Lo que se hizo (modo ejecucion):**
- Sistema interactivo: se anadio selector de modelo (4) + slider de threshold; luego se ELIMINO ipywidgets dejando solo Gradio (autocontenida)
- Dataset: integrado spam ESPANOL NATIVO `softecapps/spam_ham_spanish` (1,207 msgs, ~51% spam) como 3a fuente del espanol, con fallback y etiquetado de `source`
- Se probo filtrado de ruido por *confident learning* y se DESCARTO: el modelo de referencia (entrenado en traducido) eliminaba spam nativo bien etiquetado -> bajaba ES F1. Se conserva la fuente integra
- Notebook reejecutado (cache invalidada). Hallazgo honesto: con nativo, char n-grams es el mejor base (F1 0.923) y el mas robusto; el modelo de palabras tuneado sigue ganando global (F1 0.944) pero con sobreajuste leve (gap CV +0.051)
- Docs refrescados con numeros nuevos: `documento_tecnico.pdf` (11 pags) + `presentacion.md` + 5 figuras regeneradas

**Decision pendiente del usuario:** mantener el nativo (ES F1 0.934, mas honesto) o revertir a solo-traducido (ES F1 0.946). Actualmente: MANTENIDO sin filtrar.

---

### Sesion 4 — Modelo final = char n-grams + presentacion animada (jun 2026)

**Lo que se hizo:**
- Se cambio el criterio: en vez de elegir el modelo "a ojo", el notebook ahora **tunea AMBOS** (palabras y char n-grams) con GridSearch y elige el de mejor F1 (CV). **Gana char n-grams** (F1 test 0.950, Recall 0.957, threshold 0.253). Mejora vs el de palabras (0.944).
- Reejecutado el notebook (0 errores), regeneradas todas las figuras (incl. comparacion de matrices 0.5 vs calibrado) y el modelo `.joblib`.
- Actualizados con los numeros nuevos: `docs/documento_tecnico.pdf` (12 pags) y el **deck animado** (`presentacion/animada/`: slides.js, guion.md, notas).
- Presentacion animada: carta blanca con pliegue triangular que se abre, sello ham/spam aleatorio al avanzar, slide "Modelos utilizados", union de metricas+matriz, reorden coherente de slides.

**Pendiente (numeros viejos, NO sincronizados):** deck estatico `presentacion/presentacion.md` y `presentacion/guion.md`, `presentacion/README.md`, `docs/presentacion.md`, `docs/roadmap_dominio.md` (aun citan F1 0.944 / modelo palabras).

---

## Fases del proyecto

| Fase | Descripcion | Estado |
|------|-------------|--------|
| 1 | Entorno y dependencias | Completada |
| 2 | Dataset (descarga y exploracion) | Completada |
| 3 | Preprocesamiento | Completada |
| 4 | Vectorizacion (BoW + TF-IDF + char_wb) | Completada |
| 5 | Naive Bayes (sklearn MultinomialNB) | Completada |
| 6 | Regresion Logistica | Completada |
| 7 | Evaluacion (accuracy/precision/recall/F1/CM) | Completada |
| 8 | Pipeline completo (sklearn Pipeline + joblib) | Completada |
| 9 | CLI (argparse) | Completada |
| 10 | Soporte bilingue EN+ES + char n-grams | Completada |

---

## Conceptos dominados

*(El mentor actualiza esta lista conforme el usuario demuestra comprension)*

- [x] Entornos virtuales en Python
- [x] Manejo de archivos con `pathlib`
- [x] DataFrames con `pandas`
- [x] Preprocesamiento de texto (tokenizacion, stopwords)
- [x] Vectorizacion (Bag of Words, TF-IDF, char_wb)
- [x] Naive Bayes — implementacion manual (`naive_bayes.py` historico)
- [x] Naive Bayes — scikit-learn
- [x] Regresion Logistica (palabras + char n-grams)
- [x] Metricas de evaluacion (Precision, Recall, F1, matriz de confusion)
- [x] Pipelines de scikit-learn
- [x] Serializacion de modelos con `joblib`
- [x] CLI con `argparse`
- [x] Dataset bilingue (ingles + espanol) combinado y deduplicado
