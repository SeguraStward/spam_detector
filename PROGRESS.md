# Progreso del Proyecto — Detector de Spam

> Actualizado automaticamente por el mentor al final de cada sesion.

---

## Estado actual

- **Fase:** Completado — pipeline bilingue end-to-end
- **Ultimo paso completado:** Entrenamiento, evaluacion y CLI funcionando con datasets EN+ES combinados
- **Resultado:** Logistic + TF-IDF (1,2) alcanza F1=0.955 / Accuracy=0.987 sobre test set

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
