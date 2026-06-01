# Roadmap de dominio — Detector de Spam Bilingüe

> Objetivo: entender **cada concepto y cada línea** del notebook tan a fondo que
> puedas defenderlo como un profesional. Marca cada casilla cuando puedas
> **explicarlo en voz alta sin leer** y **responder el "¿por qué?"**.
>
> Regla de oro para la defensa: por cada decisión técnica ten lista la frase
> *"lo hice así **porque** ___, y la alternativa habría sido ___"*.

Leyenda de esfuerzo: 🟢 base · 🟡 intermedio · 🔴 clave para la defensa

---

## Cómo usar este roadmap

1. Sigue los niveles **en orden** (cada uno asume el anterior).
2. En cada tema: (a) lee la celda del notebook, (b) explícate el *qué*, (c)
   explícate el *por qué*, (d) responde las preguntas de defensa.
3. Prueba la **técnica de Feynman**: explícalo como si enseñaras a un compañero.
   Si te trabas, ahí está tu laguna.
4. Plan sugerido: ~6 sesiones de 1–2 h. No memorices código: entiende el patrón.

---

## Nivel 0 — Python que aparece en el notebook 🟢

Sin esto, todo lo demás es copiar y pegar. Domina estos idiomas del lenguaje.

- [ ] **Imports y namespaces** — `import x`, `from x import y`, alias (`import pandas as pd`). ¿Por qué alias?
- [ ] **f-strings** — `f"{len(df):,}"`, `f"{x:.2%}"`, `f"{x:.4f}"`. Sabe formatear miles, porcentajes y decimales.
- [ ] **Comprensiones** — listas (`[w for w in t.split() if w not in STOP]`) y dicts (`{c: p for c, p in zip(...)}`).
- [ ] **`zip`, `enumerate`, `.get(k, default)`** sobre dicts.
- [ ] **Funciones**: argumentos por defecto (`def predict_message(text, pipeline=best_pipeline, threshold=0.5)`), `*args`/keyword.
- [ ] **`try/except`** y jerarquía de excepciones (capturar `requests.exceptions.SSLError` antes que `Exception`).
- [ ] **`pathlib.Path`** — `Path("data")`, `/` para unir rutas, `.mkdir(exist_ok=True)`, `.exists()`, `.stat()`.
- [ ] **Sets y operador `|`** — `set(stopwords_en) | set(stopwords_es)` (unión). ¿Por qué set y no lista? (búsqueda O(1)).
- [ ] **Slicing y `str` methods** — `.lower()`, `.strip()`, `.split()`, `.startswith()`, `str[:90]`.
- [ ] **`if __name__ == "__main__"`** y la idea de módulo vs script.

**Defensa:** *"¿Por qué un `set` para las stopwords y no una lista?"* → Pertenencia
(`in`) es O(1) en set vs O(n) en lista; con 504 stopwords y miles de tokens, importa.

---

## Nivel 1 — NumPy y pandas 🟢🟡

La columna vertebral del manejo de datos. Secciones 7, 8, 11, 12 del notebook.

- [ ] **`pd.DataFrame`** desde lista de dicts y desde CSV/parquet. Filas, columnas, índice.
- [ ] **Selección**: `df["col"]`, `df[["a","b"]]`, `df[mask]` (máscara booleana), `.loc` vs `.iloc`.
- [ ] **`.apply(func)`** — `df["text"].apply(clean_text)`. ¿Cuándo evitarlo por vectorización?
- [ ] **`value_counts()`**, **`groupby([...]).size().unstack(fill_value=0)`** (tabla de contingencia idioma×clase).
- [ ] **`concat`, `drop_duplicates(subset=...)`, `dropna`, `reset_index(drop=True)`**.
- [ ] **Strings vectorizados**: `df["text"].str.len()`, `.str.strip()`, `.str.lower()`, `.str.isin([...])`.
- [ ] **`to_parquet` / `read_parquet`** y por qué parquet (columnar, comprimido, tipado) vs CSV.
- [ ] **NumPy**: `np.array`, indexado booleano (`vals[mask]`), `np.argmax`, broadcasting, `random_state`/`np.random.seed`.
- [ ] **Matrices dispersas (sparse)** — qué devuelve un vectorizador y por qué no es un array denso (memoria).

**Defensa:** *"¿Qué es `unstack` y por qué lo usas?"* → Convierte un groupby
multinivel en tabla 2D legible (idioma en filas, clase en columnas).

---

## Nivel 2 — Probabilidad y los dos modelos (teoría) 🔴

Esto es el corazón conceptual. Secciones 2 y 13. **Imprescindible** para la defensa.

### Naive Bayes
- [ ] **Teorema de Bayes**: $P(spam\mid x) = \dfrac{P(x\mid spam)\,P(spam)}{P(x)}$. Sabe nombrar cada término (posterior, verosimilitud, prior, evidencia).
- [ ] **Suposición "naive"**: independencia condicional de las palabras dada la clase. ¿Por qué es falsa pero útil?
- [ ] **Multinomial NB**: modela conteos de palabras. Por qué encaja con Bag of Words.
- [ ] **Suavizado de Laplace (`alpha=1.0`)**: evita probabilidad cero para palabras no vistas. ¿Qué pasa sin él?
- [ ] **Generativo**: modela cómo se generan los datos por clase.

### Regresión Logística
- [ ] **Función sigmoide** $\sigma(z)=\dfrac{1}{1+e^{-z}}$ y **log-odds** $z = w\cdot x + b$.
- [ ] **Frontera de decisión lineal** sobre el espacio TF-IDF.
- [ ] **Entrenamiento**: descenso de gradiente minimizando la *log-loss* (entropía cruzada).
- [ ] **Regularización y `C`**: `C` = inverso de la fuerza de regularización (C grande = menos regularización). L1 vs L2; `solver="liblinear"`.
- [ ] **`class_weight="balanced"`**: penaliza más errores en la clase minoritaria. Relación con el desbalance.
- [ ] **Discriminativo**: aprende la frontera directa, no la generación.

**Defensa (las preguntas estrella):**
- *"¿Generativo vs discriminativo?"* → NB modela $P(x\mid clase)$; LogReg modela $P(clase\mid x)$ directo.
- *"¿Qué hace `C=10`?"* → Poca regularización → el modelo se ajusta más (riesgo de overfitting, que vimos).
- *"¿Por qué `class_weight='balanced'`?"* → Compensa que hay ~13–16 % de spam antes del balanceo.

---

## Nivel 3 — De texto a números: vectorización 🔴

Secciones 9 y 10. El puente entre lenguaje y álgebra.

### Preprocesamiento (`clean_text`)
- [ ] **Regex**: `re.compile(r"https?://\S+|www\.\S+")`, `re.sub`. Lee cada patrón y di qué captura.
- [ ] **Tokens `__url__` / `__num__`**: por qué **reemplazar** en vez de **borrar** (la *presencia* es señal, el valor no).
- [ ] **`html.unescape`**, minúsculas, quitar puntuación, filtrar stopwords EN+ES.
- [ ] Por qué el preprocesamiento debe ser **idéntico** en train, test e inferencia (y por eso vive en el Pipeline).

### Vectorización
- [ ] **Bag of Words (`CountVectorizer`)**: vocabulario + conteos. `ngram_range=(1,2)` = unigramas + bigramas.
- [ ] **TF-IDF (`TfidfVectorizer`)**: $\text{tf} \times \log\frac{N}{df}$. Qué hace `sublinear_tf=True` (usa $1+\log tf$).
- [ ] **`min_df` / `max_df`**: descartar términos demasiado raros o demasiado comunes. Efecto en overfitting y ruido.
- [ ] **`analyzer="char_wb"`, `ngram_range=(3,5)`**: *n*-gramas de caracteres dentro de palabra. Por qué son **robustos a typos y agnósticos al idioma** (clave cross-lingual).
- [ ] **Vocabulario y matriz documento-término dispersa**.

**Defensa:** *"¿Por qué char n-grams para lo bilingüe?"* → Capturan subcadenas
(p. ej. "gan", " premi") que se repiten entre idiomas y resisten ofuscación;
por eso fue el modelo más robusto al añadir español nativo.

---

## Nivel 4 — Entrenamiento honesto: splits, balanceo, pipeline 🔴

Secciones 11, 12, 13. Aquí se gana o se pierde la credibilidad de los resultados.

- [ ] **Desbalance de clases**: por qué accuracy engaña (un "todo ham" acierta ~85 %).
- [ ] **Undersampling**: igualar clases muestreando la mayoritaria. Pro/contra vs oversampling/SMOTE.
- [ ] **`train_test_split(stratify=y, random_state=42)`**: qué es estratificar y por qué fija la semilla.
- [ ] **`sklearn.pipeline.Pipeline`**: encadenar `vec → clf`. **Por qué evita *data leakage*** (el vectorizador se ajusta solo con train en cada fold).
- [ ] **API de scikit-learn**: `fit`, `predict`, `predict_proba`, `.classes_`. Qué devuelve cada uno y en qué orden salen las clases.

**Defensa (muy probable):** *"¿Cómo evitas data leakage?"* → El `TfidfVectorizer`
va **dentro** del Pipeline, así que en cross-validation se ajusta con el train de
cada fold y nunca "ve" el test. Estratificar mantiene la proporción de spam.

---

## Nivel 5 — Evaluación y métricas 🔴

Secciones 13 (helper), 19. **El examinador va a preguntar mucho aquí.**

- [ ] **Matriz de confusión**: TP, TN, FP, FN. Sabe ubicarlos en la figura del notebook (537/22/20/353).
- [ ] **Precision** $=\frac{TP}{TP+FP}$ vs **Recall** $=\frac{TP}{TP+FN}$. Di con palabras qué mide cada una.
- [ ] **F1** = media armónica. Por qué armónica y no aritmética (penaliza desbalance entre P y R).
- [ ] **`pos_label="spam"`**: por qué importa definir la clase positiva.
- [ ] **`classification_report`**: leer macro avg vs weighted avg.
- [ ] **Trade-off Precision/Recall** y por qué priorizamos Recall (costo asimétrico del phishing).
- [ ] **Análisis de errores**: leer falsos positivos/negativos reales y explicarlos.

**Defensa:** *"¿Por qué F1 y no accuracy?"* y *"¿Precision o Recall, y por qué?"*
→ Ten la respuesta del costo asimétrico memorizada (un FN = phishing que entra).

---

## Nivel 6 — Validar y afinar: CV, GridSearch, curva de aprendizaje 🔴

Secciones 14, 15, 16. Demuestra rigor metodológico.

- [ ] **Sobreajuste (overfitting)** y **bias-variance**: qué es y cómo se ve (gap train–validación).
- [ ] **`StratifiedKFold(n_splits=5)`** + **`cross_validate`** con `return_train_score`. Qué es la **brecha train–CV** (en tu modelo de palabras fue **+0.051**, señal leve).
- [ ] **`GridSearchCV`**: búsqueda exhaustiva sobre `param_grid` (`C`, `min_df`, `ngram_range`), `cv=3`, `scoring="f1_macro"`, `n_jobs=-1`. Cuántos *fits* (18×3=54) y por qué.
- [ ] **`grid.best_params_` / `best_estimator_`**: qué encontró (`C=10, min_df=1, ngram=(1,1)`).
- [ ] **`learning_curve`**: F1 vs tamaño de train; tu resultado (train 0.999 / val 0.939) y qué implica (sobreajuste leve, más datos ayudarían).

**Defensa:** *"¿Cómo sabes que no sobreajusta?"* → Sé honesto: el modelo de
palabras **sí** muestra sobreajuste leve (gap +0.051), por eso char n-grams
(gap +0.036) generaliza mejor; lo mitigamos con `C`, `min_df` y el threshold.

---

## Nivel 7 — Calibración del threshold 🔴 (tu pieza diferenciadora)

Sección 17. Es lo que distingue tu proyecto de un tutorial básico.

- [ ] **`predict_proba` vs `predict`**: la decisión 0.5 es arbitraria.
- [ ] **`precision_recall_curve`**: cómo se barre el umbral y se obtienen pares (P, R).
- [ ] **Regla del proyecto**: maximizar Recall manteniendo Precision ≥ 0.85 → `argmax` sobre la máscara. Resultado **t\* = 0.278**.
- [ ] **Efecto**: a 0.5 → R 0.946; a 0.278 → R **0.976** (Precision baja a 0.851). Trade-off explícito.
- [ ] Conexión con producción: positivos a **cuarentena**, no a borrado.

**Defensa:** *"¿Por qué bajar el umbral a 0.278?"* → Para subir Recall (detectar
más spam) según la prioridad del PEAS, aceptando algún falso positivo manejable.

---

## Nivel 8 — Evaluación cross-lingual y datos en español 🟡🔴

Secciones 6, 8, 18 + la sesión de datasets.

- [ ] **Las 3 fuentes españolas**: traducido (dbarbedillo), **nativo (softecapps)**, seed local. Qué aporta cada una.
- [ ] **Cascada de *fallback* y caché parquet**: robustez ante fallos de red / evaluación offline.
- [ ] **Evaluación por idioma**: por qué medir EN y ES por separado (EN F1 0.962 / ES F1 0.934).
- [ ] **Confident learning (lo que probamos y descartamos)**: qué es, por qué un modelo de referencia entrenado en otra distribución **borraba spam nativo bueno**, y por qué fue correcto descartarlo.

**Defensa (pregunta trampa):** *"¿Tu español es solo traducción?"* → No: añadimos
spam nativo real; la evaluación es más honesta aunque el F1 baje un poco.

---

## Nivel 9 — Producto: persistencia e interfaz 🟡

Secciones 20, 21.

- [ ] **`joblib.dump/load`**: serializar el Pipeline + threshold + métricas en un artefacto. Por qué joblib y no pickle puro.
- [ ] **`predict_message()`**: cómo reusa `clean_text` + `predict_proba` + threshold y devuelve etiqueta/confianza.
- [ ] **Gradio**: `gr.Interface`, `gr.Textbox`, `gr.Dropdown` (selector de modelo), `gr.Slider` (threshold), `gr.Label`. `prevent_thread_lock=True` para no bloquear la ejecución.
- [ ] Saber **lanzar la demo en vivo** y tener plan B (vista previa estática guardada).

---

## Nivel 10 — Visión de IA del curso (PEAS y entornos) 🔴

Secciones 1–3. Conecta el código con la teoría de agentes inteligentes.

- [ ] **PEAS** completo: Performance, Environment, Actuators, Sensors — sabe justificar cada celda.
- [ ] **Tipo de entorno**: observable, un agente, determinístico, episódico, estático. Justifica cada dimensión.
- [ ] **Tipo de agente**: *model-based* (aprende un modelo interno y lo aplica en inferencia).
- [ ] **Formulación E/S**: $f:\text{texto}\to\{spam,ham\}+P(spam)$.

**Defensa:** *"¿Por qué episódico?"* → Cada mensaje se clasifica independiente; una
decisión no afecta la siguiente.

---

## Banco de preguntas de defensa (simulacro)

Responde en voz alta, cronometrado (~30 s c/u):

1. ¿Naive Bayes vs Regresión Logística — cuándo gana cada uno?
2. ¿Qué es TF-IDF y por qué no usar solo conteos (BoW)?
3. ¿Por qué char n-grams ayudan en un corpus bilingüe?
4. ¿Cómo evitas el data leakage en la validación?
5. ¿Por qué accuracy es engañosa aquí y qué métrica priorizas?
6. ¿Qué significa que el umbral óptimo sea 0.278 y no 0.5?
7. ¿Tu modelo sobreajusta? Demuéstralo con números.
8. ¿Por qué `C=10` y qué pasaría con `C=0.01`?
9. ¿Qué aporta el dataset español nativo y por qué bajó un poco el F1?
10. ¿Qué es `class_weight="balanced"` y por qué lo usas?
11. ¿Qué pasa si llega un mensaje en un idioma no visto?
12. Si tuvieras más tiempo, ¿qué mejorarías? (→ transformers multilingües, más spam nativo).

---

## Recursos recomendados

- **scikit-learn User Guide** — Text feature extraction, Pipeline, Model evaluation (docs oficiales).
- **Russell & Norvig**, *AIMA* (4ª ed.) — capítulos de agentes inteligentes y aprendizaje.
- **StatQuest (YouTube)** — Naive Bayes, Logistic Regression, ROC/AUC, Cross-validation (intuición visual).
- **3Blue1Brown** — intuición de gradiente y funciones.
- El propio **[documento técnico](documento_tecnico.md)** y la **[presentación](presentacion.md)** de este proyecto como guion base.

---

## Checklist final "estoy listo para presentar"

- [ ] Explico **cada sección** del notebook en 1–2 frases sin leer.
- [ ] Respondo las 12 preguntas del banco sin dudar.
- [ ] Sé **lanzar la demo de Gradio** y tengo plan B.
- [ ] Conozco **mis números reales** de memoria: F1 0.944, Acc 0.955, Recall 0.976 (t=0.278), EN 0.962 / ES 0.934.
- [ ] Puedo **defender las decisiones** (Recall sobre Precision, char n-grams, español nativo, sobreajuste leve).
- [ ] Tengo lista una respuesta honesta para *"¿qué falló o qué mejorarías?"*.
