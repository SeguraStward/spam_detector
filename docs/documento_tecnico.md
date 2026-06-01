# Detector de Spam Bilingüe (Inglés + Español)
## Documento Técnico — Proyecto Final

---

**Universidad Nacional**
**Curso:** Inteligencia Artificial
**Profesor:** Venegas
**Integrante(s):** _______________________________
**Fecha de entrega:** 8 de junio de 2026

---

<div class="pagebreak"></div>

## 1. Introducción

El correo y la mensajería electrónica son hoy el principal vector de ataques de
*phishing*, fraude y publicidad no deseada. El **spam** no solo degrada la
experiencia del usuario: es la puerta de entrada a estafas bancarias, robo de
credenciales y distribución de malware. Filtrar automáticamente estos mensajes
es, por tanto, un problema de seguridad de primer orden.

Este proyecto construye un **agente inteligente** capaz de clasificar mensajes
como *spam* o *ham* (legítimo) de forma automática, usando técnicas de
Procesamiento de Lenguaje Natural (NLP) y aprendizaje supervisado. A diferencia
de los filtros basados en reglas fijas (*"si contiene FREE entonces spam"*),
nuestro agente **aprende** las regularidades del lenguaje del spam a partir de
datos, y generaliza a mensajes nunca vistos.

El sistema es **bilingüe** (inglés y español), una decisión deliberada: los
usuarios hispanohablantes reciben campañas de phishing localizadas (BBVA,
Santander, SAT, CFE, Mercado Libre) que un modelo entrenado solo en inglés no
detecta. Para que el español no fuera solo traducción, el corpus combina datos
traducidos, **spam español nativo** y un *seed* de phishing local. El entregable
incluye un notebook reproducible, un modelo serializado y un sistema interactivo
de predicción en Gradio.

## 2. Planteamiento del problema

La detección de spam se formula como un problema de **clasificación binaria
supervisada**. El agente aprende una función

$$f:\ \mathcal{X} \rightarrow \mathcal{Y}$$

que mapea el texto crudo de un mensaje a una etiqueta de clase.

| | Variable | Tipo | Dominio | Descripción |
|---|---|---|---|---|
| **Entrada** $\mathcal{X}$ | `text` | Texto libre | Cadena Unicode de longitud variable (EN/ES) | Contenido crudo del mensaje, con URLs, números y emojis |
| **Salida** $\mathcal{Y}$ | `label` | Categórica binaria | $\{\text{spam}, \text{ham}\}$ | Clase predicha |
| **Salida** (aux.) | `P(spam)` | Continua | $[0, 1]$ | Probabilidad estimada de spam |

**Regla de decisión:** se predice spam si $P(\text{spam}\mid\text{texto}) \ge t^{\star}$,
donde $t^{\star}$ es un umbral calibrado (no necesariamente 0.5).

### Análisis PEAS del agente

| Componente | Definición |
|---|---|
| **P — Performance** | Accuracy, Precision, Recall y F1. Métrica prioritaria: **Recall** sobre la clase *spam*. |
| **E — Environment** | Mensajes en inglés o español, con HTML, URLs, números y errores ortográficos. |
| **A — Actuators** | Etiqueta *spam*/*ham* + nivel de confianza $\in[0,1]$ + probabilidades por clase. |
| **S — Sensors** | El texto crudo del mensaje. |

**Tipo de entorno:** totalmente observable, un solo agente, determinístico,
episódico y estático. **Tipo de agente:** *model-based* — construye una
representación interna (pesos o probabilidades) durante el entrenamiento y la
aplica en inferencia.

> **¿Por qué priorizar Recall?** Un falso negativo (spam que llega a la bandeja
> de entrada) puede derivar en fraude o robo de credenciales; un falso positivo
> (un mensaje legítimo enviado a la carpeta de spam) es una molestia recuperable.
> En un contexto de seguridad, el costo asimétrico justifica maximizar la
> detección de spam aun a costa de bloquear algún mensaje legítimo.

<div class="pagebreak"></div>

## 3. Metodología

### 3.1 Datasets

Se combinaron **tres fuentes** para construir un corpus bilingüe que no dependa
únicamente de traducción automática:

| Dataset | Idioma | Fuente | Mensajes |
|---|---|---|---|
| SMS Spam Collection (UCI) | Inglés | Almeida & Hidalgo (2011) | 5 574 |
| SMS Multilingual (traducido) | Español | dbarbedillo (UCI traducido) | 5 572 |
| **spam_ham_spanish (nativo)** | Español | softecapps (spam escrito en español) | 1 207 |
| *Seed* de phishing local | Español | Curado manualmente (BBVA, SAT, CFE…) | 61 |

Tras la unión y la **deduplicación** por texto, el corpus reúne **~11 400
mensajes** con una proporción global de spam del **16.3 %** — un dataset
**desbalanceado**, fiel a la realidad (la mayoría del correo es legítimo). La
incorporación del **spam español nativo** sube la proporción de spam respecto a
una versión solo-traducida y aporta vocabulario de phishing genuino del español.

![Distribución del corpus por idioma y clase](img/exploracion.png)

### 3.2 Preprocesamiento

Cada mensaje pasa por una función `clean_text` que:

1. Convierte a minúsculas.
2. Sustituye URLs por el token `__url__` y números por `__num__` (preservando la
   *señal* sin memorizar valores concretos).
3. Elimina HTML, puntuación y caracteres no informativos.
4. Filtra **stopwords** combinadas de inglés y español (**504** palabras de
   NLTK).

El reemplazo por *tokens* en lugar de la eliminación es una decisión clave: la
presencia de una URL o un número es altamente predictiva de spam, aunque su
valor exacto no lo sea.

### 3.3 Balanceo y partición

Para evitar que el modelo aprenda a predecir siempre *ham*, se aplicó
**undersampling** de la clase mayoritaria, llevando el corpus a **4 659
mensajes** con ~40 % de spam. Se dividió de forma **estratificada** en
**train (3 727)** y **test (932)**, con `random_state=42` para reproducibilidad.
La proporción de spam se mantiene en ~40 % en ambas particiones, y el test
conserva los dos idiomas (314 EN / 618 ES).

### 3.4 Vectorización y modelos

Se entrenaron y compararon **tres modelos**, cada uno con una estrategia de
representación distinta:

| # | Modelo | Vectorización | Intuición |
|---|---|---|---|
| 1 | **Naive Bayes** Multinomial | Bag of Words | Generativo; cuenta frecuencias de palabra por clase |
| 2 | **Regresión Logística** | TF-IDF de palabras | Discriminativo; pondera palabras por relevancia |
| 3 | **Regresión Logística** | TF-IDF de *char n-grams* (`char_wb`) | Robusto a errores ortográficos y agnóstico al idioma |

### 3.5 Validación, tuning y calibración

- **Cross-validation k-fold (k=5):** se midió la brecha *train* vs *CV* para
  detectar **overfitting**.
- **GridSearchCV:** búsqueda de hiperparámetros (18 combinaciones × 3 folds) sobre
  `C`, `min_df` y `ngram_range`.
- **Calibración de threshold:** se eligió el umbral que **maximiza Recall**
  manteniendo Precision ≥ 0.85, en lugar del 0.5 por defecto.

<div class="pagebreak"></div>

## 4. Resultados

### 4.1 Comparación de los tres modelos (test set)

| Modelo | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| Naive Bayes (BoW) | 0.9238 | 0.9339 | 0.8713 | 0.9015 |
| Regresión Logística (TF-IDF palabras) | 0.9303 | 0.9162 | 0.9088 | 0.9125 |
| Regresión Logística (char n-grams) | 0.9378 | 0.9112 | **0.9357** | **0.9233** |

![Comparación de métricas por modelo](img/comparativa.png)

Con el corpus enriquecido con español nativo, el modelo de **char n-grams** es el
**mejor sin tunear** (F1 0.923, mejor Recall): sus *n*-gramas de caracteres son
agnósticos al idioma y resisten el estilo del spam nativo. Aun así, se eligió la
**Regresión Logística de palabras como base para el tuning** por su
interpretabilidad; tras `GridSearchCV` y calibración alcanza el mejor rendimiento
global (ver 4.2).

### 4.2 Anti-overfitting y tuning

La cross-validation reveló brechas *train–CV* en F1 de **+0.039** (Naive Bayes),
**+0.051** (Logística palabras) y **+0.036** (Logística char). El modelo de
palabras muestra una **señal leve de sobreajuste** —su vocabulario disperso
memoriza algo del train—, mientras que **char n-grams generaliza mejor**. La
curva de aprendizaje confirma la tendencia (F1 train 0.999 vs validación 0.939).
Mitigamos esto con regularización (`C`), `min_df` y la calibración del threshold.

El `GridSearchCV` encontró como mejores hiperparámetros `C = 10.0`,
`min_df = 1`, `ngram_range = (1,1)`, con **F1 macro en CV de 0.939**. El modelo
final tuneado obtiene en el test set:

| Métrica | Valor |
|---|---|
| Accuracy | **0.9549** |
| Precision (spam) | 0.9413 |
| Recall (spam) | 0.9464 |
| F1 (spam) | **0.9439** |

### 4.3 Calibración del threshold

![Curva Precision–Recall y threshold óptimo](img/curva_pr.png)

El umbral óptimo resultó **t\* = 0.278** (frente a 0.5 por defecto). El efecto
sobre la clase *spam* es directo:

| Threshold | Precision | Recall | F1 |
|---|---|---|---|
| 0.500 (defecto) | 0.9413 | 0.9464 | 0.9439 |
| **0.278 (óptimo)** | 0.8505 | **0.9759** | 0.9089 |

Bajar el umbral eleva el Recall del 94.6 % al **97.6 %** — coherente con la
prioridad de seguridad — a cambio de una caída tolerable de Precision.

### 4.4 Evaluación *cross-lingual* (por idioma)

![Métricas por idioma](img/por_idioma.png)

| Idioma | n | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|---|
| Inglés (EN) | 314 | 0.9682 | 0.9474 | 0.9767 | 0.9618 |
| Español (ES) | 618 | 0.9482 | 0.9380 | 0.9303 | 0.9342 |

El modelo funciona en ambos idiomas con ~3 puntos de diferencia en F1. El
español rinde algo por debajo del inglés: su test es el doble de grande y mucho
más diverso (incluye spam **nativo** real, no solo traducción), por lo que
0.934 es una medida **más honesta y exigente** de la capacidad del modelo en
español que la de un corpus puramente traducido.

### 4.5 Matriz de confusión y análisis de errores

![Matriz de confusión](img/matriz_confusion.png)

Sobre los 932 mensajes de test: **537** verdaderos negativos, **353**
verdaderos positivos, **22** falsos positivos (ham bloqueado) y **20** falsos
negativos (spam no detectado). En total **42 errores (4.51 %)**. Al inspeccionar
los errores, los falsos negativos corresponden a mensajes cortos o ambiguos
(*"participa en el sorteo de un iphone"*) y los falsos positivos a mensajes
legítimos con vocabulario inusual — fallos esperables y de bajo impacto.

### 4.6 Sistema interactivo de predicción

El proyecto incluye una **interfaz web con Gradio** que permite clasificar
mensajes nuevos en vivo. La interfaz expone tres controles: (i) un campo de
texto para el mensaje, (ii) un **selector de modelo** entre los cuatro
entrenados —cumpliendo el objetivo de planeación de "comparar enfoques"— y
(iii) un **deslizador de threshold**. Al variar el umbral, el usuario observa
directamente el trade-off de la Sección 4.3: bajarlo a 0.28 marca como spam
mensajes dudosos (más Recall), mientras que el valor por defecto de 0.5 los deja
pasar (más Precision). La salida muestra las probabilidades por clase y la
decisión final según el umbral elegido.

<div class="pagebreak"></div>

## 5. Discusión

**Trade-off Precision/Recall.** La calibración del threshold materializa la
decisión de diseño del PEAS: priorizar la detección de spam. Pasar de 0.5 a 0.278
sube el Recall a 97.6 %, dejando pasar solo el 2.4 % del spam, a costa de
bloquear algunos *ham*. En un filtro real esto se mitiga enviando los positivos a
una carpeta de cuarentena revisable, no eliminándolos.

**Español nativo vs traducido.** Incorporar spam español nativo (softecapps)
hizo la evaluación más realista y dotó al modelo de vocabulario de phishing
genuino. El F1 en español (0.934) es algo menor que en inglés (0.962), pero se
mide sobre un test más grande y diverso; preferimos un número honesto a uno
inflado por traducciones. Se probó además un filtrado de etiquetas ruidosas por
*confident learning*, pero se descartó: el modelo de referencia, al no conocer la
distribución nativa, eliminaba spam correctamente etiquetado, así que se optó por
conservar la fuente íntegra.

**Sobre los char n-grams.** Con el corpus enriquecido, el modelo de char n-grams
pasó a ser el **mejor sin tunear** y el de **mejor generalización** (menor brecha
train–CV). Confirma su valor frente a texto multilingüe y estilísticamente
variado; sería el candidato natural si se priorizara robustez sobre
interpretabilidad.

**Sobreajuste controlado.** El modelo de palabras mostró una brecha train–CV
moderada (+0.051) y una curva de aprendizaje con separación (~0.06), señal de
sobreajuste leve atribuible a su vocabulario disperso. Se contiene con
regularización y `min_df`; añadir más datos nativos seguiría ayudando.

**Limitaciones.** (i) El dataset español todavía incluye una porción traducida y
el nativo es de tamaño modesto; (ii) el corpus es de SMS/mensajes cortos, no
correos largos con HTML; (iii) el modelo es estático: el spam evoluciona y
requeriría reentrenamiento periódico.

## 6. Conclusiones

1. Se construyó un detector de spam **bilingüe** end-to-end que alcanza
   **F1 = 0.944** y **Accuracy = 0.955** en el conjunto de prueba.
2. El tuning de hiperparámetros y la **calibración del threshold** (t\* = 0.278)
   elevaron el Recall sobre spam al **97.6 %**, alineado con el objetivo de
   seguridad.
3. El corpus combina inglés, español **traducido y nativo** y phishing local,
   evaluándose de forma *cross-lingual* (F1 EN 0.962 / ES 0.934). El modelo de
   char n-grams resultó el más robusto entre idiomas.
4. Se entregó un **sistema interactivo en Gradio** que clasifica mensajes nuevos
   en vivo, permite **seleccionar el modelo** entre los cuatro entrenados y
   **ajustar el threshold** para explorar el trade-off Precision/Recall, además
   de un modelo serializado reutilizable.

**Trabajo futuro:** ampliar el spam español nativo, probar modelos *transformer*
multilingües (p. ej. mBERT), y desplegar el filtro con reentrenamiento
incremental.

## 7. Referencias

Almeida, T. A., & Gómez Hidalgo, J. M. (2011). *Contributions to the study of SMS
spam filtering: New collection and results.* En Proceedings of the 11th ACM
Symposium on Document Engineering (DocEng '11) (pp. 259–262). ACM.

Metsis, V., Androutsopoulos, I., & Paliouras, G. (2006). *Spam filtering with
Naive Bayes — Which Naive Bayes?* En Proceedings of the 3rd Conference on Email
and Anti-Spam (CEAS).

Northcutt, C. G., Jiang, L., & Chuang, I. (2021). *Confident learning: Estimating
uncertainty in dataset labels.* Journal of Artificial Intelligence Research, 70,
1373–1411.

Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O.,
… Duchesnay, É. (2011). *Scikit-learn: Machine learning in Python.* Journal of
Machine Learning Research, 12, 2825–2830.

Russell, S., & Norvig, P. (2021). *Artificial Intelligence: A Modern Approach*
(4.ª ed.). Pearson.
