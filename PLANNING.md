# Planeacion del Proyecto — Detector de Spam

> Curso de Inteligencia Artificial  
> Fecha: Abril 2026  
> Estado: En planeacion

---

## Tabla de Contenidos

1. [Descripcion del Problema](#1-descripcion-del-problema)
2. [Analisis PEAS](#2-analisis-peas)
3. [Tipo de Entorno](#3-tipo-de-entorno)
4. [Tipo de Agente](#4-tipo-de-agente)
5. [Dataset](#5-dataset)
6. [Algoritmos Propuestos](#6-algoritmos-propuestos)
7. [Arquitectura Propuesta](#7-arquitectura-propuesta)
8. [Requisitos del Sistema](#8-requisitos-del-sistema)
9. [Plan de Implementacion](#9-plan-de-implementacion)
10. [Metricas de Exito](#10-metricas-de-exito)

---

## 1. Descripcion del Problema

El spam en correos electronicos es un problema de clasificacion binaria: dado un correo, el sistema debe determinar si es **spam** o **ham** (correo legitimo).

El objetivo del proyecto es construir un agente inteligente capaz de tomar esa decision de forma automatica, analizando el asunto y el cuerpo del correo mediante tecnicas de Procesamiento de Lenguaje Natural (NLP) y algoritmos de Machine Learning supervisado.

El usuario podra elegir el algoritmo de clasificacion que desea usar para comparar resultados entre diferentes enfoques.

### Preguntas que guian el proyecto

- Puede un modelo probabilistico simple como Naive Bayes detectar spam en correos con alta precision?
- Supera la Regresion Logistica a Naive Bayes en este problema?
- Que palabras y patrones del asunto o cuerpo del correo son los mas discriminativos entre spam y ham?

---

## 2. Analisis PEAS

### Pregunta 1 — Cual es el agente?

Un **clasificador automatico de correos electronicos** que recibe el asunto y el cuerpo de un correo y decide si es spam o ham, junto con un nivel de confianza en su decision.

---

### Pregunta 2 — Cual es el entorno?

El entorno es el **conjunto de correos electronicos** que el agente recibe para evaluar. Sus caracteristicas son:

- Correos en ingles con asunto y cuerpo de longitud variable
- Spam con patrones tipicos: ofertas, premios, urgencia, links sospechosos, remitentes desconocidos
- Ham con comunicacion profesional o personal: reportes, conversaciones, notificaciones legitimas
- Presencia de HTML, URLs, firmas y formatos variados dentro del texto

El agente no controla que correos recibe ni en que orden. Reacciona a cada correo de forma independiente.

---

### Pregunta 3 — PEAS

| Componente | Definicion |
|------------|------------|
| **P** — Performance (Desempeno) | Porcentaje de correos correctamente clasificados en el conjunto de prueba. Metricas: Accuracy, Precision, Recall y F1-Score. La metrica prioritaria es **Recall** sobre la clase spam. |
| **E** — Environment (Entorno) | Correos electronicos reales con asunto y cuerpo en texto plano. Pueden contener HTML, URLs, numeros, firmas y lenguaje formal o informal. El dataset de referencia es el Enron-Spam Dataset. |
| **A** — Actuators (Actuadores) | La salida del clasificador: etiqueta (`spam` / `ham`), nivel de confianza entre 0 y 1, y desglose de probabilidades por clase. |
| **S** — Sensors (Sensores) | El texto del asunto y cuerpo del correo. El agente elimina HTML, convierte a minusculas, elimina puntuacion y stopwords, tokeniza y vectoriza antes de clasificar. |

---

### Pregunta 4 — Cual es la funcion de agente?

La funcion del agente mapea cada percepcion a una accion:

```
f: (asunto + cuerpo del correo)  -->  { etiqueta, confianza, probabilidades }
```

**Flujo interno:**

```
Correo (asunto + cuerpo)
    |
    v
Preprocesamiento
  - Eliminar HTML y etiquetas
  - Minusculas
  - Eliminar puntuacion
  - Eliminar stopwords
  - Tokenizacion
    |
    v
Vectorizacion
  (Bag of Words o TF-IDF segun el algoritmo)
    |
    v
Modelo de clasificacion
  (Naive Bayes o Regresion Logistica)
    |
    v
{ spam / ham, confianza }
```

**Formula — Naive Bayes:**
```
P(spam | palabras) = P(palabras | spam) * P(spam) / P(palabras)
```

**Formula — Regresion Logistica:**
```
P(spam | x) = 1 / (1 + e^(-(w * x + b)))
```

---

### Pregunta 5 — Cual es la medida de desempeno?

| Metrica | Definicion | Objetivo |
|---------|-----------|----------|
| **Accuracy** | Mensajes correctamente clasificados / Total | > 97% |
| **Precision** | De los marcados spam, cuantos eran spam real | > 95% |
| **Recall** | De los spam reales, cuantos fueron detectados | > 90% |
| **F1-Score** | Media armonica entre Precision y Recall | > 92% |

> **Por que Recall es la metrica prioritaria?**  
> Un falso negativo (spam que pasa como ham) es mas danino que un falso positivo (ham filtrado como spam). Es preferible ser conservador y bloquear algun mensaje legitimo antes de dejar pasar spam.

---

## 3. Tipo de Entorno

Clasificacion segun las cinco dimensiones estudiadas:

| Dimension | Clasificacion | Justificacion |
|-----------|--------------|---------------|
| **Observabilidad** | Totalmente observable | El agente recibe el texto completo del mensaje sin informacion oculta. |
| **Agentes** | Un solo agente | Solo el clasificador actua sobre el entorno, sin competencia. |
| **Determinismo** | Deterministico | El mismo mensaje siempre produce la misma clasificacion tras el entrenamiento. |
| **Secuencialidad** | Episodico | Cada mensaje se clasifica de forma independiente; una decision no afecta la siguiente. |
| **Dinamismo** | Estatico | El modelo no cambia mientras clasifica un mensaje individual. |

> **Conclusion:** El entorno es totalmente observable, un solo agente, deterministico, episodico y estatico. Este es uno de los entornos mas simples, lo que hace viable el uso de modelos clasicos de ML sin necesidad de aprendizaje por refuerzo ni razonamiento secuencial.

---

## 4. Tipo de Agente

El agente es un **Agente Basado en Modelo** (Model-Based Agent):

- No usa reglas manuales fijas como "si contiene FREE entonces spam".
- Construye un modelo interno del mundo durante el entrenamiento: distribuciones de probabilidad (Naive Bayes) o pesos aprendidos (Regresion Logistica).
- Aplica ese modelo para clasificar nuevos mensajes nunca vistos.

| Algoritmo | Tipo de agente |
|-----------|---------------|
| Naive Bayes | Generativo — modela como se generan los datos por clase |
| Regresion Logistica | Discriminativo — aprende directamente la frontera entre clases |

Ninguno de los dos es un agente que aprende en linea: ambos son entrenados una vez sobre el dataset y luego aplicados en modo inferencia.

---

## 5. Dataset

### Enron-Spam Dataset

| Atributo | Valor |
|----------|-------|
| Fuente | Enron Email Dataset (dominio publico) — recopilado por Metsis et al., 2006 |
| Tamano | ~33,700 correos electronicos |
| Clases | `spam` (~17,170) / `ham` (~16,545) |
| Formato | Archivos de texto plano organizados en carpetas por clase |
| Idioma | Ingles |
| Desbalance de clases | ~51% spam / ~49% ham (balanceado) |

**Por que este dataset?**

- Es el benchmark mas utilizado en papers academicos para clasificacion de spam en correos electronicos.
- Contiene correos reales de empleados de la empresa Enron, filtrados para uso publico tras el escandalo corporativo de 2001.
- La mitad son correos spam reales recopilados de fuentes publicas (SpamAssassin, Honeypot), y la otra mitad son correos ham reales de los empleados.
- El dataset esta balanceado, lo que simplifica el entrenamiento inicial y permite evaluar los modelos sin tecnicas de oversampling.
- Su tamano (~33k correos) es suficiente para entrenar modelos clasicos de ML con buenos resultados sin necesidad de GPU.

**Estructura del dataset:**

```
enron_spam/
├── spam/        <- ~17,170 correos spam en archivos .txt
└── ham/         <- ~16,545 correos ham en archivos .txt
```

Cada archivo de texto contiene el cuerpo del correo en texto plano. El asunto puede estar incluido como primera linea en algunos archivos.

**Division para entrenamiento y prueba:**

```
~33,700 correos totales
├── Entrenamiento: ~26,960 (80%)  -- para ajustar el modelo
└── Prueba:         ~6,740 (20%)  -- para evaluar rendimiento final
```

La division se hara con estratificacion (`stratify=labels`) para preservar la proporcion 51/49 en ambos conjuntos.

**Ejemplo de contenido:**

| label | fragmento del correo |
|-------|----------------------|
| spam | "Congratulations! You have been selected to receive a FREE gift card..." |
| ham | "Hi all, the meeting has been rescheduled to Thursday at 3pm." |
| spam | "URGENT: Your account will be suspended. Click here to verify your details." |
| ham | "Please find attached the Q3 report for your review." |

---

## 6. Algoritmos 

### 6.1 Naive Bayes

- **Vectorizacion**: Bag of Words (frecuencia de palabras)
- **Supuesto**: Las palabras son condicionalmente independientes dada la clase
- **Suavizado**: Laplace smoothing (alpha = 1.0) para evitar probabilidades cero
- **Ventajas**: Muy rapido, funciona bien con pocos datos, altamente interpretable
- **Desventajas**: El supuesto de independencia no siempre se cumple en lenguaje natural

Hiperparametros a definir:

| Parametro | Valor propuesto | Descripcion |
|-----------|----------------|-------------|
| `alpha` | 1.0 | Suavizado de Laplace |

### 6.2 Regresion Logistica

- **Vectorizacion**: TF-IDF (penaliza palabras muy comunes, premia las discriminativas)
- **Supuesto**: La frontera de decision entre clases es lineal
- **Optimizador**: L-BFGS
- **Ventajas**: Alta precision en texto con TF-IDF, mejor calibracion de probabilidades
- **Desventajas**: Necesita mas datos para converger bien, menos interpretable palabra a palabra

Hiperparametros a definir:

| Parametro | Valor propuesto | Descripcion |
|-----------|----------------|-------------|
| `C` | 1.0 | Inverso de regularizacion (mayor = menos regularizacion) |
| `max_iter` | 1000 | Iteraciones maximas del optimizador |
| `solver` | `lbfgs` | Algoritmo de optimizacion |

### 6.3 Comparacion entre algoritmos

| Aspecto | Naive Bayes | Regresion Logistica |
|---------|-------------|---------------------|
| Vectorizacion | Bag of Words | TF-IDF |
| Velocidad de entrenamiento | Muy rapida | Iterativa, mas lenta |
| Velocidad de inferencia | Muy rapida | Rapida |
| Datos necesarios para buen resultado | Pocos | Mas datos |
| Interpretabilidad | Alta | Media |
| Rendimiento esperado (accuracy) | ~97% | ~98-99% |

---

## 7. Arquitectura Propuesta

```
[ Usuario: ruta del correo o texto ]
           |
           v
[ src/main.py — CLI ]
  - Recibe --algorithm y --email
           |
           v
[ src/preprocessor.py ]
  - Eliminar HTML y etiquetas
  - Minusculas
  - Eliminar puntuacion
  - Eliminar stopwords
  - Tokenizacion
           |
     +-----+------+
     |             |
     v             v
[ BoW ]        [ TF-IDF ]
     |             |
     v             v
[ naive_bayes ] [ logistic_regression ]
     |             |
     +-----+------+
           |
           v
[ src/evaluator.py ]
  - Accuracy, Precision, Recall, F1
  - Matriz de confusion
           |
           v
[ Salida: etiqueta + confianza ]
```

---

## 8. Requisitos del Sistema

### Funcionales

| ID | Requisito |
|----|-----------|
| RF-01 | El sistema debe clasificar un correo electronico como spam o ham. |
| RF-02 | El usuario debe poder elegir entre Naive Bayes y Regresion Logistica. |
| RF-03 | El sistema debe mostrar el nivel de confianza de cada prediccion. |
| RF-04 | El sistema debe poder evaluar ambos modelos sobre el test set y mostrar metricas. |
| RF-05 | El sistema debe poder leer correos desde archivo de texto o desde input directo por CLI. |
| RF-06 | El preprocesador debe eliminar HTML y etiquetas antes de analizar el texto del correo. |
| RF-07 | Los modelos entrenados deben guardarse en disco para no reentrenar en cada ejecucion. |

### No Funcionales

| ID | Requisito |
|----|-----------|
| RNF-01 | El tiempo de prediccion de un mensaje no debe superar 1 segundo. |
| RNF-02 | El codigo debe estar modularizado: un archivo por responsabilidad. |
| RNF-03 | Cada modulo debe tener tests basicos de funcionamiento. |
| RNF-04 | El F1-Score sobre el test set debe superar 92% en ambos modelos. |

---

## 9. Plan de Implementacion

### Fase 1 — Datos

- [ ] Crear directorio `data/`
- [ ] Implementar `src/download_data.py` para descargar y descomprimir el Enron-Spam Dataset
- [ ] Leer los archivos de las carpetas `spam/` y `ham/` y construir un DataFrame etiquetado
- [ ] Explorar el dataset: distribucion de clases, longitud promedio de correos, palabras mas frecuentes por clase

### Fase 2 — Preprocesamiento

- [ ] Implementar `src/preprocessor.py` con: eliminacion de HTML, limpieza, tokenizacion y eliminacion de stopwords
- [ ] Implementar `src/vectorizer.py` con soporte para Bag of Words y TF-IDF

### Fase 3 — Modelos

- [ ] Implementar `src/naive_bayes.py` como clase con metodos `train`, `predict`, `predict_batch`
- [ ] Implementar `src/logistic_regression.py` con la misma interfaz
- [ ] Ambos modelos deben ser intercambiables desde el CLI sin cambiar el codigo

### Fase 4 — Evaluacion

- [ ] Implementar `src/evaluator.py` con accuracy, precision, recall, F1 y matriz de confusion
- [ ] Comparar ambos modelos sobre el mismo test set y documentar resultados

### Fase 5 — CLI

- [ ] Implementar `src/main.py` con argumentos `--algorithm`, `--email`, `--evaluate`, `--train`
- [ ] Soportar input desde texto directo o desde ruta a un archivo `.txt`
- [ ] Guardar y cargar modelos con `pickle`

### Fase 6 — Documentacion

- [ ] Actualizar README con instrucciones de uso
- [ ] Completar la tabla de resultados en este documento con los valores reales

---

## 10. Metricas de Exito

Los resultados seran documentados aqui una vez que el proyecto este implementado.

### Resultados esperados

| Algoritmo | Accuracy | Precision | Recall | F1-Score |
|-----------|----------|-----------|--------|----------|
| Naive Bayes | ~97% | ~95% | ~90% | ~92% |
| Regresion Logistica | ~98-99% | ~97% | ~93% | ~95% |

### Resultados reales (por completar)

| Algoritmo | Accuracy | Precision | Recall | F1-Score |
|-----------|----------|-----------|--------|----------|
| Naive Bayes | — | — | — | — |
| Regresion Logistica | — | — | — | — |

---

## Referencias

- Russell, S. & Norvig, P. — *Artificial Intelligence: A Modern Approach* (AIMA)
- Metsis, V., Androutsopoulos, I., Paliouras, G. — *Spam Filtering with Naive Bayes — Which Naive Bayes?* (2006) — paper original del Enron-Spam Dataset
- scikit-learn Documentation — Naive Bayes, Logistic Regression, TF-IDF Vectorizer
