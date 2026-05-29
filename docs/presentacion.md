---
marp: true
theme: default
paginate: true
size: 16:9
header: 'Detector de Spam Bilingüe — Inteligencia Artificial'
---

<!--
ESQUELETO DE PRESENTACIÓN PARA LA DEFENSA ORAL (~20 min, 14 slides).
Formato Marp. Para exportar a PDF/PPTX:
    npx @marp-team/marp-cli docs/presentacion.md --pdf
    npx @marp-team/marp-cli docs/presentacion.md --pptx
Las notas del orador van en comentarios <!-- ... --> bajo cada slide.
-->

# Detector de Spam Bilingüe
## Inglés + Español con Machine Learning

**Universidad Nacional** · Curso de Inteligencia Artificial
Prof. Venegas · Junio 2026

Integrante(s): _______________________

<!--
NOTA: Preséntate (30s). Frase gancho: "El 45% del correo mundial es spam, y es
el principal vector de phishing. Hoy les muestro un agente que lo detecta solo,
en dos idiomas, con 96% de acierto."
-->

---

# 1. Problema y motivación

- El **spam** es la puerta de entrada a phishing, fraude y malware.
- Filtros por reglas (*"si contiene FREE → spam"*) se evaden fácilmente.
- Objetivo: un **agente que aprende** a distinguir *spam* de *ham* a partir de datos.
- **Bilingüe** (EN + ES): el phishing localizado (BBVA, SAT, CFE) no lo detecta un modelo solo-inglés.

<!--
NOTA (2 min): Enfatiza el COSTO ASIMÉTRICO: dejar pasar spam (phishing) es peor
que bloquear un mensaje legítimo. Esto justifica priorizar Recall más adelante.
-->

---

# 2. Formulación: clasificación binaria supervisada

Función del agente: **f : texto → {spam, ham} + P(spam)**

| | Variable | Dominio |
|---|---|---|
| Entrada | `text` | Cadena Unicode (EN/ES) |
| Salida | `label` | {spam, ham} |
| Salida aux. | `P(spam)` | [0, 1] |

**PEAS** → Performance: Recall ∙ Environment: mensajes EN/ES ∙ Actuators: etiqueta + confianza ∙ Sensors: texto.
Entorno: observable, un agente, determinístico, episódico, estático. Agente *model-based*.

<!--
NOTA (2 min): Aquí demuestras dominio teórico del curso (PEAS y tipo de entorno).
Si te preguntan "¿por qué episódico?": cada mensaje se clasifica independiente,
una decisión no afecta la siguiente.
-->

---

# 3. Dataset

- **SMS Spam Collection (UCI)** — 5 574 mensajes en inglés (Almeida & Hidalgo, 2011).
- **SMS Multilingual** — 5 572 traducidos al español + **61 seed** de phishing local real.
- Combinado y deduplicado: **~10 331 mensajes**, **12.7 % spam** (desbalanceado).

![w:560](img/exploracion.png)

<!--
NOTA (1.5 min): Menciona que el dataset es DESBALANCEADO (12.7% spam), reflejo de
la realidad. Esto motiva el balanceo y el por qué accuracy sola no basta.
-->

---

# 4. Preprocesamiento

Función `clean_text`:

1. Minúsculas
2. URLs → `__url__`, números → `__num__` *(preserva la señal, no el valor)*
3. Quita HTML y puntuación
4. Stopwords EN + ES (**504** palabras)

> Reemplazar por *tokens* en vez de borrar: la **presencia** de URL/número es predictiva de spam.

<!--
NOTA (1.5 min): El truco de los tokens __url__/__num__ es una decisión de diseño
defendible: generaliza entre idiomas y captura el patrón del spam.
-->

---

# 5. Balanceo y partición

- **Undersampling** de la clase mayoritaria → **3 287** mensajes, ~40 % spam.
- Split **estratificado**: train **2 629** / test **658** (`random_state=42`).
- El test conserva ambos idiomas: 337 EN / 321 ES.

<!--
NOTA (1 min): random_state=42 = reproducibilidad. Estratificado = misma proporción
de spam en train y test. Evita data leakage al vectorizar dentro del pipeline.
-->

---

# 6. Los tres modelos

| # | Modelo | Representación | Tipo |
|---|---|---|---|
| 1 | Naive Bayes | Bag of Words | Generativo |
| 2 | Regresión Logística | TF-IDF palabras | Discriminativo |
| 3 | Regresión Logística | TF-IDF char n-grams | Robusto a typos |

- **Naive Bayes:** P(spam\|palabras) ∝ P(palabras\|spam)·P(spam)
- **Logística:** P(spam\|x) = σ(w·x + b), σ = sigmoide

<!--
NOTA (2.5 min): Explica la INTUICIÓN, no las fórmulas. NB cuenta frecuencias por
clase ("¿qué tan típica es esta palabra en spam?"). Logística traza una frontera
ponderando palabras. Char n-grams mira pedazos de letras → resiste "g4n4 din3ro".
-->

---

# 7. Comparación de modelos (test)

| Modelo | Acc | Prec | Recall | F1 |
|---|---|---|---|---|
| Naive Bayes | 0.957 | 0.943 | 0.951 | 0.947 |
| **LogReg TF-IDF** | 0.961 | 0.941 | **0.962** | 0.951 |
| LogReg char n-grams | 0.957 | 0.943 | 0.951 | 0.947 |

![w:520](img/comparativa.png)

→ Se elige **Regresión Logística (TF-IDF)** por mejor Recall.

<!--
NOTA (1.5 min): Los tres pasan de 95% F1. La diferencia es pequeña pero la
Logística gana en Recall, que es nuestra métrica prioritaria.
-->

---

# 8. Anti-overfitting + tuning

- **Cross-validation (5-fold):** brechas train–CV de solo ≈0.02–0.03 → **sin sobreajuste**.
- **GridSearchCV** (18×3 fits): mejor `C=10`, `min_df=5`, `ngram=(1,1)` → **F1 CV 0.959**.
- Modelo final: **Accuracy 0.962 · F1 0.952**.

<!--
NOTA (1.5 min): La curva de aprendizaje (F1 train 0.989 vs val 0.959, brecha 0.030)
confirma que no hay overfitting y que más datos ayudarían un poco más.
-->

---

# 9. Calibración del threshold

![w:520](img/curva_pr.png)

| Threshold | Precision | Recall |
|---|---|---|
| 0.50 (defecto) | 0.954 | 0.951 |
| **0.189 (óptimo)** | 0.852 | **0.981** |

→ Subimos Recall a **98.1 %**: solo se escapa el 1.9 % del spam.

<!--
NOTA (2 min): ESTE es el corazón de la defensa. Mover el umbral materializa la
decisión del PEAS (priorizar Recall). Se sacrifica precisión a cambio de detectar
casi todo el spam. En producción: los positivos van a cuarentena, no se borran.
-->

---

# 10. Evaluación cross-lingual + matriz de confusión

| Idioma | Acc | F1 |
|---|---|---|
| Inglés | 0.967 | 0.959 |
| Español | 0.956 | 0.946 |

![w:380](img/matriz_confusion.png)

25 errores / 658 (**3.8 %**): 12 falsos positivos, 13 falsos negativos.

<!--
NOTA (1.5 min): El rendimiento casi idéntico en ambos idiomas prueba que el
enfoque generaliza entre lenguas. Los errores son mensajes cortos/ambiguos.
-->

---

# 11. Demo en vivo 🔴

Sistema interactivo (Gradio / ipywidgets):

- "WIN a free iPhone now! http://bit.ly/win" → **SPAM (99%)**
- "Hola, paso por ti a las 7" → **HAM (2%)**
- *(escribe un mensaje del público en vivo)*

<!--
NOTA (2 min): ¡Ensaya la demo antes! Ten el notebook ya ejecutado y la celda de
Gradio lista. Pide un mensaje al evaluador para clasificarlo en vivo. Plan B: si
falla la red/widget, usa la celda de "vista previa estática" o el fallback input().
-->

---

# 12. Conclusiones

- Detector bilingüe **end-to-end**: **F1 0.952 · Accuracy 0.962**.
- Recall sobre spam elevado a **98.1 %** vía calibración del threshold.
- **Generaliza** entre idiomas (F1 EN 0.959 / ES 0.946), sin overfitting.
- Entregables: notebook reproducible + modelo serializado + sistema interactivo.

**Futuro:** spam español nativo · transformers multilingües (mBERT) · reentrenamiento incremental.

<!--
NOTA (1 min): Cierra conectando con la motivación inicial: "logramos un agente que
detecta el 98% del spam en dos idiomas, defendible como un filtro real de seguridad".
-->

---

# Gracias — ¿Preguntas?

**Apéndice: preguntas técnicas frecuentes y respuestas →**

<!--
Pasa al apéndice solo si preguntan. Mantén la calma y responde con la intuición.
-->

---

# Apéndice — Q&A de defensa

**¿Naive Bayes vs Regresión Logística?**
NB es *generativo* (modela cómo se generan los datos por clase, asume independencia
entre palabras — "naive"). LogReg es *discriminativo* (aprende directo la frontera
spam/ham ponderando palabras). LogReg suele ganar cuando hay suficientes datos.

**¿Por qué char n-grams?**
Capturan trozos de caracteres (ej. "gan", "an4"), resistiendo errores ortográficos
y ofuscación ("V1AGR4"). Son agnósticos al idioma. Aquí GridSearch prefirió palabras
porque el corpus ya está limpio; aportarían más con ofuscación agresiva.

---

# Apéndice — Q&A de defensa (2)

**¿Por qué priorizar Recall y no Accuracy?**
Accuracy engaña en datasets desbalanceados (predecir siempre "ham" da ~87%). El
costo de un falso negativo (phishing que llega) supera al de un falso positivo
(ham en cuarentena). Por eso bajamos el threshold para maximizar Recall.

**¿Cómo evitan el overfitting / data leakage?**
Cross-validation k-fold + curva de aprendizaje (brechas pequeñas). El vectorizador
va DENTRO del `Pipeline`, así se ajusta solo con train en cada fold — sin fuga del test.

**¿Qué es TF-IDF?**
Pondera cada palabra por su frecuencia en el mensaje (TF) penalizada por lo común
que es en todo el corpus (IDF). Resalta palabras discriminativas, no las frecuentes.
