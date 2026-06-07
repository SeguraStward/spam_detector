# Guion breve — Detector de Spam Bilingüe

> Una idea clave por slide. Conciso a propósito: di lo esencial y **expande a tu criterio**.
> Coincide con el orden del deck animado (`index.html`). Las notas también están en cada slide (tecla **S**).

---

**Portada.**
"Buenos días. Les presento un detector de spam bilingüe (inglés y español) con Machine Learning. Voy a contar qué hice, qué decidí y qué aprendí."

---

**1. Objetivo.**
"Un agente que aprende a distinguir spam de mensajes legítimos y generaliza a mensajes nuevos. Bilingüe, porque el phishing en español no lo detecta un modelo solo-inglés. Idea central: dejar pasar phishing es más caro que bloquear un mensaje bueno."

---

**2. Formulación.**
"Es clasificación binaria supervisada: predigo una categoría (spam/ham), de 2 clases, aprendiendo de datos etiquetados. Entrada: el texto; salida: etiqueta + probabilidad."

---

**3. PEAS.**
"Desempeño: priorizo Recall. Entorno: mensajes EN/ES con URLs y números. Actuadores: etiqueta + confianza. Sensores: el texto crudo."

---

**4. Tipo de entorno y agente.**
"Entorno observable, un agente, determinístico, episódico y estático — el más simple, viable con modelos clásicos. Agente basado en modelo: aprende y luego aplica."

---

**5. Los datos.**
"Corpus bilingüe de ~11.400 mensajes, 16% spam. Tres fuentes en español: traducido, nativo y un seed local. El desbalance refleja la realidad."

---

**6. Exploración (EDA).**
"El desbalance hacia ham es claro; el spam es más corto. Eso ya anticipa que tendré que balancear y que la accuracy sola engaña."

---

**7. Preprocesamiento.**
"Limpio el texto y reemplazo URLs y números por tokens. Decisión clave: la presencia de un link es señal de spam aunque el valor no importe."

---

**8. Vectorización.**
"Convierto texto en números. Probé tres representaciones: Bag of Words, TF-IDF y n-gramas de caracteres."

---

**9. Desbalance y balanceo.**
"Con 84% ham el modelo se vuelve perezoso. Hago undersampling 1.5:1 y uso class_weight para que los errores en spam pesen más."

---

**10. Partición y Pipeline.**
"Split estratificado 80/20. El vectorizador va dentro del Pipeline para que nunca vea el test: así evito el data leakage."

---

**11. Modelos utilizados.**
"Tres: Naive Bayes (generativo, cuenta palabras + Bayes), Regresión Logística (discriminativa, pesos + sigmoide) y la misma con char n-grams, robusta a typos."

---

**12. Comparación de los tres modelos.**
"Los tres superan el 90% de F1. Char n-grams es el mejor sin afinar; tuneo ambos y char también gana, así que es el modelo final."

---

**13. Validación cruzada.**
"5 folds para no fiarme de un solo split. Miro el gap train–validación: el modelo de palabras tiene un sobreajuste leve."

---

**14. GridSearch (tuning).**
"Tuneé AMBOS modelos (palabras y char) y elegí el de mejor F1: ganó char n-grams (C=10, ngram 3–5). El modelo final llega a F1 0.950, Accuracy 0.959."

---

**15. Curva de aprendizaje.**
"Curva del modelo final (char n-grams): train casi perfecto vs validación 0.946 → sobreajuste leve. La curva sigue subiendo, así que más datos ayudarían."

---

**16. Métricas y matriz de confusión.**
"Las cuatro métricas juntas: accuracy engaña con desbalance, precision mide falsas alarmas, recall el spam atrapado (prioritario), F1 el equilibrio. La matriz hace tangibles los errores: 16 spam colados, 22 ham bloqueados."

---

**17. Calibración del threshold.**
"El 0.5 es arbitrario. Lo bajé a 0.253 para maximizar Recall: sube de 0.957 a 0.981. Las dos matrices muestran el trade-off: menos spam colado (16→7) a costa de más falsas alarmas (22→64)."

---

**18. Evaluación cross-lingual.**
"Por idioma: inglés F1 0.965, español 0.941. El español rinde algo menos pero su test es más grande y con spam nativo — número más honesto."

---

**19. Dificultades y decisiones.**
"Spam español escaso → 3 fuentes. Probé un filtro de ruido y lo descarté porque borraba spam bueno. Naive Bayes sobre-confiado → elegí la logística. No toda técnica sofisticada ayuda."

---

**20. Resultados finales.**
"Detector bilingüe: F1 0.950, Accuracy 0.959, Recall 98.1% tras calibrar. Generaliza entre idiomas y sin sobreajuste grave."

---

**Cierre.**
"En resumen: prioricé Recall por el contexto de seguridad, equilibré rendimiento e interpretabilidad, y fui honesto con lo que no funcionó. Gracias, quedo atento a preguntas."

---

> **Demo opcional (si la haces):** clasifica un mensaje del público y mueve el threshold para mostrar cómo un caso dudoso pasa de ham a spam.
> **Recuerda tus números:** F1 0.950 · Acc 0.959 · Recall 0.981 (t=0.253) · EN 0.965 / ES 0.941.
