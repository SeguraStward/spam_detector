# Guion de práctica — Detector de Spam Bilingüe

> Una línea base **breve** por slide; en las slides clave (⭐) se desarrolla un poco más.
> Coincide con el orden del deck animado. Habla natural, no leas palabra por palabra.
> Tiempo objetivo: **~15–18 min**.

---

**Portada.**
"Buenos noches. Mi proyecto es clasificacion de Spam y les voy a contar como lo hice, y que resultados obtuve."

---

**1. Objetivo.**
"El objetivo es un modelo que **aprende** a reconocer spam en mensajes que nunca ha visto, en dos idiomas y utilizando naive bayes y logistic regression."
 

---

**2. Formulación.** ⭐
"Formalmente es **clasificación binaria supervisada**: clasificación porque predigo una categoría, no un número; binaria porque solo hay dos clases, spam o ham; y supervisada porque entrené con datos ya etiquetados —como estudiar con un examen resuelto."

"La entrada es el texto crudo del mensaje y la salida es la etiqueta más una probabilidad de spam. Un detalle importante: la decisión final no usa el 0.5 de siempre, sino un umbral que calibré, y eso lo explico al final."

---

**3. PEAS.**
"Apliqué el marco PEAS. Desempeño: priorizo el **Recall** sobre spam, por el costo asimétrico. Entorno: mensajes en inglés o español con URLs, números y errores de ortografía. Actuadores: la etiqueta más un nivel de confianza. Sensores: el texto del mensaje."

---

**4. Tipo de entorno y agente.**
"El entorno es observable, un solo agente, determinístico, episódico y estático — de los más simples, por eso son viables modelos clásicos. Y es un agente basado en modelo: aprende una representación interna y luego la aplica."

---

**5. Los datos.** ⭐
"Aquí tuve el primer reto: datos de spam en español casi no hay. El benchmark estándar, el SMS Spam Collection de UCI, es en inglés. Así que combiné tres fuentes en español: una versión traducida del UCI, un dataset de spam **nativo**, y un puñado de phishing local que curé a mano."

"En total unos 11 400 mensajes, con 16 % de spam. Ese desbalance no lo inventé yo: refleja la realidad, la mayoría de los mensajes son legítimos."

---

**6. Exploración (EDA).**
"Antes de modelar exploré el corpus. El desbalance hacia ham es claro en ambos idiomas, y el spam tiende a ser más corto. Eso ya me anticipaba dos cosas: que tendría que balancear y que la accuracy sola me iba a engañar. El corpus armado lo guardo en caché para que el notebook sea reproducible sin internet."

---

**7. Preprocesamiento.**
"Limpio el texto y —lo más interesante— reemplazo las URLs por el token `__url__` y los números por `__num__`. ¿Por qué reemplazar en vez de borrar? Porque la **presencia** de un link o un número es señal de spam, aunque el valor exacto no importe."

---

**8. Vectorización.**
"El modelo solo entiende números, así que convierto el texto en vectores. Probé tres representaciones: Bag of Words (conteo), TF-IDF (palabras ponderadas) y n-gramas de caracteres (trozos de letras)."

---

**9. Desbalance y balanceo.**
"Con 84 % de ham, un modelo perezoso aprende a decir siempre 'ham' y acierta el 84 % sin detectar nada. Para evitarlo hago **undersampling** —me quedo con todo el spam y reduzco el ham, ratio 1.5 a 1— y además uso `class_weight='balanced'` para que los errores en spam pesen más. Doble defensa."

---

**10. Partición y Pipeline.**
"Divido 80/20 estratificado, con semilla fija para reproducibilidad. Y encadeno el vectorizador y el modelo en un **Pipeline**: así, en la validación cruzada, el vectorizador se ajusta solo con el train de cada fold y nunca ve el test. Eso evita el *data leakage*."

---

**11. Modelos utilizados.** ⭐
"Usé tres. **Naive Bayes**, que es generativo: aprende cómo se ve cada clase y, con el teorema de Bayes, elige la más probable; asume que las palabras son independientes —de ahí lo de 'naive'."

"**Regresión Logística** con TF-IDF de palabras, que es discriminativa: aprende un peso por palabra, los suma y una sigmoide los convierte en probabilidad. Y la misma logística pero con **n-gramas de caracteres**, que en vez de palabras mira trozos de letras, por lo que es robusta a errores de escritura y transfiere mejor entre idiomas."

> ❓ *¿Generativo vs discriminativo?* → "El generativo modela cómo se generan los datos de cada clase y aplica Bayes; el discriminativo aprende directo la frontera entre spam y ham."

---

**12. Comparación de los tres modelos.** ⭐
"Aquí los comparo en el test. Los tres superan el 90 % de F1. Sin afinar, el mejor es el de **char n-grams** (F1 0.923 y el mejor Recall): los trozos de caracteres aguantan mejor el español variado."

"No elegí 'a ojo': tuneé los dos mejores con GridSearch y dejé que decidiera el rendimiento. Eso lo vemos en la siguiente slide."

---

**13. Validación cruzada.** ⭐
"Para no fiarme de un solo split usé validación cruzada de 5 folds: parto el train en 5 trozos y entreno 5 veces, rotando cuál es el examen. Importante: cada vez entreno con el 80 % y valido con el 20 %, los datos se reutilizan."

"Lo que miro es la **brecha** entre train y validación: si train es mucho mayor, memorizó. Aquí las brechas son pequeñas (~0.04–0.05), así que hay solo un sobreajuste leve y los modelos generalizan; el de caracteres es el más estable."

---

**14. GridSearch (tuning).** ⭐
"GridSearch prueba combinaciones de hiperparámetros y elige la mejor por validación cruzada. **Tuneé los dos candidatos** —palabras y char— y me quedé con el de mejor F1. **Ganó char n-grams** (F1 CV 0.946 vs 0.939 del de palabras). El modelo final llega a **F1 0.950, Accuracy 0.959** en el test."

> ❓ *¿Por qué char y no palabras?* → "Tuneé ambos; char dio mejor F1 (0.950 vs 0.944) y mejor recall. No lo elegí por intuición, lo decidió el rendimiento."

---

**15. Curva de aprendizaje.**
"Esta es la curva del modelo final, char n-grams. El train va casi en 1.0 y la validación sube hasta 0.946; esa pequeña brecha es el sobreajuste leve. Lo bueno: la validación sigue subiendo, así que con más datos —sobre todo spam nativo— el modelo mejoraría."

---

**16. Métricas y matriz de confusión.** ⭐
"Aquí muestro qué tan efectivo es el modelo final con las cuatro métricas reales. La **accuracy** (0.959) es el % de aciertos, pero engaña con desbalance. La **precision** (0.942) mide, de lo que marqué spam, cuánto era spam de verdad. El **recall** (0.957) mide, de todo el spam real, cuánto atrapé —es mi prioridad. Y el **F1** (0.950) resume el equilibrio."

"La matriz de confusión lo hace concreto: de 932 mensajes, solo 38 errores —16 spam que se colaron y 22 ham bloqueados. Reporto las cuatro métricas porque cada una tapa el punto ciego de la otra."

> ❓ *¿Por qué no solo accuracy?* → "Con 16 % de spam, un 'todo ham' da 84 % de accuracy sin detectar nada. Por eso miro precision, recall y F1."

---

**17. Calibración del threshold.** ⭐ (la pieza fuerte)
"Esta es la parte que más me gusta. La decisión por defecto usa 0.5, pero ese número es arbitrario. Como mi prioridad es el Recall, **calibré el umbral**: busqué el que maximiza Recall manteniendo una precisión razonable, y salió **0.253**."

"El efecto es directo: con 0.5 el Recall era 0.957; bajándolo a 0.253 sube a **0.981** —solo se escapa el 1.9 % del spam— a cambio de algo de precisión. Las dos matrices muestran el trade-off: los falsos negativos caen de 16 a 7, y suben los falsos positivos. Es una decisión consciente por el contexto de seguridad; en un sistema real los positivos van a una carpeta de cuarentena, no se borran."

> ❓ *¿No es hacer trampa bajar el umbral?* → "No, lo elijo con una regla explícita —máximo Recall con precisión ≥ 0.85— y reporto las dos métricas. No oculto la caída de precisión, la decido a propósito."

---

**18. Evaluación cross-lingual.**
"Evalué por idioma por separado: inglés F1 0.965, español 0.941. El modelo transfiere bien entre idiomas, con apenas un par de puntos de diferencia. El español rinde un poco menos, pero se mide sobre un test más grande y con spam nativo, así que es un número honesto."

---

**19. Limitaciones.**
"Para cerrar, soy honesto con los límites: el español sigue sub-representado y en parte es traducción automática; el corpus es de mensajes cortos, no correos largos con HTML; el spam evoluciona, así que habría que reentrenar cada cierto tiempo; y el GridSearch fue acotado por costo. Reconocer los límites es parte del rigor."

---

**20. Resultados finales.**
"En resumen: un detector bilingüe con **F1 0.950, Accuracy 0.959 y Recall del 98.1 %** tras calibrar, que generaliza a ambos idiomas y sin sobreajuste grave. Más allá de los números, me llevo el criterio: priorizar Recall por el contexto, dejar que el rendimiento elija el modelo, y ser honesto con lo que no funcionó."

---

**Cierre.**
"Y eso es todo. Gracias, quedo atento a sus preguntas."

---

## Demo opcional (si la haces)
Clasifica un mensaje del público con el modelo final; cambia de modelo y mueve el threshold para mostrar cómo un caso dudoso pasa de ham a spam. *Plan B si falla:* enseña la matriz y las métricas que ya están guardadas en el notebook.

## Tus números (de memoria)
F1 **0.950** · Accuracy **0.959** · Recall **0.957 → 0.981** (threshold **0.253**) · EN **0.965** / ES **0.941** · corpus **~11 400** (16.3 % spam) · 38 errores en test.

## Preguntas que pueden caer
- **¿Naive Bayes vs Logística?** Generativo (Bayes + independencia) vs discriminativo (frontera directa).
- **¿Qué es TF-IDF?** Frecuencia de la palabra penalizada por qué tan común es en el corpus.
- **¿Por qué char n-grams?** Robusto a typos y agnóstico al idioma; ganó el tuning.
- **¿Cómo evitas data leakage?** El vectorizador va dentro del Pipeline.
- **¿Qué es el threshold 0.253?** El umbral calibrado para maximizar Recall (no el 0.5 por defecto).
- **¿Tu modelo sobreajusta?** Leve (brecha ~0.05); la validación sigue subiendo → más datos ayudarían.
