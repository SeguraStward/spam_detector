# Guion de presentación — Detector de Spam Bilingüe

> Cómo usar este guion: cada bloque corresponde a una diapositiva de `presentacion.md`.
> Lo escrito en *cursiva* es lo que dices; los **❓** son preguntas que el profe podría
> hacer, con la respuesta lista. Tiempo objetivo: **~18–20 min**. Habla natural, no leas
> palabra por palabra: usa esto como apoyo. Marca con la voz las decisiones ("decidí…",
> "me topé con…") porque ahí está el criterio que evalúan.

---

## Slide 1 — Portada (30 s)

*"Buenos días. Mi proyecto es un detector de spam bilingüe, inglés y español, hecho con Machine Learning. La idea es un agente que aprende a separar mensajes legítimos de spam, y que además sea honesto: voy a contarles no solo lo que funcionó, sino las decisiones y los problemas que tuve que resolver en el camino."*

> Tip: arranca con energía y mirando al público, no a la pantalla.

---

## Slide 2 — Agenda (20 s)

*"Voy a ir del problema a los datos, luego al preprocesamiento, los modelos, cómo los validé y afiné, las métricas, y termino con las dificultades reales y una demo en vivo."*

---

## Slide 3 — Problema y motivación (1.5 min)

*"El spam no es solo molesto: es la puerta de entrada al phishing, al fraude bancario y al malware. Un filtro por reglas fijas, tipo 'si dice FREE entonces spam', se evade en cinco minutos. Por eso quería un modelo que **aprendiera** los patrones a partir de datos y generalizara a mensajes nuevos."*

*"Lo hice bilingüe a propósito: nosotros recibimos phishing en español —el típico 'su cuenta BBVA fue bloqueada' o 'el SAT le debe una devolución'— y un modelo entrenado solo en inglés no lo detecta."*

*"Y una idea que atraviesa todo el proyecto: el **costo asimétrico**. Dejar pasar un phishing es mucho más caro que mandar por error un mensaje bueno a spam. Esa idea va a justificar varias decisiones más adelante."*

> ❓ **¿Por qué no usar un filtro de reglas o palabras clave?**
> *"Porque es frágil: el spammer cambia 'free' por 'fr33' y ya lo evadió. Un modelo estadístico aprende combinaciones de señales, no una lista fija, y se adapta mejor."*

---

## Slide 4 — Formulación (1.5 min)

*"Formalmente esto es **clasificación binaria supervisada**. Tres palabras con significado: clasificación porque predigo una categoría, no un número; binaria porque solo hay dos clases, spam o ham; y supervisada porque entrené con datos ya etiquetados, como estudiar con un examen resuelto."*

*"La entrada es el texto crudo del mensaje y la salida es la etiqueta más una probabilidad de spam. Y un detalle importante: la decisión final no usa el 0.5 de siempre, sino un umbral que calibré, y eso lo explico al final."*

> ❓ **¿Qué diferencia hay con un problema de regresión?**
> *"La regresión predice un número continuo, como un precio. Yo predigo una categoría, por eso es clasificación."*

> ❓ **¿Y si fueran más de dos clases?**
> *"Sería multiclase. De hecho encontré un dataset con una tercera clase, 'smishing', pero decidí mantenerlo binario para no complicar el alcance; el smishing es un tipo de spam."*

---

## Slide 5 — PEAS (1.5 min)

*"Apliqué el marco PEAS que vimos en clase. La **medida de desempeño** son accuracy, precision, recall y F1, pero priorizo el recall sobre spam, otra vez por el costo asimétrico. El **entorno** son mensajes en inglés o español con URLs, números y errores de ortografía. Los **actuadores** son la etiqueta más un nivel de confianza, no solo un sí/no. Y los **sensores** son únicamente el texto del mensaje."*

> ❓ **¿Por qué dar una probabilidad y no solo la etiqueta?**
> *"Porque en un sistema real no es lo mismo un 51 % de spam que un 99 %. Con la probabilidad puedes mandar los dudosos a revisión y bloquear solo los muy seguros."*

---

## Slide 6 — Tipo de entorno y agente (1 min)

*"Clasifiqué el entorno: es totalmente observable porque veo el mensaje completo; un solo agente; determinístico, el mismo texto da siempre la misma predicción; episódico, cada mensaje es independiente del anterior; y estático. Es de los entornos más simples, por eso son viables modelos clásicos sin necesidad de aprendizaje por refuerzo."*

*"Y es un agente **basado en modelo**: durante el entrenamiento construye una representación interna —pesos o probabilidades— y luego la aplica, no memoriza reglas a mano."*

> ❓ **¿Por qué episódico y no secuencial?**
> *"Porque clasificar un mensaje no afecta al siguiente; no hay memoria entre decisiones, cada uno se resuelve solo."*

---

## Slide 7 — Los datos (1.5 min)

*"Aquí tuve el primer reto real: **datos de spam en español casi no hay**. El benchmark estándar, el SMS Spam Collection de UCI, es en inglés. Entonces combiné tres fuentes: ese dataset inglés, una versión traducida al español, y —esto es clave— un dataset de spam **nativo** en español, escrito directamente, más un puñado de ejemplos de phishing local que curé a mano."*

*"En total unos 11 400 mensajes, con 16 % de spam. Ese desbalance no lo inventé yo: refleja la realidad, la mayoría de los mensajes que uno recibe son legítimos."*

> ❓ **¿El español es solo traducción?**
> *"No, y eso me importaba. Añadí spam nativo precisamente para no depender de traducción automática. Les adelanto algo honesto: añadir el nativo bajó un pelín el F1 del español, pero porque la evaluación se volvió más realista. Prefiero un número honesto a uno inflado por traducciones."*

---

## Slide 8 — Exploración (1 min)

*"Antes de modelar exploré el corpus. Aquí se ve la distribución por idioma y clase. Dos cosas: el desbalance hacia ham es claro en ambos idiomas, y el spam tiende a ser más corto y uniforme que el ham, que es más variado. Esto ya me decía que iba a tener que balancear y que la accuracy sola me iba a engañar."*

> Señala la gráfica con el cursor mientras hablas, no la dejes muda.

*"Y un detalle técnico: el corpus ya armado lo guardo en **caché** en `corpus.parquet`, así el notebook es reproducible sin internet, rápido y siempre con los mismos datos."*

---

## Slide 9 — Preprocesamiento (1.5 min)

*"El texto crudo no se le puede dar directo a un modelo. Mi función `clean_text` hace varias cosas: decodifica HTML, pasa a minúsculas, quita puntuación, y —lo más interesante— reemplaza las URLs por el token `__url__` y los números por `__num__`."*

*"¿Por qué reemplazar en vez de borrar? Porque la **presencia** de una URL o un número es una señal fortísima de spam, aunque el valor exacto no importe. Si yo borro la URL, pierdo esa señal; si la convierto en un token, el modelo aprende que 'mensaje con link' tiende a spam."*

> ❓ **¿Los SMS traen HTML? ¿No sobra esa limpieza?**
> *"Casi no, son SMS. Pero hay 516 mensajes con entidades como `&amp;` que, sin decodificar, dejarían basura como el token 'amp' en el vocabulario. Así que esa parte sí es necesaria; y de paso el pipeline serviría para correos, que sí traen HTML."*

---

## Slide 10 — Vectorización (1.5 min)

*"Para convertir texto en números probé tres representaciones. **Bag of Words** simplemente cuenta palabras. **TF-IDF** las pondera: una palabra que aparece mucho en un mensaje pero es rara en el resto pesa más; resalta lo discriminativo. Y la tercera, **n-gramas de caracteres**, que mira trozos de letras en vez de palabras enteras."*

*"Esa tercera fue una decisión pensando en lo bilingüe: los trozos de caracteres son agnósticos al idioma y, sobre todo, **resisten errores de ortografía**. Eso va a ser importante en un momento."*

> ❓ **¿Qué es TF-IDF en una frase?**
> *"Frecuencia de la palabra en el mensaje, penalizada por qué tan común es en todo el corpus. Premia las palabras que de verdad distinguen."*

---

## Slide 11 — Desbalance y balanceo (1.5 min)

*"Con 84 % de ham, un modelo perezoso aprende a decir siempre 'ham' y acierta el 84 %, sin detectar un solo spam. Para evitarlo hice **undersampling**: me quedo con todo el spam y reduzco el ham."*

*"Usé un ratio 1.5 a 1, no mitad y mitad. Es un punto medio: equilibro lo suficiente para que aprenda spam, pero mantengo algo de la realidad de que hay más ham. Lo hice por idioma separado para no quedarme con un corpus mayoritariamente inglés por accidente."*

> ❓ **¿No pierdes información al tirar ham?**
> *"Sí, es un costo real, pero aceptable: el ham es abundante y redundante, y conservo todo el spam, que es lo escaso. Además uso `class_weight='balanced'` en los modelos para compensar el desbalance restante sin tirar más datos. La validación cruzada confirma que generaliza bien."*

> ❓ **¿Por qué undersampling y no oversampling/SMOTE?**
> *"Porque tenía suficiente spam, casi 1 900 ejemplos. El oversampling arriesga sobreajuste a spam duplicado y en texto es incómodo. El undersampling era más simple y suficiente."*

---

## Slide 12 — Partición y Pipeline (1 min)

*"Dividí en 80 % entrenamiento y 20 % prueba, de forma **estratificada** para mantener la proporción de spam en ambos, con semilla fija para que sea reproducible."*

*"Y encadené el vectorizador y el modelo en un **Pipeline**. Esto no es estético: evita el **data leakage**. Como el vectorizador vive dentro del pipeline, en la validación cruzada se ajusta solo con el entrenamiento de cada fold y **nunca ve el test**. Si lo ajustara con todo, estaría haciendo trampa sin darme cuenta."*

> ❓ **¿Qué es data leakage exactamente?**
> *"Cuando información del test se filtra al entrenamiento. Por ejemplo, si calculo el TF-IDF usando también el test, el modelo ya 'conoce' palabras del test. El Pipeline lo impide."*

---

## Slide 13 — Los tres modelos (1.5 min)

*"Comparé tres. **Naive Bayes**, que es generativo: usa el teorema de Bayes y asume que las palabras son independientes —de ahí lo de 'naive'—. **Regresión Logística** con TF-IDF de palabras, que es discriminativa: aprende directamente una frontera, con una sigmoide que convierte una combinación de pesos en probabilidad. Y la misma logística pero con n-gramas de caracteres."*

> ❓ **¿Generativo vs discriminativo?**
> *"El generativo modela cómo se generan los datos de cada clase y aplica Bayes; el discriminativo aprende directo la frontera entre clases. La logística suele ganar cuando hay datos suficientes."*

> ❓ **¿Qué hace el parámetro C?**
> *"Controla la regularización; C grande es menos regularización, el modelo se ajusta más. Mi GridSearch eligió C=10, que ajusta bastante —de hecho me dio un sobreajuste leve, lo cuento en un momento."*

---

## Slide 14 — Comparación de modelos (1.5 min)

*"Aquí está la comparación en el test. Los tres superan el 90 % de F1. Lo interesante: con el español nativo incluido, el modelo de **n-gramas de caracteres** fue el mejor sin afinar, con F1 0.923 y el mejor recall. Tiene sentido: los trozos de caracteres aguantan mejor el español variado."*

*"Aun así, elegí la **logística de palabras como base para afinar**, por interpretabilidad, y tras el tuning terminó siendo la mejor en general. Lo veo en dos slides."*

> ❓ **Si char n-grams era el mejor, ¿por qué no lo elegiste?**
> *"Como modelo base era el mejor, sí. Pero al afinar la logística de palabras con GridSearch superó a todos. Y la de palabras es más interpretable: puedo ver qué palabra pesó. Fue un balance entre rendimiento e interpretabilidad."*

---

## Slide 15 — Validación cruzada (1.5 min)

*"Para no fiarme de un solo split usé **validación cruzada de 5 folds**: parto el entrenamiento en 5 trozos y entreno 5 veces, rotando cuál trozo es el examen. Importante: cada vez entreno con el 80 % y valido con el 20 %, no con pedazos diminutos; los datos se reutilizan."*

*"Lo que busco es el **gap** entre el score en train y en validación. Si train es mucho mayor, el modelo memorizó. Aquí el modelo de palabras dio un gap de 0.051, que es un **sobreajuste leve**, mientras que el de caracteres generaliza mejor, con 0.036."*

> ❓ **¿Esos 5 modelos son el modelo final?**
> *"No, son desechables, solo sirven para estimar qué tan bien generaliza el método. El modelo final se entrena con todo el train."*

> ❓ **¿De dónde sale el umbral 0.05 del gap?**
> *"Es una regla práctica: una diferencia mayor a ~5 puntos entre train y validación se considera señal de sobreajuste. No es una ley, es un criterio."*

---

## Slide 16 — Curva de aprendizaje (1 min)

*"Esta curva confirma lo anterior. El F1 en train es casi perfecto, 0.999, pero en validación se queda en 0.939; esa brecha es el sobreajuste leve. Lo bueno es que la curva de validación sigue subiendo, lo que me dice que **con más datos —sobre todo más spam nativo— el modelo mejoraría**."*

---

## Slide 17 — Tuning con GridSearch (1.5 min)

*"Para afinar usé GridSearch, que prueba todas las combinaciones de hiperparámetros. Tenía 3 perillas: el tipo de n-grama, el mínimo de frecuencia y la regularización C. Eso da 2 por 3 por 3, 18 combinaciones, cada una con validación de 3 folds, o sea 54 entrenamientos."*

*"Elegí pocos valores por perilla a propósito, porque el costo crece como producto, no como suma. La mejor combinación fue C=10, min_df=1 y unigramas, y el modelo final llegó a **F1 0.944 y accuracy 0.955** en el test."*

> ❓ **¿Por qué 3 folds aquí y 5 en la validación cruzada?**
> *"Porque GridSearch ya multiplica el costo por 18. Con 3 folds lo mantengo rápido; la validación seria de 5 folds la hago por separado."*

> ❓ **¿Por qué C en 0.1, 1, 10 y no 0.1, 0.2, 0.3?**
> *"Porque el efecto de C es multiplicativo, por eso se explora en escala logarítmica, de diez en diez. Es lo estándar."*

---

## Slide 18 — Matriz de confusión (1.5 min)

*"Todas las métricas salen de cuatro números. Esta es la matriz de confusión. Los verdaderos negativos, 537, son ham bien clasificado; los verdaderos positivos, 353, spam detectado. Los dos errores: 22 falsos positivos, ham bloqueado por error —molesto—; y 20 falsos negativos, spam que se coló —peligroso—. En total 42 errores, un 4.5 %."*

*"La clave es que esos dos errores **no cuestan lo mismo**, y por eso necesito varias métricas, no una."*

---

## Slide 19 — Por qué cuatro métricas (1.5 min)

*"La **accuracy** es el porcentaje total de aciertos, pero engaña con datos desbalanceados: 'todo ham' da 84 % sin servir de nada. La **precision** mide, de lo que marqué spam, cuánto era spam de verdad; vigila las falsas alarmas. El **recall** mide, de todo el spam real, cuánto atrapé; es mi prioridad por seguridad. Y el **F1** combina precision y recall con media armónica, que castiga si una de las dos está baja."*

*"Por eso reporto las cuatro más la matriz: cada una tapa el punto ciego de la otra, y cualquiera sola se puede falsear."*

> ❓ **¿Por qué media armónica y no promedio normal en F1?**
> *"Porque el promedio normal se deja engañar: precision 100 y recall 0 daría 50, parece decente. La armónica da casi cero. Solo es alta si las dos son altas."*

> ❓ **Diferencia entre accuracy y precision?**
> *"Accuracy mira las dos clases sobre el total. Precision mira solo lo que predije como spam: qué tan confiable es mi alarma."*

---

## Slide 20 — Calibración del threshold (2 min) ← PIEZA CLAVE

*"Esta es la parte que más me gusta. La decisión por defecto usa 0.5, pero ese número es arbitrario. Como mi prioridad es el recall, **calibré el umbral**: busqué el que maximiza recall manteniendo una precisión razonable, y salió 0.278."*

*"El efecto es directo: con 0.5 tenía recall 0.946; bajando a 0.278 subo el recall a **0.976**, es decir, solo se me escapa el 2.4 % del spam. A cambio baja un poco la precisión. Ese es el trade-off, y lo decidí conscientemente por el contexto de seguridad. En un sistema real, los falsos positivos van a una carpeta de cuarentena revisable, no se borran."*

> ❓ **¿No es hacer trampa bajar el umbral para que dé mejor recall?**
> *"No, porque lo elijo con una regla explícita —máximo recall manteniendo precisión ≥ 0.85— y reporto las dos métricas. No estoy ocultando la caída de precisión, la estoy decidiendo a propósito."*

---

## Slide 21 — Evaluación cross-lingual (1 min)

*"Evalué por idioma por separado. En inglés F1 0.962, en español 0.934. El español rinde algo menos, pero su test es el doble de grande y mucho más diverso, porque incluye el spam nativo. Para mí ese 0.934 es más honesto que un número alto medido solo sobre traducciones."*

---

## Slide 22 — Dificultades y decisiones (2 min) ← MUY IMPORTANTE

*"Quiero ser transparente con el camino, porque ahí está el aprendizaje real:"*

- *"Los datasets se caían: lo resolví con cascada de fallback y caché."*
- *"El spam español escaseaba: combiné tres fuentes."*
- *"El desbalance: undersampling más class_weight."*
- *"Y un intento que **falló**: el dataset nativo tenía etiquetas ruidosas, así que probé limpiarlas con una técnica llamada *confident learning*, entrenando un modelo de referencia para detectar mal etiquetados. Pero al revisarlo, estaba **borrando spam que estaba bien etiquetado**, solo porque el modelo de referencia no conocía ese estilo. Así que lo **descarté**. Para mí eso es una decisión de criterio: no toda técnica sofisticada ayuda."*
- *"Descubrí que los modelos de palabras son frágiles con typos: si escribo 'winn' en vez de 'win', la palabra no está en el vocabulario y el modelo la ignora. Eso es justo lo que motivó los n-gramas de caracteres."*
- *"Y noté que Naive Bayes da probabilidades exageradas, del 99 %, pero tiene el F1 más bajo: está mal calibrado por su suposición de independencia. Por eso elegí la logística."*

> ❓ **¿Por qué Naive Bayes da más alto pero no es el mejor?**
> *"Porque está sobre-confiado: multiplica probabilidades asumiendo independencia y se satura al 99 %. Pero confianza no es acierto: tiene el recall y el F1 más bajos. La logística está mejor calibrada."*

---

## Slide 23 — Demo Gradio (2 min)

*"Por último, hice una interfaz en Gradio con dos pestañas. En la primera clasifico cualquier mensaje y muestro las probabilidades reales, la decisión, y —esto me parece útil— qué palabras reconoce el modelo y cuáles ignora. En la segunda calculo las métricas sobre el test con la matriz de confusión y las fórmulas, y se recalculan al mover el umbral."*

*"Hagamos una prueba en vivo…"* [clasifica un mensaje del público; cambia de modelo; mueve el threshold y muestra cómo un dudoso pasa de ham a spam].

> Plan B si falla la red: *"Aquí está la salida ya guardada en el notebook"* — y muestras la celda.
> Si alguien mete un typo y sale ham: *"Justo lo que hablábamos: 'winn' no está en el vocabulario, miren, la interfaz lo marca como palabra desconocida."*

---

## Slide 24–25 — Resultados y conclusiones (1.5 min)

*"En resumen: un detector bilingüe con F1 0.944 y accuracy 0.955, recall del 97.6 % tras calibrar, que generaliza a los dos idiomas y sin sobreajuste grave. Pero más allá de los números, lo que me llevo es el criterio: priorizar recall por el contexto, equilibrar interpretabilidad y rendimiento, y tener la honestidad de descartar lo que no funcionó."*

*"De trabajo futuro: más spam español nativo, probar modelos tipo transformer multilingüe, y reentrenar periódicamente porque el spam evoluciona. Muchas gracias, quedo atento a sus preguntas."*

---

## Apéndice — Banco de preguntas rápidas

> Para repasar antes de entrar. Respóndelas en voz alta en ~20 s cada una.

1. **¿Qué es clasificación binaria supervisada?** → Predecir una de 2 categorías aprendiendo de datos etiquetados.
2. **¿Naive Bayes vs Logística?** → Generativo (Bayes + independencia) vs discriminativo (frontera directa). Logística gana con datos suficientes y está mejor calibrada.
3. **¿Qué es TF-IDF?** → Frecuencia del término penalizada por su frecuencia en el corpus; resalta palabras discriminativas.
4. **¿Por qué char n-grams?** → Robustos a typos y agnósticos al idioma; mejores en corpus bilingüe.
5. **¿Cómo evitas data leakage?** → El vectorizador dentro del Pipeline se ajusta solo con train en cada fold.
6. **¿Por qué no fiarte de accuracy?** → Con 16 % de spam, 'todo ham' da 84 % sin detectar nada.
7. **¿Por qué priorizas recall?** → Costo asimétrico: un phishing que entra es peor que un ham bloqueado.
8. **¿Qué significa el umbral 0.278?** → Lo calibré para maximizar recall manteniendo precisión ≥ 0.85.
9. **¿Tu modelo sobreajusta?** → Levemente el de palabras (gap 0.051); por eso char n-grams generaliza mejor.
10. **¿Qué es el corpus?** → Toda mi colección de mensajes etiquetados, ~11 400, bilingüe.
11. **¿Qué hace el undersampling 1.5:1?** → Conserva todo el spam y reduce ham a 1.5 por cada spam.
12. **¿Qué harías con más tiempo?** → Más spam español nativo y transformers multilingües.
13. **¿Qué dificultad te marcó más?** → El filtro de ruido que falló: me enseñó que no toda técnica avanzada ayuda.
14. **Si llega un idioma no visto, ¿qué pasa?** → Los char n-grams dan algo de robustez, pero el modelo no está entrenado para eso; lo honesto es decir que requeriría datos de ese idioma.
