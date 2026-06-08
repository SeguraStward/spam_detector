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
"Formalmente es **clasificación binaria supervisada**: Dos categorias, y los datos estan etiquetados.

---

**3. PEAS.**
"Apliqué el marco PEAS. Desempeño: priorizo el **Recall** sobre spam, para que pasen la menor cantidad posible de spam. Entorno: mensajes en inglés o español con URLs, números y errores de ortografía. Actuadores: la etiqueta más un nivel de confianza. Sensores: el texto del mensaje."

---

**4. Tipo de entorno y agente.**
"El entorno es observable, un solo agente, determinístico, episódico y estático — de los más simples, por eso son viables modelos clásicos. Y es un agente basado en modelo: aprende una representación interna y luego la aplica."

---

**5. Los datos.** ⭐
"Aquí tuve el primer reto: datos de spam en español casi no hay. El benchmark estándar, el SMS Spam Collection de UCI, es en inglés. Así que combiné el inglés de UCI con un dataset de spam **nativo** en español y un puñado de phishing local que curé a mano. Antes probé también una traducción automática del UCI, pero la **descarté**: eran los mismos mensajes ingleses traducidos, así que el original y su gemelo coexistían y el split los repartía entre train y test → una fuga que inflaba el español tramposamente. Me quedé con español genuino."

"En total unos 6 290 mensajes, con 19.6 % de spam. Ese desbalance no lo inventé yo: refleja la realidad, la mayoría de los mensajes son legítimos."

---

**6. Exploración (EDA).**
"Antes de modelar exploré el corpus. El desbalance hacia ham es claro en ambos idiomas, y el spam tiende a ser más largo (media ~94 caracteres vs ~68 del ham). Eso ya me anticipaba dos cosas: que tendría que balancear y que la accuracy sola me iba a engañar. El corpus armado lo guardo en caché para que el notebook sea reproducible sin internet."

---

**7. Preprocesamiento.**
"Limpio el texto y —lo más interesante— reemplazo las URLs por el token `__url__` y los números por `__num__`. ¿Por qué reemplazar en vez de borrar? Porque la **presencia** de un link o un número es señal de spam, aunque el valor exacto no importe."

---

**8. Vectorización.**
"El modelo solo entiende números, así que convierto el texto en vectores. Probé tres representaciones: Bag of Words (conteo), TF-IDF (palabras ponderadas) y n-gramas de caracteres (trozos de letras)."

---

**9. Desbalance y balanceo.**
"Con 80 % de ham, un modelo perezoso aprende a decir siempre 'ham' y acierta el 80 % sin detectar nada. Para evitarlo hago **undersampling** —me quedo con todo el spam y reduzco el ham, ratio ~1.5 a 1 (en español, donde el ham nativo es escaso, queda casi 1 a 1)— y además uso `class_weight='balanced'` para que los errores en spam pesen más. Doble defensa."

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
"Aquí los comparo en el test. Los tres rondan 0.90–0.93 de F1. Sin afinar, el mejor es el de **char n-grams** (F1 0.926 y el mejor Recall, 0.907): los trozos de caracteres aguantan mejor el español variado. Naive Bayes tiene la mejor precisión pero el peor recall (0.842)."

"No elegí 'a ojo': tuneé los dos mejores con GridSearch y dejé que decidiera el rendimiento. Eso lo vemos en la siguiente slide."

---

**13. Validación cruzada.** ⭐
"Para no fiarme de un solo split usé validación cruzada de 5 folds: parto el train en 5 trozos y entreno 5 veces, rotando cuál es el examen. Importante: cada vez entreno con el 80 % y valido con el 20 %, los datos se reutilizan."

"Acá comparo dos notas. La de **train** es qué tan bien le va al modelo con los **mismos datos** con los que aprendió; la de **validación** es con datos que **no vio**. La analogía que uso: un estudiante que practica con un cuadernillo de ejercicios resueltos. La nota de train es si en el examen le ponen **los mismos ejercicios** que ya practicó; la de validación, si le ponen **ejercicios nuevos** del mismo tema. La que importa es la segunda, porque en producción siempre llegan mensajes nuevos."

"Lo que miro es la **brecha**: train menos validación. Si el train es mucho mayor que la validación, es que **memorizó** en vez de entender. Comparen dos estudiantes: el A saca 95 en lo visto y 92 en lo nuevo —brecha de 3, **aprendió de verdad**; el B saca 99 en lo visto pero 70 en lo nuevo —brecha de 29, **se aprendió las respuestas de memoria** y se hunde en cuanto cambia el ejercicio. Eso último es el overfitting."

"Como regla práctica uso un umbral de **0.05**: por debajo, el modelo memoriza poco y generaliza; por encima, alarma. No es una ley exacta, es un semáforo —y nunca lo miro solo: lo que confirma que sirve es que la **nota de validación sea alta**, no solo que la brecha sea chica. Aquí las tres brechas quedan por debajo de 0.05 (Naive Bayes 0.041, palabras 0.047, char 0.044) **y** la validación ronda 0.91–0.92, así que generalizan bien; el de caracteres es de los más estables."

> ❓ *¿Por qué 0.05 y no otro número?* → "Es una regla práctica, no una ley; cinco puntos de caída es una alarma razonable. Pero el veredicto real son las dos cosas juntas: brecha chica **y** validación alta."
> ❓ *¿Un train alto es malo?* → "No por sí solo. Es malo solo si la validación es mucho más baja; lo preocupante es la caída entre una y otra, no el número del train."

---

**14. GridSearch (tuning).** ⭐
"GridSearch prueba combinaciones de hiperparámetros y elige la mejor por validación cruzada. **Tuneé los dos candidatos** —palabras y char— y me quedé con el de mejor F1. **Ganó char n-grams** (F1 macro CV 0.926 vs 0.915 del de palabras). El modelo final llega a **F1 0.938, Accuracy 0.946** en el test."

> ❓ *¿Por qué char y no palabras?* → "Tuneé ambos; char dio mejor F1 macro en CV (0.926 vs 0.915) y mejor recall. No lo elegí por intuición, lo decidió el rendimiento."

---

**15. Curva de aprendizaje.**
"Esta gráfica se parece a la de validación cruzada —también tiene una línea de train y otra de validación— pero **responde otra pregunta**. La de validación cruzada, con los datos fijos, preguntaba '¿el modelo memoriza o generaliza?'. Esta, en cambio, va **agregando datos** —el eje X es la cantidad de datos de entrenamiento, del 10 % al 100 %— y pregunta '¿me ayudaría conseguir **más datos**?'."

"Es la curva del modelo final, char n-grams. El train va casi en 1.0 y la validación sube hasta 0.926; esa brecha (~0.07, algo mayor que antes porque el corpus es más pequeño tras quitar las traducciones) es el sobreajuste leve. Lo bueno: la línea de validación **sigue subiendo** al llegar al 100 % —no se aplanó—, así que con más datos —sobre todo spam nativo— el modelo mejoraría."

> ❓ *¿No es la misma gráfica que la de validación cruzada?* → "Se parecen, pero no. Las dos usan validación cruzada por dentro; la diferencia es el eje X: allá barría los folds con datos fijos (mide memorización), acá barro la cantidad de datos (mide si más datos ayudarían)."

---

**16. Calibración del threshold.** ⭐ (la pieza fuerte)
"Esta es la parte que más me gusta, y la pongo **antes** de las métricas porque las métricas finales ya usan este umbral; primero explico de dónde sale. El modelo no dice 'spam' o 'ham': dice una **probabilidad**, P(spam). Una regla decide: si esa probabilidad supera un **umbral**, es spam. Por defecto el umbral es 0.5, pero ese número es arbitrario. Como mi prioridad es el Recall, lo **bajo** para atrapar más spam, y busqué el valor óptimo: salió **0.307**."

"La pregunta clave es: ¿con qué datos pruebo los umbrales? Y aquí va lo importante: **no con el test**. El test es mi examen final, que solo debo tocar una vez, al final. Si elijo el umbral mirando el test y luego reporto la nota del test, es como **elegir mis respuestas viendo la hoja de respuestas del examen**: la nota sale inflada, es trampa."

"Entonces lo hago con el train. Pero ojo, hay un truco: si le pido la probabilidad a mensajes que el modelo **ya usó para entrenar**, está demasiado seguro —casi los memorizó— y me da números irreales. Es como preguntarle a un estudiante qué tan difícil es un ejercicio que ya practicó mil veces: dirá 'facilísimo', pero porque se lo sabe. Por eso uso **out-of-fold**: parto el train en trozos y, para cada mensaje, su probabilidad la calcula un modelo que **NO lo vio** al entrenar —como darle al estudiante ejercicios que aparté y no le dejé ver—. Así obtengo probabilidades honestas, como si fueran datos nuevos, **sin gastar el examen final**. Con esas elijo el 0.307."

"El efecto es directo: con 0.5 el Recall era 0.923; bajándolo a 0.307 sube a **0.947** a cambio de algo de precisión. Es una decisión consciente por el contexto de seguridad; en un sistema real los positivos van a una carpeta de cuarentena, no se borran."

> ❓ *¿Qué es out-of-fold?* → "La probabilidad de cada mensaje del train la da un modelo que no lo usó para entrenar (k-folds dentro del train). Me da una estimación honesta, como de datos nuevos, sin tocar el test."
> ❓ *¿Por qué no calibrar con el test?* → "Porque elegiría el umbral que justo le queda bien al test y luego reportaría ese mismo test: trampa, número inflado. El test se usa una sola vez, con el umbral ya fijado."
> ❓ *¿No es hacer trampa bajar el umbral?* → "No, lo elijo con una regla explícita —máximo Recall con precisión ≥ 0.85—, lo calibro sobre el train out-of-fold y muestro las dos métricas. No oculto la caída de precisión, la decido a propósito."

---

**17. Métricas y matriz de confusión.** ⭐
"Ahora sí, las métricas del modelo final —y ojo, ya usan el umbral calibrado de la slide anterior. La **accuracy** (0.946) es el % de aciertos, pero engaña con desbalance. La **precision** (0.954) mide, de lo que marqué spam, cuánto era spam de verdad. El **recall** (0.923) mide, de todo el spam real, cuánto atrapé —es mi prioridad. Y el **F1** (0.938) resume el equilibrio."

"La matriz de confusión lo hace concreto: de 551 mensajes, solo 30 errores —19 spam que se colaron y 11 ham bloqueados con el umbral por defecto. Y las dos matrices muestran el trade-off de calibrar: al bajar a 0.307 los falsos negativos caen de 19 a 13 y suben los falsos positivos de 11 a 28. Reporto las cuatro métricas porque cada una tapa el punto ciego de la otra."

> ❓ *¿Por qué no solo accuracy?* → "Con ~20 % de spam, un 'todo ham' da ~80 % de accuracy sin detectar nada. Por eso miro precision, recall y F1."

---

**18. Evaluación cross-lingual.** ⭐
"Evalué por idioma por separado: inglés F1 0.971, español 0.905. Y quiero explicar **por qué el inglés sale tan alto**, porque no es casualidad: son cuatro cosas que se suman. Una, el inglés es el dataset de UCI, un benchmark clásico, limpio y muy bien separado, donde los modelos rutinariamente sacan 97–98 %; ese 0.971 es lo esperado. Dos, el spam en inglés es largo y obvio —'Free', 'WINNER', '£900 prize', números de teléfono—, así que tiene muchísima señal. Tres, hay como cinco veces más datos en inglés (5 100 vs 1 100), así que el modelo aprende mucho mejor sus patrones."

"Y la cuarta, la más interesante: el español es **genuinamente más difícil**. Su spam es corto y genérico ('Haz clic aquí para ganar un premio'), y sobre todo el ham se **parece** al spam —hay ham etiquetado como 'Compra ahora y recibe un descuento especial', que suena a publicidad—. La frontera entre clases es borrosa. Así que esos 7 puntos de diferencia no son un fallo del modelo: reflejan que separar spam en español, con estos datos, es un problema más duro. Por eso es un número honesto."

"Y aquí va algo clave: el F1 **global** (0.938) no es el promedio de los dos; se calcula juntando todos los mensajes en una bolsa. Cae entre el inglés y el español, ponderado por cuántos mensajes aporta cada uno. En mi test hay casi el mismo spam en cada idioma (124 vs 123), por eso queda casi en el medio. Justo por esto evalúo **por idioma separado**: el global escondería que el inglés va a 0.97 y el español a 0.90."

> ❓ *¿Por qué el inglés sale tan alto?* → "UCI es un benchmark fácil y limpio, su spam es obvio y hay 5× más datos. El español es más difícil: spam corto y un ham que se parece al spam."
> ❓ *¿El global es el promedio de los dos idiomas?* → "No, se calcula sobre todos los mensajes juntos; queda entre los dos, ponderado por cantidad. Coincide con el medio solo porque hay casi el mismo spam por idioma."

---

**19. Limitaciones.**
"Para cerrar, soy honesto con los límites: el español sigue sub-representado —uso solo spam genuino (nativo + seed) porque descarté las traducciones que causaban la fuga—; el corpus es de mensajes cortos, no correos largos con HTML; el spam evoluciona, así que habría que reentrenar cada cierto tiempo; y el GridSearch fue acotado por costo. Reconocer los límites es parte del rigor."

---

**20. Resultados finales.**
"En resumen: un detector bilingüe con **F1 0.938, Accuracy 0.946 y Recall del 94.7 %** tras calibrar, que generaliza a ambos idiomas (EN 0.971 / ES 0.905, sin trampas de traducción) y sin sobreajuste grave. Más allá de los números, me llevo el criterio: priorizar Recall por el contexto, dejar que el rendimiento elija el modelo, y ser honesto con lo que no funcionó —incluido descartar las traducciones que inflaban el español."

---

**Cierre.**
"Y eso es todo. Gracias, quedo atento a sus preguntas."

---

## Demo opcional (si la haces)
Clasifica un mensaje del público con el modelo final; cambia de modelo y mueve el threshold para mostrar cómo un caso dudoso pasa de ham a spam. *Plan B si falla:* enseña la matriz y las métricas que ya están guardadas en el notebook.

## Tus números (de memoria)
F1 **0.938** · Accuracy **0.946** · Recall **0.923 → 0.947** (threshold **0.307**, calibrado en train) · EN **0.971** / ES **0.905** · corpus **~6 290** (19.6 % spam) · 30 errores en test.

## Preguntas que pueden caer
- **¿Naive Bayes vs Logística?** Generativo (Bayes + independencia) vs discriminativo (frontera directa).
- **¿Qué es TF-IDF?** Frecuencia de la palabra penalizada por qué tan común es en el corpus.
- **¿Por qué char n-grams?** Robusto a typos y agnóstico al idioma; ganó el tuning.
- **¿Cómo evitas data leakage?** El vectorizador va dentro del Pipeline.
- **¿Qué es el threshold 0.307?** El umbral calibrado (sobre el train) para maximizar Recall (no el 0.5 por defecto).
- **¿Tu modelo sobreajusta?** Leve (brecha ~0.07 en la curva de aprendizaje, gaps de CV <0.05); la validación sigue subiendo → más datos ayudarían.
- **¿Por qué quitaste las traducciones?** Eran el UCI inglés traducido: original y gemelo coexistían y el split causaba fuga cross-lingual; con solo español nativo el número es honesto.
