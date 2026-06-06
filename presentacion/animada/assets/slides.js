// Contenido de las diapositivas. Edita aquí el texto/HTML o las notas.
const SLIDES = [
  {
    "kind": "portada",
    "title": "Detector de Spam Bilingüe (Inglés + Español)",
    "html": "<p class='sub'>Clasificación de mensajes con Machine Learning</p><p><strong>Universidad Nacional</strong> · Curso de Inteligencia Artificial<br>Profesor: Venegas · Junio 2026<br>Integrante(s): _______________________</p>",
    "notes": "Buenos días. Mi proyecto es un detector de spam bilingüe, inglés y español, hecho con Machine Learning. Un agente que aprende a separar mensajes legítimos de spam, y quiero ser honesto: les contaré lo que funcionó y también las decisiones y problemas del camino.\n\nTip: arranca con energía, mirando al público."
  },
  {
    "kind": "agenda",
    "title": "Agenda",
    "html": "<ol><li>Objetivo</li><li>Formulación y PEAS</li><li>Los datos: corpus bilingüe</li><li>Preprocesamiento y vectorización</li><li>Balanceo y partición</li><li>Los tres modelos</li><li>Validación, tuning y calibración</li><li>Métricas y resultados</li><li>Dificultades y decisiones</li><li>Sistema interactivo (demo)</li><li>Conclusiones</li></ol>",
    "notes": "Voy del problema a los datos, luego preprocesamiento, modelos, cómo los validé y afiné, las métricas, y cierro con las dificultades reales y una demo."
  },
  {
    "kind": "normal",
    "title": "1. Objetivo",
    "html": "<ul><li><strong>Objetivo:</strong> un agente que <strong>aprende</strong> a distinguir si un mensaje que nunca ha visto es spam.</li><li>Usando <strong>Naive Bayes</strong> y <strong>Logistic Regression</strong> como modelos.</li><li><strong>Bilingüe (EN + ES):</strong> abarcar dos idiomas para visualizar limitaciones.</li></ul>",
    "notes": "El spam es la puerta de entrada al phishing, fraude y malware. Un filtro de reglas fijas (si dice FREE es spam) se evade fácil; por eso quiero un modelo que APRENDA los patrones y generalice a mensajes nuevos.\nLo hice bilingüe a propósito: el phishing en español (BBVA, SAT) no lo detecta un modelo solo-inglés.\nIdea clave que atraviesa todo: el COSTO ASIMÉTRICO — dejar pasar phishing es peor que bloquear un mensaje bueno.\n\nP: ¿Por qué no un filtro de reglas? R: Es frágil; cambian 'free' por 'fr33' y lo evaden. El modelo aprende combinaciones, no una lista fija."
  },
  {
    "kind": "normal",
    "title": "2. Formulación: clasificación binaria supervisada",
    "html": "<p><strong>Clasificación</strong> (predice categoría) · <strong>binaria</strong> (2 clases) · <strong>supervisada</strong> (datos etiquetados).</p><p><code>f : texto → {spam, ham} + P(spam)</code></p><table><thead><tr><th></th><th>Variable</th><th>Dominio</th></tr></thead><tbody><tr><td>Entrada</td><td><code>text</code></td><td>Cadena Unicode (EN/ES)</td></tr><tr><td>Salida</td><td><code>label</code></td><td>{spam, ham}</td></tr><tr><td>Salida aux.</td><td><code>P(spam)</code></td><td>[0, 1]</td></tr></tbody></table><p><strong>Regla de decisión:</strong> spam si P(spam) ≥ threshold (calibrado, no 0.5 fijo).</p>",
    "notes": "Es clasificación binaria supervisada: clasificación porque predigo una categoría; binaria por 2 clases; supervisada porque entrené con datos etiquetados, como estudiar con un examen resuelto.\nEntrada = texto crudo; salida = etiqueta + probabilidad. La decisión no usa 0.5, sino un umbral calibrado (lo veo al final).\n\nP: ¿Diferencia con regresión? R: La regresión predice un número (un precio); yo predigo una categoría.\nP: ¿Y más de 2 clases? R: Sería multiclase; encontré una clase 'smishing' pero lo mantuve binario."
  },
  {
    "kind": "normal",
    "title": "3. Análisis PEAS del agente",
    "html": "<table><thead><tr><th>Componente</th><th>Definición</th></tr></thead><tbody><tr><td><strong>P</strong> — Performance</td><td>Accuracy, Precision, Recall, F1. Prioridad: <strong>Recall</strong> sobre spam</td></tr><tr><td><strong>E</strong> — Environment</td><td>Mensajes EN/ES con URLs, números, errores ortográficos</td></tr><tr><td><strong>A</strong> — Actuators</td><td>Etiqueta spam/ham + confianza ∈ [0,1] + probabilidades</td></tr><tr><td><strong>S</strong> — Sensors</td><td>El texto crudo del mensaje</td></tr></tbody></table>",
    "notes": "Apliqué PEAS. Desempeño: accuracy, precision, recall, F1, priorizando recall por el costo asimétrico. Entorno: mensajes EN/ES con URLs, números, typos. Actuadores: etiqueta + confianza, no solo sí/no. Sensores: el texto.\n\nP: ¿Por qué una probabilidad y no solo la etiqueta? R: 51% no es 99%; con la probabilidad mandas dudosos a revisión y bloqueas los muy seguros."
  },
  {
    "kind": "normal",
    "title": "4. Tipo de entorno y de agente",
    "html": "<ul><li><strong>Observable</strong> (recibe el mensaje completo)</li><li><strong>Un solo agente</strong> (sin competencia)</li><li><strong>Determinístico</strong> (mismo texto → misma predicción)</li><li><strong>Episódico</strong> (cada mensaje es independiente)</li><li><strong>Estático</strong> (no cambia mientras decide)</li></ul><p><strong>Agente <em>model-based</em>:</strong> aprende un modelo interno (pesos/probabilidades) y lo aplica en inferencia.</p>",
    "notes": "El entorno es observable, un agente, determinístico, episódico y estático: de los más simples, por eso bastan modelos clásicos. Es un agente basado en modelo: aprende una representación interna y la aplica, no usa reglas a mano.\n\nP: ¿Por qué episódico? R: Clasificar un mensaje no afecta al siguiente; no hay memoria entre decisiones."
  },
  {
    "kind": "normal",
    "title": "5. Los datos: corpus bilingüe (3 fuentes)",
    "html": "<table><thead><tr><th>Fuente</th><th>Idioma</th><th>Mensajes</th></tr></thead><tbody><tr><td>SMS Spam Collection (UCI)</td><td>Inglés</td><td>5 574</td></tr><tr><td>SMS Multilingual (traducido)</td><td>Español</td><td>5 572</td></tr><tr><td>spam_ham_spanish (<strong>nativo</strong>)</td><td>Español</td><td>1 207</td></tr><tr><td>Seed phishing local</td><td>Español</td><td>61</td></tr></tbody></table><p><strong>Corpus:</strong> ~11 400 mensajes · <strong>16.3 % spam</strong> (desbalanceado, fiel a la realidad).</p>",
    "notes": "Primer reto real: datos de spam en español casi no hay. El benchmark (UCI) es inglés. Combiné tres fuentes: inglés UCI, una traducción al español, y —clave— un dataset de spam NATIVO en español, más un seed de phishing local que curé a mano.\n~11 400 mensajes, 16% spam. El desbalance refleja la realidad: la mayoría del correo es legítimo.\n\nP: ¿El español es solo traducción? R: No, añadí spam nativo. Bajó un pelín el F1 del español, pero la evaluación es más honesta."
  },
  {
    "kind": "normal",
    "title": "6. Exploración del corpus (EDA)",
    "html": "<img src='img/exploracion.png' alt='Distribución por idioma y clase'><ul><li>Mayoría ham en ambos idiomas → <strong>desbalance real</strong>.</li><li>El spam suele ser más corto/uniforme; el ham, más variado.</li><li>El corpus se guarda en <strong>caché</strong> (<code>corpus.parquet</code>) para reproducibilidad offline.</li></ul>",
    "notes": "Exploré el corpus: el desbalance hacia ham es claro, y el spam tiende a ser más corto y uniforme. Eso ya me decía que tendría que balancear y que la accuracy sola me engañaría.\nDetalle técnico: el corpus armado se guarda en caché (corpus.parquet) para reproducibilidad offline, rápido y determinístico."
  },
  {
    "kind": "normal",
    "title": "7. Preprocesamiento del texto (clean_text)",
    "html": "<ol><li>Decodifica HTML (<code>&amp;amp;</code> → <code>&amp;</code>) y elimina etiquetas.</li><li>URLs → <code>__url__</code>, números → <code>__num__</code>, emails → <code>__email__</code>.</li><li>Minúsculas, quita puntuación, normaliza espacios.</li></ol><blockquote><strong>Decisión:</strong> reemplazar por <em>tokens</em> en vez de borrar — la <strong>presencia</strong> de una URL es señal de spam, aunque el valor concreto no.</blockquote>",
    "notes": "El texto crudo no va directo al modelo. clean_text decodifica HTML, baja a minúsculas, quita puntuación, y reemplaza URLs por __url__ y números por __num__.\n¿Por qué reemplazar y no borrar? Porque la presencia de una URL es señal fuerte de spam; si la borro, pierdo esa señal.\n\nP: ¿Los SMS traen HTML? R: Casi no, pero 516 mensajes traen entidades como &amp; que sin decodificar dejarían basura ('amp') en el vocabulario; sí es necesaria."
  },
  {
    "kind": "normal",
    "title": "8. Vectorización: de texto a números",
    "html": "<table><thead><tr><th>Técnica</th><th>Qué captura</th></tr></thead><tbody><tr><td><strong>Bag of Words</strong></td><td>Conteo de palabras</td></tr><tr><td><strong>TF-IDF</strong></td><td>Palabras ponderadas por relevancia</td></tr><tr><td><strong>char n-grams</strong> (char_wb 3–5)</td><td>Trozos de caracteres → robusto a typos y agnóstico al idioma</td></tr></tbody></table><p>Stopwords EN + ES combinadas (<strong>504</strong> palabras) eliminadas por el vectorizador.</p>",
    "notes": "El objetivo de vectorizar es convertir texto en números, porque el modelo solo hace matemáticas. Bag of Words cuenta palabras; TF-IDF las pondera por relevancia; y los n-gramas de caracteres miran trozos de letras, agnósticos al idioma y robustos a errores de ortografía (importante luego).\n\nP: ¿TF-IDF en una frase? R: Frecuencia de la palabra penalizada por qué tan común es en el corpus; premia las discriminativas."
  },
  {
    "kind": "normal",
    "title": "9. Desbalance y balanceo",
    "html": "<ul><li>Sin balancear: 84 % ham → el modelo aprende a decir 'ham' siempre.</li><li><strong>Undersampling 1.5:1</strong> (ham:spam) por idioma → conserva todo el spam.</li></ul><table><thead><tr><th></th><th>ham</th><th>spam</th></tr></thead><tbody><tr><td>Antes</td><td>9 525</td><td>1 864 (16 %)</td></tr><tr><td>Después</td><td>2 795</td><td>1 864 (<strong>40 %</strong>)</td></tr></tbody></table><blockquote>Doble defensa: undersampling <strong>+</strong> <code>class_weight=\"balanced\"</code> en los modelos.</blockquote>",
    "notes": "Con 84% ham, el modelo perezoso dice siempre 'ham'. Hice undersampling: conservo todo el spam y reduzco el ham, ratio 1.5:1 por idioma (punto medio, no 50/50).\nAdemás uso class_weight='balanced': hace que los errores en spam pesen ~1.5× más, sin tirar datos. Doble defensa.\n\nP: ¿No pierdes info al tirar ham? R: Sí, pero el ham es abundante y redundante; conservo todo el spam y compenso con class_weight. La CV confirma que generaliza."
  },
  {
    "kind": "normal",
    "title": "10. Partición y Pipeline (evitar data leakage)",
    "html": "<ul><li><code>train_test_split</code> <strong>estratificado</strong>, <code>random_state=42</code> → train 3 727 / test 932.</li><li><strong>Pipeline</strong> = vectorizador + modelo encadenados.</li></ul><blockquote>El vectorizador vive <strong>dentro</strong> del Pipeline → en validación se ajusta solo con el train de cada fold y <strong>nunca ve el test</strong>. Eso evita el <em>data leakage</em>.</blockquote>",
    "notes": "Dividí 80/20 estratificado (misma proporción de spam) con semilla fija. Encadené vectorizador + modelo en un Pipeline.\nNo es estético: evita data leakage. El vectorizador, dentro del Pipeline, se ajusta solo con el train de cada fold y nunca ve el test.\n\nP: ¿Qué es data leakage? R: Cuando info del test se filtra al entrenamiento; p.ej. calcular TF-IDF usando el test. El Pipeline lo impide."
  },
  {
    "kind": "normal",
    "title": "11. Los tres modelos",
    "html": "<table><thead><tr><th>#</th><th>Modelo</th><th>Representación</th><th>Tipo</th></tr></thead><tbody><tr><td>1</td><td>Naive Bayes</td><td>Bag of Words</td><td>Generativo</td></tr><tr><td>2</td><td>Reg. Logística</td><td>TF-IDF palabras</td><td>Discriminativo</td></tr><tr><td>3</td><td>Reg. Logística</td><td>char n-grams</td><td>Robusto a typos</td></tr></tbody></table><ul><li><strong>Naive Bayes:</strong> P(spam|x) ∝ P(x|spam)·P(spam) — asume independencia ('naive').</li><li><strong>Logística:</strong> P(spam|x) = σ(w·x + b) — frontera lineal, regularización <code>C</code>.</li></ul>",
    "notes": "Comparé tres. Naive Bayes es generativo: usa Bayes y asume palabras independientes ('naive'). La Logística es discriminativa: aprende directo la frontera con una sigmoide. La tercera es la logística con char n-grams.\n\nP: ¿Generativo vs discriminativo? R: El generativo modela cómo se ve cada clase y aplica Bayes; el discriminativo aprende solo la frontera.\nP: ¿Qué hace C? R: Regularización; C grande = menos regularización, ajusta más. GridSearch eligió C=10 (de ahí un sobreajuste leve)."
  },
  {
    "kind": "normal",
    "title": "12. Comparación de los tres modelos (test)",
    "html": "<img src='img/comparativa.png' alt='Comparación de métricas'><table><thead><tr><th>Modelo</th><th>Acc</th><th>Prec</th><th>Recall</th><th>F1</th></tr></thead><tbody><tr><td>Naive Bayes</td><td>0.924</td><td>0.934</td><td>0.871</td><td>0.902</td></tr><tr><td>LogReg palabras</td><td>0.930</td><td>0.916</td><td>0.909</td><td>0.913</td></tr><tr><td><strong>LogReg char n-grams</strong></td><td>0.938</td><td>0.911</td><td><strong>0.936</strong></td><td><strong>0.923</strong></td></tr></tbody></table>",
    "notes": "En el test, los tres pasan 90% de F1. Con el español nativo, el de char n-grams es el mejor sin afinar (F1 0.923, mejor recall): los trozos de caracteres aguantan el español variado.\nAun así elegí la logística de palabras como base para afinar (por interpretabilidad), y tras el tuning ganó en general.\n\nP: ¿Si char era mejor por qué no lo elegiste? R: Como base sí; pero al afinar, palabras superó a todos y es más interpretable."
  },
  {
    "kind": "normal",
    "title": "13. Validación cruzada (anti-overfitting)",
    "html": "<img src='img/validacion_cruzada.png' alt='Train vs Validación por fold'><table><thead><tr><th>Modelo</th><th>gap (train − CV)</th><th>Veredicto</th></tr></thead><tbody><tr><td>Naive Bayes</td><td>+0.039</td><td>OK</td></tr><tr><td>LogReg palabras</td><td>+0.051</td><td>⚠️ sobreajuste leve</td></tr><tr><td>LogReg char</td><td>+0.036</td><td>OK (más robusto)</td></tr></tbody></table>",
    "notes": "Para no fiarme de un solo split usé validación cruzada de 5 folds: parto el train en 5 trozos y entreno 5 veces, rotando cuál es el examen. Cada vez entreno con el 80% y valido con el 20%; los datos se reutilizan.\nBusco el gap train−validación: si train es mucho mayor, memorizó. El de palabras dio 0.051 (sobreajuste leve); el de caracteres 0.036.\n\nP: ¿Esos 5 modelos son el final? R: No, son desechables; estiman generalización. El final usa todo el train."
  },
  {
    "kind": "normal",
    "title": "14. Curva de aprendizaje",
    "html": "<img src='img/curva_aprendizaje.png' alt='Curva de aprendizaje'><ul><li>F1 train <strong>0.999</strong> vs validación <strong>0.939</strong> → brecha ~0.06.</li><li>Confirma <strong>sobreajuste leve</strong> del modelo de palabras.</li><li>La curva de validación sigue subiendo → <strong>más datos ayudarían</strong>.</li></ul>",
    "notes": "Esta curva confirma lo anterior: F1 en train casi perfecto (0.999) vs validación 0.939; esa brecha es el sobreajuste leve. Lo bueno: la validación sigue subiendo, así que con más datos —sobre todo spam nativo— el modelo mejoraría."
  },
  {
    "kind": "normal",
    "title": "15. Tuning de hiperparámetros (GridSearch)",
    "html": "<ul><li>Rejilla: <code>ngram</code> (2) × <code>min_df</code> (3) × <code>C</code> (3) = <strong>18 combinaciones</strong> × 3 folds = <strong>54 fits</strong>.</li><li>Mejor: <strong>C=10, min_df=1, ngram=(1,1)</strong> → F1 macro CV <strong>0.939</strong>.</li></ul><table><thead><tr><th>Métrica (modelo tuneado, test)</th><th>Valor</th></tr></thead><tbody><tr><td>Accuracy</td><td><strong>0.955</strong></td></tr><tr><td>Precision</td><td>0.941</td></tr><tr><td>Recall</td><td>0.946</td></tr><tr><td><strong>F1</strong></td><td><strong>0.944</strong></td></tr></tbody></table>",
    "notes": "GridSearch prueba todas las combinaciones. 3 perillas: ngram, min_df y C → 2×3×3 = 18 combinaciones, cada una con 3 folds = 54 entrenamientos. Elegí pocos valores por perilla porque el costo crece como producto. Mejor: C=10, min_df=1, unigramas → F1 0.944, accuracy 0.955.\n\nP: ¿Por qué 3 folds aquí y 5 antes? R: GridSearch ya multiplica por 18; con 3 lo mantengo rápido.\nP: ¿Por qué C en 0.1,1,10? R: Su efecto es multiplicativo; se explora en escala logarítmica."
  },
  {
    "kind": "normal",
    "title": "16. Métricas: la matriz de confusión",
    "html": "<img src='img/matriz_confusion.png' alt='Matriz de confusión'><table><thead><tr><th></th><th>Pred. HAM</th><th>Pred. SPAM</th></tr></thead><tbody><tr><td><strong>Real HAM</strong></td><td>TN=537</td><td>FP=22</td></tr><tr><td><strong>Real SPAM</strong></td><td>FN=20</td><td>TP=353</td></tr></tbody></table><p>42 errores / 932 = <strong>4.5 %</strong>.</p>",
    "notes": "Todas las métricas salen de 4 números. TN=537 (ham bien), TP=353 (spam detectado), FP=22 (ham bloqueado, molesto), FN=20 (spam colado, peligroso). 42 errores, 4.5%.\nClave: esos dos errores no cuestan lo mismo; por eso necesito varias métricas, no una."
  },
  {
    "kind": "normal",
    "title": "17. ¿Por qué cuatro métricas y no una?",
    "html": "<ul><li><strong>Accuracy</strong> = (TP+TN)/total — engaña con desbalance.</li><li><strong>Precision</strong> = TP/(TP+FP) — fiabilidad de la alarma (costo de FP).</li><li><strong>Recall</strong> = TP/(TP+FN) — spam atrapado (costo de FN, lo prioritario).</li><li><strong>F1</strong> = media armónica de P y R — castiga el desequilibrio.</li></ul><blockquote>'Todo ham' → Acc 84 %, Recall 0 %. Por eso ninguna métrica sola basta.</blockquote>",
    "notes": "Accuracy es el % total de aciertos, pero engaña con desbalance ('todo ham' da 84%). Precision: de lo que marqué spam, cuánto era spam (falsas alarmas). Recall: de todo el spam, cuánto atrapé (mi prioridad). F1: combina P y R con media armónica, castiga si una está baja.\nReporto las cuatro: cada una tapa el punto ciego de la otra.\n\nP: ¿Por qué media armónica? R: El promedio normal se engaña (100/0 daría 50); la armónica da casi 0. Solo es alta si ambas lo son."
  },
  {
    "kind": "normal",
    "title": "18. Calibración del threshold",
    "html": "<img src='img/curva_pr.png' alt='Curva Precision–Recall'><table><thead><tr><th>Threshold</th><th>Precision</th><th>Recall</th></tr></thead><tbody><tr><td>0.50 (defecto)</td><td>0.941</td><td>0.946</td></tr><tr><td><strong>0.278 (óptimo)</strong></td><td>0.851</td><td><strong>0.976</strong></td></tr></tbody></table><blockquote>Bajar el umbral → Recall <strong>97.6 %</strong> (solo escapa 2.4 % del spam), a costa de algo de Precisión.</blockquote>",
    "notes": "Mi parte favorita. La decisión por defecto usa 0.5, pero es arbitrario. Como priorizo recall, calibré el umbral: el que maximiza recall manteniendo precisión razonable, salió 0.278. Con 0.5 recall 0.946; con 0.278 sube a 0.976 (solo escapa 2.4%), a costa de algo de precisión. Decisión consciente por seguridad; en producción los FP van a cuarentena, no se borran.\n\nP: ¿No es trampa bajar el umbral? R: No, lo elijo con regla explícita (máx recall con precisión ≥ 0.85) y reporto ambas métricas."
  },
  {
    "kind": "normal",
    "title": "19. Evaluación cross-lingual (por idioma)",
    "html": "<img src='img/por_idioma.png' alt='Métricas por idioma'><table><thead><tr><th>Idioma</th><th>n</th><th>Accuracy</th><th>F1</th></tr></thead><tbody><tr><td>Inglés</td><td>314</td><td>0.968</td><td>0.962</td></tr><tr><td>Español</td><td>618</td><td>0.948</td><td>0.934</td></tr></tbody></table><blockquote>El español rinde algo menos pero se mide sobre un test más grande y con spam <strong>nativo</strong> → número honesto.</blockquote>",
    "notes": "Evalué por idioma: inglés F1 0.962, español 0.934. El español rinde algo menos, pero su test es el doble de grande y más diverso (incluye spam nativo). Ese 0.934 es más honesto que un número alto medido solo sobre traducciones."
  },
  {
    "kind": "normal",
    "title": "20. Dificultades y decisiones clave",
    "html": "<table><thead><tr><th>Dificultad</th><th>Decisión</th></tr></thead><tbody><tr><td>Spam español escaso</td><td>3 fuentes (traducido + nativo + seed)</td></tr><tr><td>Desbalance 84/16</td><td>Undersampling 1.5:1 + class_weight</td></tr><tr><td>Etiquetas nativas ruidosas</td><td>Probamos <em>confident learning</em> → <strong>lo descartamos</strong> (borraba spam bueno)</td></tr><tr><td>Typos rompen modelos de palabras</td><td>char n-grams (robusto a OOV)</td></tr><tr><td>Naive Bayes sobre-confiado</td><td>Elegir LogReg (mejor calibrado)</td></tr><tr><td>Umbral 0.5 no óptimo</td><td>Calibrar a 0.278 (prioridad Recall)</td></tr></tbody></table>",
    "notes": "Quiero ser transparente con el camino, ahí está el aprendizaje real:\n- Spam español escaso → 3 fuentes.\n- Desbalance → undersampling + class_weight.\n- Un intento que FALLÓ: el nativo tenía etiquetas ruidosas; probé limpiarlas con confident learning, pero borraba spam BIEN etiquetado porque el modelo de referencia no conocía ese estilo. Lo descarté: no toda técnica sofisticada ayuda.\n- Los modelos de palabras fallan con typos ('winn' no está en el vocabulario) → motivó los char n-grams.\n- Naive Bayes da 99% pero tiene el F1 más bajo: mal calibrado por la independencia. Por eso elegí la logística.\n\nP: ¿Por qué NB da más alto pero no es el mejor? R: Está sobre-confiado (multiplica probabilidades asumiendo independencia, se satura). Confianza no es acierto."
  },
  {
    "kind": "normal",
    "title": "22. Resultados finales",
    "html": "<ul><li><strong>F1 = 0.944 · Accuracy = 0.955</strong> (modelo tuneado, test).</li><li><strong>Recall 97.6 %</strong> tras calibrar el threshold (objetivo de seguridad).</li><li>Generaliza entre idiomas: <strong>EN 0.962 / ES 0.934</strong>.</li><li>Sin overfitting grave (validado con CV y curva de aprendizaje).</li></ul>",
    "notes": "En resumen: detector bilingüe con F1 0.944 y accuracy 0.955, recall 97.6% tras calibrar, generaliza a ambos idiomas, sin sobreajuste grave. Más allá de los números, me llevo el criterio: priorizar recall por el contexto, equilibrar interpretabilidad y rendimiento, y la honestidad de descartar lo que no funcionó.\nTrabajo futuro: más spam español nativo, transformers multilingües (mBERT), reentrenamiento periódico."
  },
  {
    "kind": "cierre",
    "title": "¡Gracias! ¿Preguntas?",
    "html": "<p class='sub'>Detector de Spam Bilingüe — Machine Learning</p><p>F1 <strong>0.944</strong> · Accuracy <strong>0.955</strong> · Recall <strong>97.6 %</strong></p><p style='opacity:.7'>📭 Has clasificado toda la bandeja.</p>",
    "notes": "Muchas gracias, quedo atento a sus preguntas. (Pulsa S para ver el banco de preguntas si lo necesitas.)"
  }
];
