// Contenido de las diapositivas. Edita aquí el texto/HTML o las notas.
const SLIDES = [
  {
    "kind": "portada",
    "title": "Detector de Spam Bilingüe (Inglés + Español)",
    "html": "<p class='sub'>Clasificación de mensajes con Machine Learning</p><p><strong>Universidad Nacional</strong> · Curso de Inteligencia Artificial<br>Profesor: Venegas · Junio 2026<br>Integrante: Angel Segura</p>",
    "notes": "Buenos días. Mi proyecto es un detector de spam bilingüe, inglés y español, hecho con Machine Learning. Un agente que aprende a separar mensajes legítimos de spam, y quiero ser honesto: les contaré lo que funcionó y también las decisiones y problemas del camino.\n\nTip: arranca con energía, mirando al público."
  },
  {
    "kind": "normal",
    "title": "1. Objetivo",
    "html": "<ul><li><strong>Objetivo:</strong> un agente que <strong>aprende</strong> a distinguir si un mensaje que nunca ha visto es spam.</li><li>Usando <strong>Naive Bayes</strong> y <strong>Logistic Regression</strong> como modelos.</li><li><strong>Bilingüe (EN + ES):</strong> abarcar dos idiomas.</li></ul>",
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
    "title": "11. Modelos utilizados",
    "html": "<p class='sub'>Tres clasificadores de Machine Learning, entrenados y comparados.</p><table><thead><tr><th>Modelo</th><th>Cómo funciona (en breve)</th></tr></thead><tbody><tr><td><strong>Naive Bayes</strong><br>(Bag of Words)</td><td>Cuenta qué palabras aparecen en spam vs ham y, con el teorema de Bayes, elige la clase más probable. Es <em>generativo</em>.</td></tr><tr><td><strong>Regresión Logística</strong><br>(TF-IDF de palabras)</td><td>Asigna un <strong>peso</strong> a cada palabra, los suma y una <strong>sigmoide</strong> convierte ese total en P(spam). Es <em>discriminativo</em>.</td></tr><tr><td><strong>Regresión Logística</strong><br>(char n-grams)</td><td>Igual, pero sobre <strong>trozos de caracteres</strong> → robusto a errores de ortografía y entre idiomas.</td></tr></tbody></table><blockquote>Naive Bayes = probabilístico / generativo &nbsp;·&nbsp; Logística = lineal / discriminativa.</blockquote>",
    "notes": "Usé tres modelos. Naive Bayes es generativo: cuenta palabras por clase y aplica Bayes. La Regresión Logística es discriminativa: pondera cada palabra con un peso, los suma y una sigmoide da la probabilidad de spam. La tercera es la misma logística pero con n-gramas de caracteres, robusta a typos y entre idiomas.\n\nP: ¿Generativo vs discriminativo? R: El generativo modela cómo se ve cada clase y aplica Bayes; el discriminativo aprende directo la frontera entre spam y ham."
  },
  {
    "kind": "normal",
    "title": "12. Comparación de los tres modelos (test)",
    "html": "<img src='img/comparativa.png' alt='Comparación de métricas'><table><thead><tr><th>Modelo</th><th>Acc</th><th>Prec</th><th>Recall</th><th>F1</th></tr></thead><tbody><tr><td>Naive Bayes</td><td>0.924</td><td>0.934</td><td>0.871</td><td>0.902</td></tr><tr><td>LogReg palabras</td><td>0.930</td><td>0.916</td><td>0.909</td><td>0.913</td></tr><tr><td><strong>LogReg char n-grams</strong></td><td>0.938</td><td>0.911</td><td><strong>0.936</strong></td><td><strong>0.923</strong></td></tr></tbody></table>",
    "notes": "En el test, los tres pasan 90% de F1. Con el español nativo, el de char n-grams es el mejor sin afinar (F1 0.923, mejor recall): los trozos de caracteres aguantan el español variado.\nTuneé AMBOS con GridSearch y me quedé con el de mejor F1: ganó char n-grams (F1 0.950, recall 0.957). Que decida el rendimiento, no la intuición.\n\nP: ¿Por qué char y no palabras? R: Tuneé los dos; char dio mejor F1 (0.950 vs 0.944) y mejor recall."
  },
  {
    "kind": "normal",
    "title": "13. Validación cruzada (anti-overfitting)",
    "html": "<img src='img/validacion_cruzada.png' alt='Train vs Validación por fold'><table><thead><tr><th>Modelo</th><th>gap (train − CV)</th><th>Veredicto</th></tr></thead><tbody><tr><td>Naive Bayes</td><td>+0.039</td><td>OK</td></tr><tr><td>LogReg palabras</td><td>+0.051</td><td>⚠️ sobreajuste leve</td></tr><tr><td>LogReg char</td><td>+0.036</td><td>OK (más robusto)</td></tr></tbody></table>",
    "notes": "Para no fiarme de un solo split usé validación cruzada de 5 folds: parto el train en 5 trozos y entreno 5 veces, rotando cuál es el examen. Cada vez entreno con el 80% y valido con el 20%; los datos se reutilizan.\nBusco el gap train−validación: si train es mucho mayor, memorizó. El de palabras dio 0.051 (sobreajuste leve); el de caracteres 0.036.\n\nP: ¿Esos 5 modelos son el final? R: No, son desechables; estiman generalización. El final usa todo el train."
  },
  {
    "kind": "normal",
    "title": "14. Tuning de hiperparámetros (GridSearch)",
    "html": "<ul><li>Tuneamos <strong>ambos</strong> candidatos (palabras y char n-grams) y elegimos el de mejor F1 (CV).</li><li>Gana <strong>char n-grams</strong>: <code>C=10, min_df=2, ngram=(3,5)</code> → F1 macro CV <strong>0.946</strong> (palabras 0.939).</li></ul><table><thead><tr><th>Métrica (modelo tuneado, test)</th><th>Valor</th></tr></thead><tbody><tr><td>Accuracy</td><td><strong>0.959</strong></td></tr><tr><td>Precision</td><td>0.942</td></tr><tr><td>Recall</td><td>0.957</td></tr><tr><td><strong>F1</strong></td><td><strong>0.950</strong></td></tr></tbody></table>",
    "notes": "GridSearch prueba todas las combinaciones. 3 perillas: ngram, min_df y C → 2×3×3 = 18 combinaciones, cada una con 3 folds = 54 entrenamientos. Elegí pocos valores por perilla porque el costo crece como producto. Tuneé ambos modelos y ganó char n-grams (C=10, min_df=2, ngram 3–5) → F1 0.950, accuracy 0.959.\n\nP: ¿Por qué 3 folds aquí y 5 antes? R: GridSearch ya multiplica por 18; con 3 lo mantengo rápido.\nP: ¿Por qué C en 0.1,1,10? R: Su efecto es multiplicativo; se explora en escala logarítmica."
  },
  {
    "kind": "normal",
    "title": "15. Curva de aprendizaje",
    "html": "<p class='sub'>Modelo final: Regresión Logística + char n-grams (ganador del GridSearch).</p><img src='img/curva_aprendizaje.png' alt='Curva de aprendizaje'><ul><li>F1 train <strong>0.998</strong> vs validación <strong>0.946</strong> → brecha ~0.05.</li><li><strong>Sobreajuste leve</strong>: generaliza bien.</li><li>La curva de validación sigue subiendo → <strong>más datos ayudarían</strong>.</li></ul>",
    "notes": "Esta curva confirma lo anterior: F1 en train casi perfecto (0.998) vs validación 0.946; esa brecha es el sobreajuste leve. Lo bueno: la validación sigue subiendo, así que con más datos —sobre todo spam nativo— el modelo mejoraría."
  },
  {
    "kind": "normal",
    "title": "16. Métricas y matriz de confusión",
    "html": "<p class='sub'>Modelo final: <strong>char n-grams (tuneado)</strong> — conjunto de prueba (932 mensajes).</p><table><thead><tr><th>Métrica</th><th>Valor</th><th>Qué mide</th></tr></thead><tbody><tr><td><strong>Accuracy</strong></td><td><strong>0.959</strong></td><td>% total de aciertos (engaña con clases desbalanceadas)</td></tr><tr><td><strong>Precision</strong></td><td>0.942</td><td>De lo marcado como spam, cuánto era spam real (falsas alarmas)</td></tr><tr><td><strong>Recall</strong></td><td>0.957 &rarr; <strong>0.981</strong></td><td>Del spam real, cuánto se atrapó (prioritario; 0.981 al calibrar el umbral)</td></tr><tr><td><strong>F1</strong></td><td><strong>0.950</strong></td><td>Equilibrio entre Precision y Recall (media armónica)</td></tr></tbody></table><img src='img/matriz_comparacion.png' alt='Matriz de confusión: threshold 0.5 vs calibrado'><p style='text-align:center;color:#6b7280'>Efecto de calibrar el umbral, sobre 932 mensajes:</p><table><thead><tr><th>Threshold</th><th>Falsos negativos<br>(spam colado)</th><th>Falsos positivos<br>(ham bloqueado)</th></tr></thead><tbody><tr><td>0.5 (defecto)</td><td>16</td><td>22</td></tr><tr><td><strong>0.253 (calibrado)</strong></td><td><strong>7</strong> ↓</td><td>64 ↑</td></tr></tbody></table><blockquote>Bajar el umbral <strong>reduce el spam que se cuela</strong> (16→7 falsos negativos) a costa de más falsas alarmas (22→64). Es el trade-off que decidimos a propósito: en seguridad priorizamos <strong>Recall</strong>. Ninguna métrica sola basta; por eso reportamos las cuatro + la matriz.</blockquote>",
    "notes": "Aquí muestro qué tan efectivo es el modelo con las cuatro métricas reales y la matriz de confusión, todo junto.\nAccuracy 0.959 es el % total de aciertos, pero engaña con desbalance. Precision 0.942: de lo que marqué spam, cuánto era spam (mide falsas alarmas). Recall 0.957 —y 0.981 tras calibrar el umbral—: del spam real, cuánto atrapé; es mi prioridad por seguridad. F1 0.950 resume el equilibrio.\nLa matriz de confusión lo hace concreto: 16 falsos negativos (spam colado, lo peligroso) y 22 falsos positivos (ham bloqueado) de 932, 4.1% de error. Reporto las cuatro porque cada una tapa el punto ciego de la otra y cualquiera sola se puede falsear.\n\nP: ¿Por qué media armónica en F1? R: El promedio normal se engaña (100/0 daría 50); la armónica da casi 0. Solo es alta si Precision y Recall son altas."
  },
  {
    "kind": "normal",
    "title": "17. Calibración del threshold",
    "html": "<img src='img/curva_pr.png' alt='Curva Precision–Recall'><table><thead><tr><th>Threshold</th><th>Precision</th><th>Recall</th></tr></thead><tbody><tr><td>0.50 (defecto)</td><td>0.942</td><td>0.957</td></tr><tr><td><strong>0.253 (óptimo)</strong></td><td>0.851</td><td><strong>0.981</strong></td></tr></tbody></table><blockquote>Bajar el umbral → Recall <strong>98.1 %</strong> (solo escapa 1.9 % del spam), a costa de algo de Precisión.</blockquote>",
    "notes": "Mi parte favorita. La decisión por defecto usa 0.5, pero es arbitrario. Como priorizo recall, calibré el umbral: el que maximiza recall manteniendo precisión razonable, salió 0.253. Con 0.5 recall 0.957; con 0.253 sube a 0.981 (solo escapa 1.9%), a costa de algo de precisión. Decisión consciente por seguridad; en producción los FP van a cuarentena, no se borran.\n\nP: ¿No es trampa bajar el umbral? R: No, lo elijo con regla explícita (máx recall con precisión ≥ 0.85) y reporto ambas métricas."
  },
  {
    "kind": "normal",
    "title": "18. Evaluación cross-lingual (por idioma)",
    "html": "<img src='img/por_idioma.png' alt='Métricas por idioma'><table><thead><tr><th>Idioma</th><th>n</th><th>Accuracy</th><th>F1</th></tr></thead><tbody><tr><td>Inglés</td><td>314</td><td>0.971</td><td>0.965</td></tr><tr><td>Español</td><td>618</td><td>0.953</td><td>0.941</td></tr></tbody></table><blockquote>El español rinde algo menos pero se mide sobre un test más grande y con spam <strong>nativo</strong> → número honesto.</blockquote>",
    "notes": "Evalué por idioma: inglés F1 0.965, español 0.941. El español rinde algo menos, pero su test es el doble de grande y más diverso (incluye spam nativo). Ese 0.941 es más honesto que un número alto medido solo sobre traducciones."
  },
  {
    "kind": "normal",
    "title": "19. Limitaciones",
    "html": "<ul><li><strong>Desbalance de idiomas:</strong> el español sigue sub-representado y en parte es traducción automática; el spam nativo es de tamaño modesto.</li><li><strong>Mensajes cortos:</strong> el corpus es de SMS, no correos largos con HTML.</li><li><strong>El spam evoluciona:</strong> requiere reentrenamiento periódico para no quedar obsoleto.</li><li><strong>GridSearch acotado:</strong> por costo se probaron pocas combinaciones; un grid mayor podría mejorar.</li></ul>",
    "notes": "Cierro con las limitaciones honestas del proyecto: el español sigue sub-representado y en parte es traducción automática (el nativo es pequeño); el corpus es de mensajes cortos, no correos con HTML; el spam evoluciona, así que habría que reentrenar; y el GridSearch fue acotado por costo. Reconocer límites es parte del rigor."
  },
  {
    "kind": "normal",
    "title": "20. Resultados finales",
    "html": "<ul><li><strong>F1 = 0.950 · Accuracy = 0.959</strong> (modelo tuneado, test).</li><li><strong>Recall 98.1 %</strong> tras calibrar el threshold (objetivo de seguridad).</li><li>Generaliza entre idiomas: <strong>EN 0.965 / ES 0.941</strong>.</li><li>Sin overfitting grave (validado con CV y curva de aprendizaje).</li></ul>",
    "notes": "En resumen: detector bilingüe con F1 0.950 y accuracy 0.959, recall 98.1% tras calibrar, generaliza a ambos idiomas, sin sobreajuste grave. Más allá de los números, me llevo el criterio: priorizar recall por el contexto, equilibrar interpretabilidad y rendimiento, y la honestidad de descartar lo que no funcionó.\nTrabajo futuro: más spam español nativo, transformers multilingües (mBERT), reentrenamiento periódico."
  },
  {
    "kind": "cierre",
    "title": "¡Gracias! ¿Preguntas?",
    "html": "<p class='sub'>Detector de Spam Bilingüe — Machine Learning</p><p>F1 <strong>0.950</strong> · Accuracy <strong>0.959</strong> · Recall <strong>98.1 %</strong></p><p style='opacity:.7'>📭 Has clasificado toda la bandeja.</p>",
    "notes": "Muchas gracias, quedo atento a sus preguntas. (Pulsa S para ver el banco de preguntas si lo necesitas.)"
  }
];
