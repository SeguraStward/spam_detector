# Plan: Cerrar brechas para 100/100 — Proyecto Final Detector de Spam

## Contexto

El notebook [spam_detection.ipynb](spam_detection.ipynb) ya es **content-complete y de alta calidad**: cubre formulación del problema, contexto teórico (Naive Bayes con ejemplo numérico, dos variantes de Regresión Logística), análisis PEAS, EDA, preprocesamiento, balanceo, 3 modelos, cross-validation, GridSearchCV, curva de aprendizaje, tuning de threshold, evaluación cross-lingual por idioma, matriz de confusión, análisis de errores, conclusiones y referencias.

Sin embargo, contra la rúbrica del Prof. Venegas hay **3 brechas en el notebook que cuestan puntos directos** y **2 entregables externos faltantes**. El objetivo es cerrarlas todas para asegurar 100/100.

### Diagnóstico contra la rúbrica

| Rubro | Estado | Riesgo |
|---|---|---|
| Formulación (10) | ✅ Casi completo | Falta formalizar explícito variables E/S |
| Dataset + preproc. (10) | ✅ Completo | Outputs no visibles |
| Implementación (20) | 🔴 **Sistema interactivo NO cumple** | Celda 53 solo evalúa lista fija → la rúbrica dice "se considerará **incompleto**" (5 pts) |
| Comparación modelos (15) | ✅ Completo | Outputs no visibles |
| Evaluación/métricas (15) | ✅ Completo | Outputs no visibles |
| Análisis crítico (10) | ✅ Completo | — |
| Documentación técnica (10) | 🔴 **PDF 8-12 págs no existe** | 10 pts |
| Defensa oral (10) | ⚪ Externo | Preparar presentación de apoyo |

**Decisiones del usuario:** sistema interactivo = ipywidgets **+** Gradio (extensión opcional para sumar). Alcance = todo lo entregable (notebook + PDF + esqueleto de presentación).

---

## A. Notebook — cerrar las 3 brechas

### A1. Sistema interactivo de ingreso manual (CRÍTICO, 5 pts)
Reemplazar/ampliar la sección 20. Insertar **dos celdas nuevas** después de la celda 53, reutilizando la función ya existente `predict_message()` (celda 53) y las variables `best_pipeline` y `optimal_threshold`:

- **Celda ipywidgets** (cumple el mínimo de la rúbrica, queda dentro del .ipynb y es reproducible):
  - `widgets.Textarea` para el mensaje + `widgets.Button` + `widgets.Output`.
  - Al hacer click llama a `predict_message(texto, threshold=optimal_threshold)` y muestra etiqueta, P(spam), confianza y barra de color (rojo spam / verde ham).
  - Incluir comentario con **fallback `input()`** por si el evaluador no renderiza widgets.
- **Celda Gradio** (extensión opcional, sección VII de la rúbrica → suma nota):
  - `gr.Interface(fn=..., inputs=gr.Textbox, outputs=gr.Label)` envolviendo `predict_message`.
  - `demo.launch(share=False, inline=True)` — con nota markdown de que es la interfaz visual externa.
- Añadir `gradio>=4.0` e `ipywidgets>=8.0` a [requirements.txt](requirements.txt).

### A2. Ejecutar el notebook y guardar outputs (afecta "ejecutable" 5 pts + interpretación de métricas)
Los 29 code cells tienen `execution_count: None` y **cero outputs guardados**. El evaluador no puede ver tablas de comparación, gráficos ni la matriz de confusión sin correrlo.
- Ejecutar end-to-end (`jupyter nbconvert --to notebook --execute --inplace` o Run All en el IDE) con el venv del proyecto.
- Verificar que `optimal_threshold`, `best_pipeline`, `clean_text` y `evaluate_model` queden definidos antes de su uso (el orden actual ya es correcto: celdas 18 → 26 → 37 → 42 → 53).
- Confirmar que se generan los plots de las secciones 8, 14, 16, 18, 19 y que la matriz de confusión renderiza.
- Commit del notebook **con outputs** (NO limpiarlos).

### A3. Robustez de reproducibilidad (datos desde internet)
La sección 5/6 descarga datasets de mirrors (ya hay cascada de fallback, bien). Para blindar la evaluación offline:
- Tras la carga unificada (celda 11), **cachear el corpus a disco** (`data/corpus.parquet`) y al inicio intentar cargar el caché antes de descargar.
- Documentar en una celda markdown que si no hay internet, se usa el caché versionado.

### A4. Retoque menor de formulación (asegurar subcriterios)
- En la sección 1 o 3, añadir tabla explícita **Variable de entrada / Variable de salida** (entrada: texto crudo del mensaje; salida: etiqueta ∈ {spam, ham} + P(spam) ∈ [0,1]) — vale 3 pts del rubro Formulación.

---

## B. Documento técnico PDF (8-12 páginas, 10 pts)

Crear `docs/documento_tecnico.md` → exportar a PDF (pandoc o el conversor del IDE). Estructura **exacta** de la rúbrica:
1. Portada (Universidad Nacional, curso, integrantes, fecha 8-jun-2026)
2. Introducción
3. Planteamiento del problema (clasificación binaria supervisada, variables E/S)
4. Metodología (datasets, preprocesamiento, 3 modelos, CV, GridSearch, threshold)
5. Resultados (tabla comparativa de métricas + matriz de confusión + eval por idioma — **copiar los números reales tras ejecutar A2**)
6. Discusión (tradeoff Precision/Recall, cross-lingual, overfitting)
7. Conclusiones
8. Referencias en **formato APA** (Almeida & Hidalgo 2011; Metsis et al. 2006; Russell & Norvig; scikit-learn docs)

Reutilizar el contenido ya escrito en las celdas markdown del notebook (secciones 1-3, 22) como base.

---

## C. Esqueleto de presentación (defensa oral, 10 pts)

Crear `docs/presentacion.md` (formato Marp/reveal o guion para slides), ~12-15 slides para 20 min:
- Problema y motivación · Dataset · Preprocesamiento · Los 3 modelos (intuición) · Comparación de métricas · Demo en vivo (Gradio) · Análisis crítico · Conclusiones.
- Incluir guion de posibles **preguntas técnicas** y respuestas (Naive Bayes vs LogReg, por qué char n-grams, por qué priorizar Recall) para preparar la defensa.

---

## Archivos a crear/modificar
- **Modificar:** [spam_detection.ipynb](spam_detection.ipynb) — celdas interactivas (A1), caché de datos (A3), tabla E/S (A4), + ejecutar con outputs (A2).
- **Modificar:** [requirements.txt](requirements.txt) — `gradio`, `ipywidgets`.
- **Crear:** `docs/documento_tecnico.md` (+ PDF), `docs/presentacion.md`, `data/corpus.parquet` (caché).

## Verificación
1. `jupyter nbconvert --to notebook --execute --inplace spam_detection.ipynb` corre sin errores y guarda outputs.
2. La celda ipywidgets acepta texto escrito a mano y devuelve predicción; la celda Gradio levanta interfaz local.
3. Probar mensajes EN y ES manualmente (ej. "FELICIDADES! Has ganado..." → SPAM; "Hola, paso por ti a las 7" → HAM).
4. PDF generado tiene 8-12 páginas con la estructura de la rúbrica y números reales.
5. `models/spam_classifier.joblib` se regenera y la prueba de carga (celda 55) pasa.
