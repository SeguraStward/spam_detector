# Presentación — Detector de Spam Bilingüe

Paquete completo para la exposición en clase (~18–20 min).

## Contenido

| Archivo | Qué es |
|---|---|
| [`presentacion.md`](presentacion.md) | **Las diapositivas** (25 slides + portada y cierre). Markdown listo para pasar a LaTeX/Beamer. |
| [`guion.md`](guion.md) | **El guion hablado**, slide por slide, en voz de estudiante, con las dificultades narradas y un **banco de preguntas con respuestas**. |
| [`img/`](img/) | Los 9 gráficos reales del notebook (referenciados desde las diapositivas). |

## Imágenes incluidas

- `exploracion.png` — distribución del corpus por idioma y clase
- `longitud_mensajes.png` — longitud de los mensajes
- `comparativa.png` — comparación de los 3 modelos
- `validacion_cruzada.png` — train vs validación por fold
- `curva_aprendizaje.png` — curva de aprendizaje
- `matriz_confusion.png` — matriz de confusión
- `matriz_comparacion.png` — matriz de confusión: threshold 0.5 vs calibrado
- `curva_pr.png` — curva Precision–Recall y threshold óptimo
- `por_idioma.png` — métricas por idioma (cross-lingual)

## Cómo pasarlo a LaTeX / PDF

**Opción rápida (Pandoc → Beamer):**
```bash
pandoc presentacion.md -t beamer -o slides.pdf
```
Los separadores `---` marcan cada diapositiva y los títulos `##` son los títulos de slide. Las imágenes usan rutas relativas (`img/...`), así que ejecuta el comando **dentro de la carpeta `presentacion/`**.

**Si lo escribes a mano en LaTeX:** cada bloque entre `---` es un `\begin{frame}`; el `##` es el `\frametitle{}`; las tablas y listas se traducen directo.

## Números clave (memorízalos)

- **F1 = 0.938 · Accuracy = 0.946** (modelo tuneado, test n=551)
- **Recall = 0.947** con threshold calibrado **0.307** (calibrado sobre el train, no el test)
- Cross-lingual: **EN F1 0.971 / ES F1 0.905** (español solo nativo, número honesto)
- Corpus: **~6 290** mensajes, **19.6 %** spam, español genuino (nativo + seed)
- Mejores hiperparámetros: **C=10, min_df=2, ngram=(3,5)** (char n-grams)

## Antes de presentar (checklist)

- [ ] Ensayar la **demo de Gradio** (clasificar + cambiar modelo + mover threshold) y tener plan B.
- [ ] Repasar el **banco de preguntas** del final de `guion.md`.
- [ ] Rellenar el/los **nombre(s)** en la portada.
- [ ] Tener el notebook **ya ejecutado** abierto para la demo.
