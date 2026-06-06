# Presentación animada — "Bandeja de entrada"

Presentación web **animada y autónoma**: cada diapositiva es un **correo** que llega,
se abre, y tú lo **clasificas como spam o ham** para avanzar (en sintonía con el tema del proyecto).

## Cómo usarla

1. Abre **`index.html`** con doble clic (se abre en tu navegador). No necesitas instalar ni ejecutar nada.
2. Lee el correo y pulsa **Ham** o **Spam** para avanzar al siguiente.
3. Para presentar: pulsa **F** (pantalla completa).

> Si los gráficos no se vieran, asegúrate de que la carpeta `img/` está junto a `index.html`.

## Controles

| Acción | Tecla / botón |
|---|---|
| Clasificar como **Ham** y avanzar | **←** o botón *Ham* |
| Clasificar como **Spam** y avanzar | **→** o botón *Spam* |
| Avanzar (rápido) | **Espacio** / **Enter** |
| Volver al correo anterior | **⌫ Backspace** / **↑** / botón *Anterior* |
| Mostrar/ocultar **notas del orador** | **S** |
| Pantalla completa | **F** |
| Reiniciar la bandeja | **R** |

- Arriba ves el progreso (**Correo N / 24**) y las pilas **Ham** / **Spam**.
- Ambas direcciones avanzan: la izquierda/derecha es solo visual (tú "clasificas" cada slide).
- Las **notas del orador** (tecla **S**) salen del guion y solo las ves tú.

## Contenido

- 24 "correos": portada + las 21 diapositivas de [`../presentacion.md`](../presentacion.md) + cierre.
- Los 8 gráficos están en [`img/`](img/) (copiados de `../img/`).

## ¿Cómo editar el contenido?

El contenido está **incrustado** dentro de `index.html`, en el bloque
`<script type="application/json" id="slides-data">` (un arreglo de objetos
`{kind, title, html, notes}`). Edita ahí el texto/HTML del slide o sus notas.
No hay paso de compilación: guardas y recargas el navegador.

> Si cambias `../presentacion.md`, recuerda reflejarlo a mano aquí (son dos archivos distintos).

## Accesibilidad

Respeta `prefers-reduced-motion`: si tu sistema tiene activado "reducir movimiento",
las animaciones se simplifican a transiciones suaves.
