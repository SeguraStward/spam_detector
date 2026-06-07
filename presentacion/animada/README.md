# Presentación animada — Detector de Spam

Presentación web **animada y autónoma**: cada diapositiva es una **carta blanca** que llega y se
**abre** (un pliegue triangular sube de abajo hacia arriba y revela el texto). Al avanzar se estampa
un sello **HAM o SPAM al azar**. Diseño en **blanco y negro**. No requiere instalar ni compilar nada.

## Cómo usarla

1. Abre **`index.html`** con doble clic (se abre en el navegador).
2. La carta llega y se abre sola.
3. **Clic en la mitad derecha** (o **→**) para avanzar — aparece un sello HAM/SPAM aleatorio.
   **Clic en la mitad izquierda** (o **←**) para retroceder.
4. Para presentar: pulsa **F** (pantalla completa).

> Si los gráficos no se vieran, mantén las carpetas `img/` y `assets/` junto a `index.html`.

## Controles

| Acción | Cómo |
|---|---|
| Avanzar (sello ham/spam aleatorio) | Clic mitad derecha · **→** · **Espacio** · **Enter** |
| Retroceder | Clic mitad izquierda · **←** · **⌫ Backspace** |
| Notas del orador (mostrar/ocultar) | **S** |
| Pantalla completa | **F** |
| Reiniciar | **R** |

- Arriba hay una **barra de progreso** fina (sin contadores).
- Las **notas del orador** (tecla **S**) salen del guion y solo las ves tú.

## Estructura (código modular)

```
animada/
├── index.html          # estructura mínima; enlaza estilos y scripts
├── assets/
│   ├── styles.css      # todo el diseño y las animaciones
│   ├── app.js          # motor: navegación, animación, notas
│   └── slides.js       # CONTENIDO de las diapositivas (edita aquí)
└── img/                # los 8 gráficos del proyecto
```

## ¿Cómo editar el contenido?

Edita **`assets/slides.js`**: es un arreglo `SLIDES` de objetos
`{ kind, title, html, notes }`. Cambia el `html` (acepta listas, tablas, `<img src="img/...">`,
`<blockquote>`, `<strong>`, `<code>`) o las `notes`. Guarda y recarga el navegador — sin compilar.

- `kind`: `"portada"`, `"agenda"`, `"normal"` o `"cierre"` (afecta el estilo).
- Para cambiar colores/animación, edita `assets/styles.css`.

> Si cambias `../presentacion.md`, recuerda reflejarlo aquí (son archivos distintos).

## Accesibilidad

Respeta `prefers-reduced-motion`: con "reducir movimiento" activado, las animaciones se simplifican.
