/* Motor de la presentación animada.
   Avanzar: clic mitad derecha · → · Espacio · Enter  → muestra un sello HAM/SPAM ALEATORIO y pasa.
   Retroceder: clic mitad izquierda · ← · ⌫ Backspace.
   Notas: S · Pantalla completa: F · Reiniciar: R. */
(function(){
  "use strict";
  const stage = document.getElementById('stage');
  const notesEl = document.getElementById('notes');
  const progressEl = document.getElementById('progress');
  let idx = 0, busy = false, notesOn = false, current = null;

  function esc(s){ return String(s).replace(/[&<>]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;'}[c])); }

  function progress(){ progressEl.style.width = ((idx + 1) / SLIDES.length * 100) + '%'; }

  function renderNotes(){
    const s = SLIDES[idx];
    notesEl.innerHTML = "<h4>🎤 Notas — " + esc(s.title) + "</h4>" + esc(s.notes || "");
  }

  function build(){
    const s = SLIDES[idx];
    const env = document.createElement('div');
    env.className = 'env kind-' + s.kind;
    const eyebrow = s.kind === 'portada' ? 'Presentación'
                  : s.kind === 'cierre' ? 'Cierre'
                  : 'Detector de Spam';
    env.innerHTML =
      "<div class='content'>" +
        "<div class='ehead'><span class='eyebrow'>" + eyebrow + "</span>" +
        "<span class='subject'>" + esc(s.title) + "</span></div>" +
        "<div class='ebody'>" + s.html + "</div>" +
      "</div>" +
      "<div class='cover'>" +
        "<svg viewBox='0 0 100 100' preserveAspectRatio='none' aria-hidden='true'>" +
          "<polygon points='0,0 100,0 100,60 50,100 0,60' fill='#e7ebf2'/>" +            // cuerpo del sobre
          "<polygon points='0,0 50,46 100,0' fill='#cfd6e2'/>" +                          // solapa de cierre
          "<polyline points='0,0 50,46 100,0' fill='none' stroke='#8b95a7' stroke-width='1.6' vector-effect='non-scaling-stroke'/>" +
          "<polyline points='0,60 50,100 100,60' fill='none' stroke='#8b95a7' stroke-width='1.6' vector-effect='non-scaling-stroke'/>" +
          "<polyline points='0,0 0,60 50,100 100,60 100,0' fill='none' stroke='#b8c0cd' stroke-width='1.2' vector-effect='non-scaling-stroke'/>" +
        "</svg>" +
      "</div>" +
      "<div class='stamp stamp-ham'>Ham ✓</div>" +
      "<div class='stamp stamp-spam'>Spam ✕</div>";
    return env;
  }

  function show(){
    if (current){ current.remove(); current = null; }
    const env = build();
    stage.appendChild(env);
    void env.offsetWidth;                 // reflow → anima la llegada (carta cerrada)
    env.classList.add('arrived');
    setTimeout(() => env.classList.add('open'), 560);  // el pliegue sube y revela el texto
    current = env;
    progress();
    if (notesOn) renderNotes();
  }

  // Avanzar: muestra un sello HAM/SPAM ALEATORIO y pasa al siguiente
  function advance(){
    if (busy || !current) return;
    busy = true;
    const side = Math.random() < 0.5 ? 'ham' : 'spam';
    const env = current;
    env.classList.add(side === 'ham' ? 'show-ham' : 'show-spam');
    setTimeout(() => env.classList.add(side === 'ham' ? 'leave-left' : 'leave-right'), 420);
    setTimeout(() => {
      env.remove(); current = null;
      if (idx < SLIDES.length - 1){ idx++; show(); }
      else {
        stage.innerHTML = "<div class='done'>📭 Fin de la bandeja.<br>Pulsa R para reiniciar.</div>";
        progress();
      }
      busy = false;
    }, 1050);
  }

  function prev(){
    if (busy || idx === 0) return;
    busy = true;
    if (current){ current.classList.add('leave-down'); }
    setTimeout(() => { idx--; show(); busy = false; }, 260);
  }

  function reset(){ if (busy) return; idx = 0; show(); }
  function toggleNotes(){ notesOn = !notesOn; notesEl.classList.toggle('hidden', !notesOn); if (notesOn) renderNotes(); }
  function fs(){ if (!document.fullscreenElement) (document.documentElement.requestFullscreen || function(){})(); else (document.exitFullscreen || function(){})(); }

  // Clic: mitad derecha = avanzar · mitad izquierda = retroceder
  stage.addEventListener('click', e => {
    if (e.clientX < window.innerWidth / 2) prev(); else advance();
  });

  document.addEventListener('keydown', e => {
    if (e.key === 'ArrowRight' || e.key === ' ' || e.key === 'Enter') advance();
    else if (e.key === 'ArrowLeft' || e.key === 'Backspace' || e.key === 'ArrowUp') { e.preventDefault(); prev(); }
    else if (e.key === 's' || e.key === 'S') toggleNotes();
    else if (e.key === 'f' || e.key === 'F') fs();
    else if (e.key === 'r' || e.key === 'R') reset();
  });

  show();
})();
