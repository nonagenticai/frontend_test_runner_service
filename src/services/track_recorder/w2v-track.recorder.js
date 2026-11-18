/*
 * w2v-track.recorder.js — directed-footage cursor/focus/viewport recorder.
 *
 * Injected into every page via `page.addInitScript` (see w2v.fixtures.ts).
 * Runs on every document (top frame + navigations). It records, in CSS
 * pixels, a per-SCENARIO timeline of cursor motion, clicks, and focus targets,
 * exposing it on `window.__w2vTrack` for the fixture to harvest into a
 * `w2v-track` testInfo attachment.
 *
 * Cross-navigation continuity: a hard navigation gives a fresh `window`, which
 * would otherwise reset the clock and drop earlier events. The whole track —
 * INCLUDING its anchor epoch `_t0` — is therefore mirrored into `sessionStorage`
 * (which survives same-origin navigations within the tab) on every event and
 * restored on each document init, so a login->dashboard flow keeps ONE timeline
 * with `t` relative to scenario start. (A cross-ORIGIN navigation drops
 * sessionStorage — a rare shape for a single-app product demo — and starts a
 * fresh segment; documented limitation.) The fixture prefers the sessionStorage
 * copy so it reads the accumulated track, not just the last document's window.
 *
 * Wire shape (CSS-px; FTR later scales to recorded-video px):
 *   { _t0, focus: [{t, bbox:{x,y,w,h}, action}], cursor: [{t,x,y,click}], viewport:{w,h} }
 *
 * Everything is wrapped in try/catch so a recorder fault can never throw
 * into the page under test.
 */
(function () {
  try {
    // Same-document guard: addInitScript re-runs on soft navigations too; if the
    // recorder is already installed on THIS window, don't re-bind listeners.
    if (window.__w2vTrack) {
      return;
    }

    var KEY = "__w2vTrack";
    // Restore the accumulated track across a same-origin hard navigation; else
    // start a fresh scenario timeline anchored now. `_t0` lives IN the blob so
    // the clock survives the fresh `window`.
    var track = null;
    try {
      var saved = window.sessionStorage.getItem(KEY);
      if (saved) {
        track = JSON.parse(saved);
      }
    } catch (e) {
      /* sessionStorage blocked/unavailable — fall back to in-memory only */
    }
    if (!track || typeof track !== "object" || !track.focus || !track.cursor) {
      track = {
        _t0: Date.now(),
        focus: [],
        cursor: [],
        viewport: { w: window.innerWidth, h: window.innerHeight },
      };
    }
    window.__w2vTrack = track;

    var now = function () {
      return (Date.now() - track._t0) / 1000;
    };
    var persist = function () {
      try {
        window.sessionStorage.setItem(KEY, JSON.stringify(track));
      } catch (e) {
        /* ignore — the live window copy is still harvestable */
      }
    };
    var syncViewport = function () {
      try {
        track.viewport.w = window.innerWidth;
        track.viewport.h = window.innerHeight;
      } catch (e) {
        /* ignore */
      }
    };

    // Throttle cursor sampling to ~50ms so a long drag doesn't flood the track.
    var lastMove = 0;
    document.addEventListener(
      "mousemove",
      function (e) {
        try {
          var t = Date.now();
          if (t - lastMove < 50) {
            return;
          }
          lastMove = t;
          syncViewport();
          track.cursor.push({ t: now(), x: e.clientX, y: e.clientY, click: false });
          persist();
        } catch (err) {
          /* ignore */
        }
      },
      true
    );

    document.addEventListener(
      "mousedown",
      function (e) {
        try {
          syncViewport();
          var t = now();
          var el = document.elementFromPoint(e.clientX, e.clientY);
          var r = el && el.getBoundingClientRect ? el.getBoundingClientRect() : null;
          if (r) {
            track.focus.push({
              t: t,
              bbox: { x: r.x, y: r.y, w: r.width, h: r.height },
              action: "click",
            });
          }
          track.cursor.push({ t: t, x: e.clientX, y: e.clientY, click: true });
          persist();
        } catch (err) {
          /* ignore */
        }
      },
      true
    );

    var recordEditable = function () {
      try {
        syncViewport();
        var el = document.activeElement;
        if (!el || !el.getBoundingClientRect) {
          return;
        }
        var editable =
          el.matches &&
          el.matches("input, textarea, select, [contenteditable], [contenteditable='true']");
        if (!editable) {
          return;
        }
        var r = el.getBoundingClientRect();
        track.focus.push({
          t: now(),
          bbox: { x: r.x, y: r.y, w: r.width, h: r.height },
          action: "type",
        });
        persist();
      } catch (err) {
        /* ignore */
      }
    };
    document.addEventListener("input", recordEditable, true);
    document.addEventListener("focusin", recordEditable, true);
  } catch (outer) {
    /* recorder must never throw into the page */
  }
})();
