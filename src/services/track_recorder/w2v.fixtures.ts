// w2v-test-runner: directed-footage auto-fixture (template).
//
// Wired into the downloaded playwright-bdd package via
// `importTestFrom: './fixtures.ts'` in defineBddConfig, so every generated
// spec inherits the `_w2vRecorder` auto-fixture. The fixture injects the
// track recorder as a page-init script and, after the scenario, harvests
// `window.__w2vTrack` into a `w2v-track` testInfo attachment (JSON).
//
// The recorder source is INLINED at write time by
// ExecutionEngine._ensure_track_recorder — the `"__W2V_RECORDER_SOURCE__"`
// placeholder below is replaced with the JSON-encoded recorder JS so there is
// no filesystem path to resolve at test runtime.
import { test as base } from 'playwright-bdd';

const RECORDER_SOURCE = "__W2V_RECORDER_SOURCE__";

export const test = base.extend<{ _w2vRecorder: void }>({
  _w2vRecorder: [
    async ({ page }, use, testInfo) => {
      await page.addInitScript({ content: RECORDER_SOURCE });
      await use();
      try {
        // Prefer the sessionStorage copy: it holds the track ACCUMULATED across
        // same-origin navigations (see w2v-track.recorder.js), whereas the live
        // `window.__w2vTrack` only holds the current document's segment. Fall
        // back to the window object when storage is unavailable.
        const track = await page.evaluate(() => {
          const w = window as unknown as { __w2vTrack?: unknown };
          try {
            const saved = window.sessionStorage.getItem('__w2vTrack');
            if (saved) {
              return JSON.parse(saved);
            }
          } catch {
            /* storage blocked — fall through to the live window copy */
          }
          return w.__w2vTrack || null;
        });
        const t = track as { focus?: unknown[]; cursor?: unknown[] } | null;
        if (t && ((t.focus && t.focus.length) || (t.cursor && t.cursor.length))) {
          await testInfo.attach('w2v-track', {
            body: Buffer.from(JSON.stringify(track)),
            contentType: 'application/json',
          });
        }
      } catch {
        /* never fail a scenario on a recorder harvest fault */
      }
    },
    { auto: true },
  ],
});
