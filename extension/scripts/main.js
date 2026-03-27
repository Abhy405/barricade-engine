/**
 * Barricade Engine — stealth overlay for barricade.gg
 *
 * Reads board from DOM, calls local engine, overlays best move.
 * Press ` (backtick) to toggle overlay on/off.
 * Press Escape to hide.
 */

const ENGINE_URL = "http://127.0.0.1:5123";
const POLL_MS = 800;

let enabled = true;
let lastState = null;
let pollTimer = null;
let depth = 64;
let timeLimit = 30.0;

// ============================================================
// BOARD READER — reads barricade.gg CSS grid DOM
// ============================================================
// Board is a 19x19 CSS grid. Cells at even positions (2,4,...,18).
// Cell grid-area "R / C" → board row=(R-2)/2, col=(C-2)/2
// H-wall slots: 129x14, grid-area "R / C1 / R2 / C2" → wall_row=(R-3)/2, wall_col=(C1-2)/2
// V-wall slots: 14x129, grid-area "R1 / C / R2 / C2" → wall_row=(R1-2)/2, wall_col=(C-3)/2
// Blue pawn (P1) = bg-blue-500, Red pawn (P2) = bg-red-500

function getBoard() {
  const boardEl = document.querySelector('.aspect-square.relative');
  if (!boardEl) return null;

  // Find pawns
  const blue = boardEl.querySelector('.bg-blue-500');
  const red = boardEl.querySelector('.bg-red-500');
  if (!blue || !red) return null;

  function cellPos(pawnEl) {
    // Pawn is inside a cell div. Walk up to find the cell with grid-area.
    let cell = pawnEl.closest('[style*="grid-area"]');
    if (!cell) cell = pawnEl.parentElement;
    const style = cell?.getAttribute('style') || '';
    const m = style.match(/grid-area:\s*(\d+)\s*\/\s*(\d+)/);
    if (!m) return null;
    const gr = parseInt(m[1]), gc = parseInt(m[2]);
    return [(gr - 2) / 2, (gc - 2) / 2];
  }

  const p1 = cellPos(blue);
  const p2 = cellPos(red);
  if (!p1 || !p2) return null;

  // Find placed walls by checking wall slot children
  const walls = [];
  const wallSlots = boardEl.querySelectorAll('.group.focus\\:outline-none');

  wallSlots.forEach(slot => {
    const style = slot.getAttribute('style') || '';
    const m = style.match(/grid-area:\s*(\d+)\s*\/\s*(\d+)\s*\/\s*(\d+)\s*\/\s*(\d+)/);
    if (!m) return;

    const r1 = parseInt(m[1]), c1 = parseInt(m[2]);
    const r2 = parseInt(m[3]), c2 = parseInt(m[4]);
    const w = slot.getBoundingClientRect().width;
    const h = slot.getBoundingClientRect().height;

    // Check if wall is placed: look for a colored child (not transparent)
    const child = slot.firstElementChild;
    if (!child) return;

    const bg = getComputedStyle(child).backgroundColor;
    const isPlaced = bg && bg !== 'rgba(0, 0, 0, 0)' && bg !== 'transparent'
      && !bg.includes('156, 163, 175')  // gray-400 = unplaced hover hint
      && child.getBoundingClientRect().width > 10; // actually visible

    // Also check if the child has a non-transparent explicit bg class
    const cls = child.className || '';
    const hasWallColor = cls.includes('bg-') && !cls.includes('bg-transparent')
      && !cls.includes('bg-gray') && !cls.includes('opacity-0');

    // Additional check: if child has significant visual presence
    const childRect = child.getBoundingClientRect();
    const isVisible = (childRect.width > 20 || childRect.height > 20);

    if ((isPlaced || hasWallColor) && isVisible) {
      if (w > h) {
        // Horizontal wall
        const wr = (r1 - 3) / 2;
        const wc = (c1 - 2) / 2;
        if (wr >= 0 && wr < 8 && wc >= 0 && wc < 8) walls.push([wr, wc, 'h']);
      } else {
        // Vertical wall
        const wr = (r1 - 2) / 2;
        const wc = (c1 - 3) / 2;
        if (wr >= 0 && wr < 8 && wc >= 0 && wc < 8) walls.push([wr, wc, 'v']);
      }
    }
  });

  // Detect whose turn — look for visual cues outside the board
  // The page shows player info panels; the active one usually has a highlight
  // For now, infer from move count parity or just default to 0
  // We'll check if there's a green border or ring on either player panel
  let currentPlayer = 0;
  const panels = document.querySelectorAll('[class*="ring-emerald"], [class*="border-l-green"], [class*="ring-2"]');
  if (panels.length > 0) {
    // If a panel with emerald ring is closer to the bottom, it's player 2's turn indicator
    // Check position relative to board
    const boardRect = boardEl.getBoundingClientRect();
    panels.forEach(p => {
      const pr = p.getBoundingClientRect();
      // If the highlighted panel is below the board center, player 2 (red) is active
      if (pr.top > boardRect.top + boardRect.height / 2) {
        currentPlayer = 1;
      }
    });
  }

  // Wall counts — look for text near player panels showing wall count
  // Default to calculating from placed walls
  const totalWalls = walls.length;
  // Rough split — we'll refine if we find the actual counters
  let wallsLeft = [10, 10];
  const wallCountEls = document.querySelectorAll('[class*="font-mono"]');
  // Try to find wall count numbers in the UI
  // For now, estimate from placed walls
  // P1 walls placed = walls placed during P1 turns ≈ ceil(total/2)
  // This is imprecise; the actual counts come from the UI
  wallsLeft = [10 - Math.ceil(totalWalls / 2), 10 - Math.floor(totalWalls / 2)];

  return {
    boardEl,
    pawns: [p1, p2],
    walls,
    wallsLeft,
    currentPlayer
  };
}

// ============================================================
// ENGINE
// ============================================================

async function analyze(state) {
  // Log board state to server for debugging
  try {
    fetch(`${ENGINE_URL}/api/log`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        event: 'board_read',
        pawns: state.pawns,
        walls: state.walls,
        wallsLeft: state.wallsLeft,
        currentPlayer: state.currentPlayer
      })
    }).catch(() => {});
  } catch {}

  try {
    const resp = await fetch(`${ENGINE_URL}/api/best-move`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        pawns: state.pawns,
        walls: state.walls,
        wallsLeft: state.wallsLeft,
        currentPlayer: state.currentPlayer,
        depth,
        timeLimit
      })
    });
    if (!resp.ok) return null;
    return await resp.json();
  } catch { return null; }
}

// ============================================================
// OVERLAY
// ============================================================

function clearOverlay() {
  document.querySelectorAll('.be-overlay').forEach(el => el.remove());
}

function showMove(move, boardEl) {
  clearOverlay();
  if (!enabled || !boardEl) return;

  const rect = boardEl.getBoundingClientRect();
  // Calculate cell positions from the grid
  // Grid: 19 columns, cells at even positions (2,4,...,18)
  // We need to find actual pixel positions of the target cell/wall

  if (move.type === 'pawn') {
    // Find the target cell element by grid-area
    const gridRow = move.row * 2 + 2;
    const gridCol = move.col * 2 + 2;
    const targetCell = boardEl.querySelector(`[style*="grid-area: ${gridRow} / ${gridCol}"]`);
    if (targetCell) {
      const cr = targetCell.getBoundingClientRect();
      const hl = document.createElement('div');
      hl.className = 'be-overlay';
      hl.style.cssText = `
        position:fixed; left:${cr.left}px; top:${cr.top}px;
        width:${cr.width}px; height:${cr.height}px;
        background:rgba(0,212,255,0.35); border:2px solid #00d4ff;
        border-radius:4px; pointer-events:none; z-index:99999;
        box-shadow:0 0 12px rgba(0,212,255,0.5);
      `;
      document.body.appendChild(hl);
    }
  } else if (move.type === 'wall') {
    // Find the wall slot element
    let gridArea;
    if (move.orientation === 'h') {
      const gr = move.row * 2 + 3;
      const gc = move.col * 2 + 2;
      gridArea = `${gr} / ${gc} / ${gr + 1} / ${gc + 3}`;
    } else {
      const gr = move.row * 2 + 2;
      const gc = move.col * 2 + 3;
      gridArea = `${gr} / ${gc} / ${gr + 3} / ${gc + 1}`;
    }
    // Find the slot with matching grid-area
    const slots = boardEl.querySelectorAll('.group.focus\\:outline-none');
    for (const slot of slots) {
      const style = slot.getAttribute('style') || '';
      if (style.includes(`grid-area: ${gridArea}`)) {
        const sr = slot.getBoundingClientRect();
        const hl = document.createElement('div');
        hl.className = 'be-overlay';
        hl.style.cssText = `
          position:fixed; left:${sr.left}px; top:${sr.top}px;
          width:${sr.width}px; height:${sr.height}px;
          background:rgba(255,193,7,0.5); border:2px solid #ffc107;
          border-radius:3px; pointer-events:none; z-index:99999;
          box-shadow:0 0 10px rgba(255,193,7,0.4);
        `;
        document.body.appendChild(hl);
        break;
      }
    }
  }
}

// ============================================================
// POLL LOOP
// ============================================================

function stateKey(s) {
  return JSON.stringify({ p: s.pawns, w: s.walls, c: s.currentPlayer });
}

async function poll() {
  if (!enabled) return;
  const state = getBoard();
  if (!state) return;

  const key = stateKey(state);
  if (key === lastState) return;
  lastState = key;

  const result = await analyze(state);
  if (result && result.move) {
    showMove(result.move, state.boardEl);
  }
}

function start() {
  if (pollTimer) clearInterval(pollTimer);
  pollTimer = setInterval(poll, POLL_MS);
  poll(); // immediate first check
}

function stop() {
  if (pollTimer) { clearInterval(pollTimer); pollTimer = null; }
  clearOverlay();
}

// ============================================================
// TOGGLE — backtick or Escape
// ============================================================

document.addEventListener('keydown', (e) => {
  if (e.key === '`' || e.key === 'Escape') {
    enabled = !enabled;
    if (enabled) {
      lastState = null; // force re-analyze
      start();
    } else {
      stop();
    }
  }
});

// ============================================================
// INIT
// ============================================================

(async function init() {
  // Wait for page to load
  await new Promise(r => setTimeout(r, 2000));

  // Check engine
  try {
    const resp = await fetch(`${ENGINE_URL}/api/health`);
    const data = await resp.json();
    if (data.status === 'ok') {
      console.log('Barricade Engine: connected, overlay active. Press ` to toggle.');
      start();
    }
  } catch {
    console.log('Barricade Engine: server offline. Run: python3 server/app.py');
  }
})();
