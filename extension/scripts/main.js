/**
 * Barricade Engine - Content script for barricade.gg
 *
 * Pattern: same as chesscheat (read DOM → engine → overlay highlights)
 * Polls the board for changes, sends state to local engine server,
 * overlays the best move on the board.
 *
 * DOM READING: The board-reading functions below need to be adapted
 * to barricade.gg's actual DOM structure. The current implementation
 * includes multiple strategies to detect the board.
 */

const ENGINE_URL = "http://127.0.0.1:5123";
const POLL_INTERVAL = 1000; // ms between board checks

var engineRunning = false;
var lastBoardState = null;
var pollTimer = null;
var currentDepth = 4;
var autoMode = false;

// ============================================================
// BOARD READING - tries multiple strategies to find the board
// ============================================================

/**
 * Attempt to find the game board element in the DOM.
 * Tries multiple selectors since we don't know the exact structure yet.
 */
function findBoard() {
  const selectors = [
    // Common game board selectors to try
    'canvas',                          // if rendered on canvas
    '[class*="board"]',                // any element with "board" in class
    '[class*="Board"]',
    '[class*="game"]',
    '[id*="board"]',
    '[id*="game"]',
    'svg',                             // if SVG-based
    '[class*="grid"]',
    '[class*="quoridor"]',
    '[class*="barricade"]',
  ];

  for (const sel of selectors) {
    const el = document.querySelector(sel);
    if (el) return { element: el, selector: sel };
  }
  return null;
}

/**
 * Try to read the board state from the DOM.
 * Returns a board state object or null if we can't read it.
 *
 * This function implements multiple strategies:
 * 1. Look for grid cells with pawn/wall indicators
 * 2. Look for data attributes on elements
 * 3. Look for a game state in JavaScript variables
 * 4. Canvas pixel reading (fallback)
 */
function readBoardFromDOM() {
  // Strategy 1: Look for pawn elements by class/attribute
  let state = tryReadFromElements();
  if (state) return state;

  // Strategy 2: Check for game state in window/global variables
  state = tryReadFromGlobals();
  if (state) return state;

  // Strategy 3: Check for React fiber (internal state)
  state = tryReadFromReact();
  if (state) return state;

  return null;
}

function tryReadFromElements() {
  // Look for elements that represent pawns
  const allElements = document.querySelectorAll('[class*="pawn"], [class*="piece"], [class*="player"], [class*="token"]');
  if (allElements.length < 2) return null;

  // Look for elements that represent walls
  const wallElements = document.querySelectorAll('[class*="wall"], [class*="fence"], [class*="barrier"]');

  // Look for cell/square elements to determine grid
  const cellElements = document.querySelectorAll('[class*="cell"], [class*="square"], [class*="tile"]');

  if (cellElements.length < 9) return null; // need at least some cells

  logPanel("Found DOM elements - attempting to parse board");

  // Try to extract positions from element positions/classes
  // This is board-structure-dependent and will need refinement
  try {
    const board = findBoard();
    if (!board) return null;

    const boardRect = board.element.getBoundingClientRect();
    const cellSize = boardRect.width / 9;

    let pawns = [null, null];
    let walls = [];

    // Read pawn positions from their position within the board
    allElements.forEach((el, idx) => {
      if (idx >= 2) return;
      const rect = el.getBoundingClientRect();
      const col = Math.round((rect.left - boardRect.left + rect.width / 2) / cellSize - 0.5);
      const row = Math.round((rect.top - boardRect.top + rect.height / 2) / cellSize - 0.5);
      if (row >= 0 && row < 9 && col >= 0 && col < 9) {
        pawns[idx] = [row, col];
      }
    });

    // Read walls from their positions
    wallElements.forEach(el => {
      const rect = el.getBoundingClientRect();
      const x = (rect.left - boardRect.left) / cellSize;
      const y = (rect.top - boardRect.top) / cellSize;
      const w = rect.width / cellSize;
      const h = rect.height / cellSize;

      // Horizontal wall: wider than tall
      // Vertical wall: taller than wide
      if (w > h * 1.5) {
        const wr = Math.round(y - 0.5);
        const wc = Math.round(x - 0.5);
        if (wr >= 0 && wr < 8 && wc >= 0 && wc < 8) {
          walls.push([wr, wc, 'h']);
        }
      } else if (h > w * 1.5) {
        const wr = Math.round(y - 0.5);
        const wc = Math.round(x - 0.5);
        if (wr >= 0 && wr < 8 && wc >= 0 && wc < 8) {
          walls.push([wr, wc, 'v']);
        }
      }
    });

    if (pawns[0] && pawns[1]) {
      return {
        pawns: pawns,
        walls: walls,
        wallsLeft: [10 - Math.ceil(walls.length / 2), 10 - Math.floor(walls.length / 2)],
        currentPlayer: 0 // will need to detect whose turn it is
      };
    }
  } catch (e) {
    console.log("Barricade Engine: Element parsing error:", e);
  }

  return null;
}

function tryReadFromGlobals() {
  // Many web games store state in global variables
  try {
    // Check common patterns
    if (window.__NEXT_DATA__) {
      logPanel("Found Next.js data - checking for game state");
      // Next.js app - game state might be in props
      const data = window.__NEXT_DATA__;
      // Search through props for game-related data
      const str = JSON.stringify(data);
      if (str.includes("pawn") || str.includes("wall") || str.includes("board")) {
        logPanel("Game data found in __NEXT_DATA__");
        // Would need to parse the specific structure
      }
    }

    // Check for game state on the window object
    for (const key of Object.keys(window)) {
      if (key.toLowerCase().includes('game') || key.toLowerCase().includes('board') ||
          key.toLowerCase().includes('quoridor')) {
        logPanel(`Found global: ${key}`);
      }
    }
  } catch (e) { /* ignore */ }

  return null;
}

function tryReadFromReact() {
  // React apps store state in fiber nodes
  try {
    const rootEl = document.getElementById('__next') || document.getElementById('root') || document.getElementById('app');
    if (!rootEl) return null;

    // Walk React fiber tree to find game state
    const fiberKey = Object.keys(rootEl).find(k => k.startsWith('__reactFiber$') || k.startsWith('__reactInternalInstance$'));
    if (!fiberKey) return null;

    logPanel("Found React fiber - searching for game state...");

    // Traverse fiber tree looking for game state
    let fiber = rootEl[fiberKey];
    let depth = 0;
    const maxDepth = 50;

    while (fiber && depth < maxDepth) {
      if (fiber.memoizedState || fiber.memoizedProps) {
        const state = fiber.memoizedState;
        const props = fiber.memoizedProps;

        // Check if this component has game-related state
        if (state && typeof state === 'object') {
          const stateStr = JSON.stringify(state).substring(0, 500);
          if (stateStr.includes('pawn') || stateStr.includes('wall') || stateStr.includes('board')) {
            logPanel("Found game state in React component!");
            // Parse and return
          }
        }
      }
      fiber = fiber.child || fiber.sibling || fiber.return;
      depth++;
    }
  } catch (e) {
    console.log("Barricade Engine: React parsing error:", e);
  }

  return null;
}

// ============================================================
// ENGINE COMMUNICATION
// ============================================================

async function getBestMove(boardState) {
  try {
    const response = await fetch(`${ENGINE_URL}/api/best-move`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        ...boardState,
        depth: currentDepth,
        timeLimit: 3.0
      })
    });

    if (!response.ok) throw new Error(`Server error: ${response.status}`);
    return await response.json();
  } catch (e) {
    logPanel(`Engine error: ${e.message}`);
    return null;
  }
}

async function checkEngine() {
  try {
    const response = await fetch(`${ENGINE_URL}/api/health`);
    const data = await response.json();
    return data.status === 'ok';
  } catch (e) {
    return false;
  }
}

// ============================================================
// OVERLAY / HIGHLIGHTING
// ============================================================

function clearHighlights() {
  document.querySelectorAll('.barricade-engine-highlight').forEach(el => el.remove());
}

function highlightMove(move, board) {
  clearHighlights();
  if (!board) return;

  const boardEl = board.element;
  const boardRect = boardEl.getBoundingClientRect();
  const cellSize = boardRect.width / 9;

  if (move.type === 'pawn') {
    // Highlight target cell
    const highlight = document.createElement('div');
    highlight.className = 'barricade-engine-highlight target';
    highlight.style.position = 'fixed';
    highlight.style.left = `${boardRect.left + move.col * cellSize}px`;
    highlight.style.top = `${boardRect.top + move.row * cellSize}px`;
    highlight.style.width = `${cellSize}px`;
    highlight.style.height = `${cellSize}px`;
    document.body.appendChild(highlight);
  } else if (move.type === 'wall') {
    // Highlight wall placement
    const highlight = document.createElement('div');
    highlight.className = 'barricade-engine-highlight wall-suggest';
    highlight.style.position = 'fixed';

    if (move.orientation === 'h') {
      // Horizontal wall spans 2 columns between rows
      highlight.style.left = `${boardRect.left + move.col * cellSize}px`;
      highlight.style.top = `${boardRect.top + (move.row + 1) * cellSize - 3}px`;
      highlight.style.width = `${cellSize * 2}px`;
      highlight.style.height = `6px`;
    } else {
      // Vertical wall spans 2 rows between columns
      highlight.style.left = `${boardRect.left + (move.col + 1) * cellSize - 3}px`;
      highlight.style.top = `${boardRect.top + move.row * cellSize}px`;
      highlight.style.width = `6px`;
      highlight.style.height = `${cellSize * 2}px`;
    }

    document.body.appendChild(highlight);
  }
}

// ============================================================
// UI PANEL
// ============================================================

function createPanel() {
  if (document.getElementById('barricade-engine-panel')) return;

  const panel = document.createElement('div');
  panel.id = 'barricade-engine-panel';
  panel.innerHTML = `
    <h3>Barricade Engine</h3>
    <div class="info-row">
      <span class="info-label">Status:</span>
      <span class="info-value" id="be-status">Connecting...</span>
    </div>
    <div class="info-row">
      <span class="info-label">Best Move:</span>
      <span class="info-value" id="be-best-move">—</span>
    </div>
    <div class="info-row">
      <span class="info-label">Eval:</span>
      <span class="info-value" id="be-eval">—</span>
    </div>
    <div class="info-row">
      <span class="info-label">Nodes:</span>
      <span class="info-value" id="be-nodes">—</span>
    </div>
    <div style="margin-top: 6px;">
      <span class="info-label">Depth:</span>
      <select id="be-depth">
        <option value="2">2 (fast)</option>
        <option value="3">3</option>
        <option value="4" selected>4</option>
        <option value="5">5</option>
        <option value="6">6 (strong)</option>
      </select>
    </div>
    <button id="be-analyze">Analyze Position</button>
    <button id="be-auto">Auto-Analyze: OFF</button>
    <button id="be-debug">Debug DOM</button>
    <div id="be-log" style="margin-top:6px;font-size:10px;color:#666;max-height:60px;overflow-y:auto;"></div>
  `;
  document.body.appendChild(panel);

  // Make panel draggable
  let isDragging = false, offsetX, offsetY;
  panel.addEventListener('mousedown', (e) => {
    if (e.target.tagName === 'BUTTON' || e.target.tagName === 'SELECT') return;
    isDragging = true;
    offsetX = e.clientX - panel.getBoundingClientRect().left;
    offsetY = e.clientY - panel.getBoundingClientRect().top;
  });
  document.addEventListener('mousemove', (e) => {
    if (!isDragging) return;
    panel.style.left = `${e.clientX - offsetX}px`;
    panel.style.top = `${e.clientY - offsetY}px`;
    panel.style.right = 'auto';
  });
  document.addEventListener('mouseup', () => isDragging = false);

  // Button handlers
  document.getElementById('be-analyze').addEventListener('click', analyzeOnce);
  document.getElementById('be-auto').addEventListener('click', toggleAuto);
  document.getElementById('be-debug').addEventListener('click', debugDOM);
  document.getElementById('be-depth').addEventListener('change', (e) => {
    currentDepth = parseInt(e.target.value);
  });
}

function logPanel(msg) {
  console.log(`Barricade Engine: ${msg}`);
  const log = document.getElementById('be-log');
  if (log) {
    log.innerHTML = `${msg}<br>${log.innerHTML}`.substring(0, 500);
  }
}

function updatePanel(data) {
  const setEl = (id, val) => {
    const el = document.getElementById(id);
    if (el) el.textContent = val;
  };

  if (data.move) {
    const m = data.move;
    const moveStr = m.type === 'pawn'
      ? `Pawn → (${m.row}, ${m.col})`
      : `Wall ${m.orientation.toUpperCase()} @ (${m.row}, ${m.col})`;
    setEl('be-best-move', moveStr);
  }
  if (data.evaluation !== undefined) setEl('be-eval', data.evaluation.toFixed(4));
  if (data.nodes !== undefined) setEl('be-nodes', data.nodes.toLocaleString());
}

// ============================================================
// MAIN LOGIC
// ============================================================

async function analyzeOnce() {
  const boardState = readBoardFromDOM();
  if (!boardState) {
    logPanel("Could not read board from DOM. Use Debug to inspect.");
    return;
  }

  logPanel("Analyzing position...");
  document.getElementById('be-status').textContent = 'Thinking...';

  const result = await getBestMove(boardState);
  if (result) {
    updatePanel(result);
    const board = findBoard();
    if (board) highlightMove(result.move, board);
    document.getElementById('be-status').textContent = 'Ready';
    logPanel(`Best: ${result.move.type} (${result.move.row},${result.move.col})`);
  } else {
    document.getElementById('be-status').textContent = 'Engine error';
  }
}

function toggleAuto() {
  autoMode = !autoMode;
  const btn = document.getElementById('be-auto');
  if (btn) {
    btn.textContent = `Auto-Analyze: ${autoMode ? 'ON' : 'OFF'}`;
    btn.classList.toggle('active', autoMode);
  }

  if (autoMode) {
    startPolling();
  } else {
    stopPolling();
    clearHighlights();
  }
}

function startPolling() {
  if (pollTimer) clearInterval(pollTimer);
  pollTimer = setInterval(async () => {
    if (!autoMode) return;

    const boardState = readBoardFromDOM();
    if (!boardState) return;

    // Check if board changed
    const stateStr = JSON.stringify(boardState);
    if (stateStr === lastBoardState) return;
    lastBoardState = stateStr;

    logPanel("Board changed - analyzing...");
    const result = await getBestMove(boardState);
    if (result) {
      updatePanel(result);
      const board = findBoard();
      if (board) highlightMove(result.move, board);
    }
  }, POLL_INTERVAL);
}

function stopPolling() {
  if (pollTimer) {
    clearInterval(pollTimer);
    pollTimer = null;
  }
}

function debugDOM() {
  logPanel("=== DOM DEBUG ===");

  // Report what we find
  const board = findBoard();
  if (board) {
    logPanel(`Board found: <${board.element.tagName}> via "${board.selector}"`);
    logPanel(`Board size: ${board.element.getBoundingClientRect().width}x${board.element.getBoundingClientRect().height}`);
  } else {
    logPanel("No board element found!");
  }

  // List all potentially interesting elements
  const interesting = document.querySelectorAll(
    '[class*="pawn"], [class*="piece"], [class*="player"], [class*="token"], ' +
    '[class*="wall"], [class*="fence"], [class*="barrier"], ' +
    '[class*="cell"], [class*="square"], [class*="tile"], ' +
    '[class*="board"], [class*="Board"], [class*="game"], [class*="Game"], ' +
    'canvas, svg'
  );

  logPanel(`Found ${interesting.length} interesting elements:`);
  const seen = new Set();
  interesting.forEach(el => {
    const desc = `<${el.tagName}> .${el.className.toString().substring(0, 60)}`;
    if (!seen.has(desc)) {
      seen.add(desc);
      logPanel(`  ${desc}`);
    }
  });

  // Check for canvas (would need pixel reading)
  const canvases = document.querySelectorAll('canvas');
  if (canvases.length > 0) {
    logPanel(`⚠ Found ${canvases.length} canvas element(s) - game may use canvas rendering`);
    canvases.forEach((c, i) => {
      logPanel(`  Canvas ${i}: ${c.width}x${c.height}`);
    });
  }

  // Check for React
  const root = document.getElementById('__next') || document.getElementById('root');
  if (root) {
    const fiberKey = Object.keys(root).find(k => k.startsWith('__reactFiber'));
    logPanel(fiberKey ? "React app detected" : "No React fiber found");
  }

  // Dump full body class list for clues
  logPanel(`Body classes: ${document.body.className}`);
  logPanel(`Title: ${document.title}`);
}

// ============================================================
// INIT
// ============================================================

async function init() {
  console.log("Barricade Engine: Initializing...");

  // Wait a moment for the page to fully render
  await new Promise(r => setTimeout(r, 1500));

  createPanel();

  // Check engine connection
  const connected = await checkEngine();
  const statusEl = document.getElementById('be-status');
  if (connected) {
    statusEl.textContent = 'Connected';
    statusEl.style.color = '#4caf50';
    logPanel("Engine server connected!");
  } else {
    statusEl.textContent = 'Engine offline - start server';
    statusEl.style.color = '#f44336';
    logPanel("Cannot reach engine at " + ENGINE_URL);
    logPanel("Run: python server/app.py");
  }

  // Auto-detect board
  const board = findBoard();
  if (board) {
    logPanel(`Board detected: ${board.selector}`);
  } else {
    logPanel("No board detected yet. Click Debug to inspect.");
  }
}

init();
