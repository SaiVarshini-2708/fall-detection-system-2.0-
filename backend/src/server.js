// Express + socket.io server — receives alerts from ML model or mock emitter, validates and persists them, broadcasts to dashboard clients.

import 'dotenv/config';
import express from 'express';
import { createServer } from 'http';
import { Server } from 'socket.io';
import cors from 'cors';
import { buildAlert } from './context_engine.js';
import { validateAlert } from './schema_validator.js';
import { initDatabase, insertAlert, getRecentAlerts, clearAlerts } from './database.js';
import { startMockEmitter } from './mock_emitter.js';

const PORT             = parseInt(process.env.PORT || '3001', 10);
const MOCK_INTERVAL_MS = parseInt(process.env.MOCK_INTERVAL_MS || '4000', 10);
const USE_MOCK         = process.env.USE_MOCK === 'true';

const app    = express();
const server = createServer(app);
const io     = new Server(server, {
  cors: { origin: '*', methods: ['GET', 'POST', 'DELETE'] },
});

app.use(cors());
app.use(express.json());

// ── Health check ──────────────────────────────────────────────────────────────
app.get('/health', (_req, res) => {
  res.json({
    status:           'ok',
    timestamp:        new Date().toISOString(),
    connectedClients: io.engine.clientsCount,
  });
});

// ── Receive alert from ML model (or test curl) ────────────────────────────────
app.post('/api/alert', (req, res) => {
  const { fall_type, pre_activity, post_state, confidence, confirmation_window_ms, location, severity_proxy } = req.body;

  if (!fall_type || !pre_activity || !post_state || confidence === undefined) {
    return res.status(400).json({
      success: false,
      error: 'Missing required fields: fall_type, pre_activity, post_state, confidence',
    });
  }

  // confirmation_window_ms is optional — only present when the ML side ran the
  // adaptive confirmation window. Passed through to buildAlert() so it appears
  // in the stored alert and the real-time dashboard broadcast.
  const alert = buildAlert({ fall_type, pre_activity, post_state, confidence, confirmation_window_ms, location, severity_proxy });
  const { valid, errors } = validateAlert(alert);

  if (!valid) {
    return res.status(422).json({ success: false, errors });
  }

  insertAlert(alert);
  io.emit('fall_alert', alert);

  console.log(`[POST /api/alert] ${alert.severity} | ${alert.fall_type} | ${alert.post_state} | conf: ${alert.confidence}`);

  return res.status(201).json({ success: true, alert });
});

// ── Fetch persisted alert history ─────────────────────────────────────────────
app.get('/api/alerts', (_req, res) => {
  const alerts = getRecentAlerts(100);
  res.json({ success: true, alerts });
});

// ── Clear all alerts ──────────────────────────────────────────────────────────
app.delete('/api/alerts', (_req, res) => {
  clearAlerts();
  io.emit('alerts_cleared');
  console.log('[DELETE /api/alerts] All alerts cleared');
  res.json({ success: true });
});

// ── Socket.io connection lifecycle ────────────────────────────────────────────
io.on('connection', (socket) => {
  console.log(`[Socket] Connected:    ${socket.id}`);
  socket.on('disconnect', () => {
    console.log(`[Socket] Disconnected: ${socket.id}`);
  });
});

// ── Startup ───────────────────────────────────────────────────────────────────
await initDatabase();
server.listen(PORT, () => {
  console.log('');
  console.log('╔══════════════════════════════════════════════════════╗');
  console.log('║        FALL DETECTION SYSTEM — BACKEND READY        ║');
  console.log('╠══════════════════════════════════════════════════════╣');
  console.log(`║  Health:   http://localhost:${PORT}/health              ║`);
  console.log(`║  POST:     http://localhost:${PORT}/api/alert           ║`);
  console.log(`║  GET:      http://localhost:${PORT}/api/alerts          ║`);
  console.log(`║  DELETE:   http://localhost:${PORT}/api/alerts          ║`);
  console.log(`║  Mock:     ${USE_MOCK ? 'ON  (every ' + MOCK_INTERVAL_MS + 'ms)' : 'OFF (USE_MOCK=false)'}                   ║`);
  console.log('╚══════════════════════════════════════════════════════╝');
  console.log('');

  if (USE_MOCK) {
    startMockEmitter(io, MOCK_INTERVAL_MS);
  }
});
