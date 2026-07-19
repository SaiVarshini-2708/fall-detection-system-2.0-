import { readFileSync } from 'fs';
import { fileURLToPath } from 'url';
import path from 'path';

// ─────────────────────────────────────────────────────────────────────────────
// context_engine.js
//
// PURPOSE:
//   This file is the "brain" that converts raw ML model output into something
//   meaningful for a caretaker.
//
//   Our ML model (Team A) detects a fall and sends back three key labels:
//     - fall_type   → HOW the person fell  (slip / trip / faint)
//     - pre_activity → WHAT they were doing just before falling (walking, bending, etc.)
//     - post_state  → WHAT they are doing after hitting the ground (moving, unconscious, etc.)
//     - confidence  → HOW sure the model is (0.0 to 1.0)
//
//   This file takes those four values and produces:
//     1. A SEVERITY level  (CRITICAL / HIGH / MEDIUM)  — tells the caretaker how urgent it is
//     2. A readable MESSAGE — a plain English sentence shown on the dashboard
//     3. A full ALERT object — combines everything + a timestamp, saved to DB and shown in UI
//
//   Flow:
//     ML model output  →  buildAlert()  →  { severity, message, timestamp, ... }
//                                          ↓
//                                    Saved to DB (alerts table)
//                                    + Pushed to caretaker dashboard in real time
// ─────────────────────────────────────────────────────────────────────────────


// ─────────────────────────────────────────────────────────────────────────────
// SEVERITY_MATRIX
//
// This is a 2D lookup table:
//   rows    = fall_type  (how the person fell)
//   columns = post_state (what the person is doing after the fall)
//
// WHY a matrix instead of if/else chains?
//   Because every (fall_type × post_state) combination has a different medical
//   risk level. A flat matrix makes every rule visible at a glance and easy
//   to update if the medical team changes the thresholds.
//
// HOW TO READ IT:
//   SEVERITY_MATRIX['faint']['unconscious']  → 'CRITICAL'
//   SEVERITY_MATRIX['slip']['moving']        → 'MEDIUM'
//
// REASONING BEHIND THE VALUES:
//
//   slip / trip rows:
//     - unconscious → CRITICAL  : No movement after a fall = worst case, call for help immediately
//     - stunned     → HIGH      : Limited movement may mean injury (head, spine, joints)
//     - moving      → MEDIUM    : Person is likely recovering; still needs a check
//     - unknown     → HIGH      : We can't tell what happened → assume the worst for safety
//
//   faint row (always one level more serious than slip/trip):
//     - Fainting is a MEDICAL EVENT (heart, blood pressure, neurological)
//       not just an accident. Even if the person starts moving again, the
//       underlying cause could be life-threatening.
//     - unconscious → CRITICAL  : Fainted + no movement = emergency
//     - stunned     → CRITICAL  : Even slight movement after a faint needs immediate attention
//     - moving      → HIGH      : Moving ≠ safe; the medical cause is still unknown
//     - unknown     → CRITICAL  : Unknown post-state after a faint = always assume emergency
// ─────────────────────────────────────────────────────────────────────────────
const SEVERITY_MATRIX = {
  //           post_state →  unconscious    stunned       moving       unknown
  slip:  { unconscious: 'CRITICAL', stunned: 'HIGH',     moving: 'MEDIUM', unknown: 'HIGH'     },
  trip:  { unconscious: 'CRITICAL', stunned: 'HIGH',     moving: 'MEDIUM', unknown: 'HIGH'     },
  faint: { unconscious: 'CRITICAL', stunned: 'CRITICAL', moving: 'HIGH',   unknown: 'CRITICAL' },
};

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const IMPACT_SEVERITY_CONFIG_PATH = path.join(__dirname, 'impact_severity_config.json');
const IMPACT_SEVERITY_CONFIG = JSON.parse(readFileSync(IMPACT_SEVERITY_CONFIG_PATH, 'utf8'));

const USE_IMPACT_BASED_SEVERITY = process.env.USE_IMPACT_BASED_SEVERITY === 'true';

function bumpSeverityLevel(level) {
  const rank = { MEDIUM: 1, HIGH: 2, CRITICAL: 3 };
  if (rank[level] == null) return 'HIGH';
  if (level === 'CRITICAL') return 'CRITICAL';
  return Object.keys(rank).find((key) => rank[key] === rank[level] + 1) ?? 'HIGH';
}

function deriveSeverityFromImpactData({ severity_proxy, post_state, fall_type }) {
  const proxyLevel = severity_proxy?.toLowerCase();
  const postState = post_state?.toLowerCase();
  const baseSeverity = IMPACT_SEVERITY_CONFIG?.[proxyLevel]?.[postState] ?? 'HIGH';

  let severity = baseSeverity;
  if (fall_type === 'faint') {
    severity = bumpSeverityLevel(severity);
  }

  return severity;
}


// ─────────────────────────────────────────────────────────────────────────────
// FALL_TYPE_PHRASES
//
// Maps the ML model's fall_type label → the opening sentence of the alert message.
//
// The ML model gives us a short code word like 'faint'.
// Caretakers on the dashboard should see plain English, not model codes.
//
// Each phrase is deliberately written in a clinical but readable tone so the
// caretaker immediately understands what happened without needing technical knowledge.
// ─────────────────────────────────────────────────────────────────────────────
const FALL_TYPE_PHRASES = {
  slip:  'Patient slipped',                    // lost footing on a surface (wet floor, slope)
  trip:  'Patient tripped',                    // foot caught on an obstacle
  faint: 'Patient appears to have fainted',   // sudden loss of consciousness / balance — medical cause
};


// ─────────────────────────────────────────────────────────────────────────────
// PRE_ACTIVITY_PHRASES
//
// Maps the ML model's pre_activity label → a context phrase that explains
// WHAT the patient was doing just before the fall occurred.
//
// WHY this matters for caretakers:
//   "Patient slipped while bending down" gives very different medical context
//   than "Patient slipped while walking." Bending + fall could hint at dizziness
//   or blood pressure drop; walking + fall could hint at a trip hazard.
//
// This information also helps caretakers write accurate incident reports.
// ─────────────────────────────────────────────────────────────────────────────
const PRE_ACTIVITY_PHRASES = {
  walking:  'while walking',                          // patient was in motion
  standing: 'while standing still',                   // patient was stationary — may hint at sudden faint
  bending:  'while bending down',                     // positional change — common cause of dizziness falls
  sitting:  'while getting up from a seated position', // orthostatic — blood pressure drop on standing up
};


// ─────────────────────────────────────────────────────────────────────────────
// POST_STATE_PHRASES
//
// Maps the ML model's post_state label → a sentence describing what the patient
// is doing AFTER hitting the ground.
//
// WHY this matters:
//   The post_state is the most critical piece of information for triage.
//   A patient who is moving is probably recovering; a patient with no movement
//   for 15+ seconds may be unconscious and needs immediate intervention.
//
// These phrases are written to be cautious — they never say "the patient is fine"
// because the system cannot make a medical judgement, only flag a risk.
// ─────────────────────────────────────────────────────────────────────────────
const POST_STATE_PHRASES = {
  unconscious: 'No movement for over 15 seconds — may be unconscious.',  // highest risk: no motor response detected
  stunned:     'Showing limited movement — possibly stunned.',            // some movement but very slow/weak
  moving:      'Patient is moving — may be recovering.',                  // active movement detected post-fall
  unknown:     'Post-fall movement status unclear.',                      // model could not classify post-fall state
};


// ─────────────────────────────────────────────────────────────────────────────
// deriveSeverity(fall_type, post_state)
//
// PURPOSE:
//   Looks up the SEVERITY_MATRIX using the two most important signals from the
//   ML model — fall_type and post_state — and returns a severity level string.
//
// PARAMETERS:
//   fall_type  (string) — 'slip' | 'trip' | 'faint'
//                          Comes directly from the ML model's classification output.
//   post_state (string) — 'unconscious' | 'stunned' | 'moving' | 'unknown'
//                          Derived from how much movement was detected after the fall.
//
// RETURNS:
//   (string) — 'CRITICAL' | 'HIGH' | 'MEDIUM'
//
// HOW IT WORKS:
//   SEVERITY_MATRIX[fall_type]?.[post_state]
//     The ?. is "optional chaining" — if fall_type is not a key in SEVERITY_MATRIX
//     (e.g. the model returns an unexpected label), it returns undefined instead of crashing.
//
//   ?? 'HIGH'
//     The ?? is "nullish coalescing" — if the result is undefined (unknown combination),
//     we default to 'HIGH' instead of 'MEDIUM' or 'LOW'.
//     WHY HIGH as default? → In a safety-critical system like fall detection, it is
//     better to over-alert (caretaker checks and finds the patient is fine) than
//     to under-alert (caretaker misses a real emergency).
// ─────────────────────────────────────────────────────────────────────────────
function deriveSeverity(fall_type, post_state) {
  // Look up the row (fall_type) then the column (post_state) in the 2D matrix.
  // If either key is missing (unexpected model output), fall back to 'HIGH'.
  return SEVERITY_MATRIX[fall_type]?.[post_state] ?? 'HIGH';
}

function deriveSeverityWithImpactFallback(raw) {
  if (!USE_IMPACT_BASED_SEVERITY) {
    return deriveSeverity(raw.fall_type, raw.post_state);
  }

  const { severity_proxy, post_state, fall_type } = raw;
  if (severity_proxy == null || post_state == null || fall_type == null) {
    return deriveSeverity(fall_type, post_state);
  }

  return deriveSeverityFromImpactData({ severity_proxy, post_state, fall_type });
}


// ─────────────────────────────────────────────────────────────────────────────
// buildMessage(fall_type, pre_activity, post_state, confidence)
//
// PURPOSE:
//   Assembles a single, plain-English sentence that caretakers see on the
//   dashboard when an alert fires. It combines all three ML labels into one
//   readable narrative, then optionally appends a low-confidence warning.
//
// PARAMETERS:
//   fall_type    (string) — how the patient fell ('slip' | 'trip' | 'faint')
//   pre_activity (string) — what they were doing before ('walking' | 'standing' | ...)
//   post_state   (string) — what they are doing after  ('unconscious' | 'moving' | ...)
//   confidence   (number) — model's confidence score between 0.0 and 1.0
//
// RETURNS:
//   (string) — a complete sentence, e.g.:
//   "Patient tripped while walking. No movement for over 15 seconds — may be unconscious."
//   "Patient slipped while bending down. Patient is moving — may be recovering. (Low confidence — verify manually.)"
//
// HOW IT WORKS (step by step):
//
//   Step 1 — Look up each phrase from the phrase maps using the ML labels.
//             If a label is unrecognised (??), use a safe fallback string so
//             the message is never blank or broken.
//
//   Step 2 — Combine into one sentence:
//             "[fallPhrase] [activityPhrase]. [statePhrase]"
//
//   Step 3 — Confidence check:
//             If the model's confidence is below 0.70 (70%), we cannot fully
//             trust the classification. We append a visible warning so the
//             caretaker knows to physically verify rather than rely on the alert alone.
//             The 0.70 threshold was chosen as the minimum acceptable confidence
//             for acting on an automated alert without additional verification.
// ─────────────────────────────────────────────────────────────────────────────
function buildMessage(fall_type, pre_activity, post_state, confidence, location) {
  // Step 1: Resolve each ML label to its human-readable phrase.
  // ?? provides a safe fallback if the model returns an unexpected label.
  const fallPhrase     = FALL_TYPE_PHRASES[fall_type]       ?? 'Patient fell';       // fallback if unknown fall_type
  const activityPhrase = PRE_ACTIVITY_PHRASES[pre_activity] ?? '';                   // fallback: blank (activity phrase is optional context)
  const statePhrase    = POST_STATE_PHRASES[post_state]     ?? 'Status unknown.';    // fallback if unknown post_state
  const locationPhrase = location && location !== 'location_unknown' ? ` in the ${location}` : '';

  // Step 2: Combine the three phrases into one sentence.
  // Template: "[how they fell] [what they were doing]. [what they're doing now]"
  // Example:  "Patient tripped while walking. No movement for over 15 seconds — may be unconscious."
  let message = `${fallPhrase}${locationPhrase} ${activityPhrase}. ${statePhrase}`;

  // Step 3: If the model's confidence is below 70%, append a manual-verify warning.
  // This protects against false positives causing unnecessary panic, and false
  // negatives where a real fall gets low confidence and the caretaker dismisses it.
  if (confidence < 0.70) {
    message += ' (Low confidence — verify manually.)';
  }

  return message;
}


// ─────────────────────────────────────────────────────────────────────────────
// buildAlert(raw)   ← THE ONLY EXPORTED FUNCTION IN THIS FILE
//
// PURPOSE:
//   This is the single entry point that the rest of the backend calls.
//   It takes the raw output object from the ML model and returns a complete,
//   structured alert object that is:
//     - Saved to the 'alerts' table in the database (via the DB layer)
//     - Sent to the caretaker dashboard in real time (via WebSocket / polling)
//
// PARAMETER:
//   raw (object) — the direct output from Team A's ML model, expected shape:
//     {
//       fall_type:    string,  // 'slip' | 'trip' | 'faint'
//       pre_activity: string,  // 'walking' | 'standing' | 'bending' | 'sitting'
//       post_state:   string,  // 'unconscious' | 'stunned' | 'moving' | 'unknown'
//       confidence:   number,  // 0.0 – 1.0 (how certain the model is)
//     }
//
// RETURNS:
//   (object) — a fully structured alert:
//     {
//       timestamp:    string,  // ISO 8601 datetime of when the alert was generated
//       fall_type:    string,  // passed through from raw (for DB storage + filtering)
//       pre_activity: string,  // passed through from raw (for incident report context)
//       post_state:   string,  // passed through from raw (for DB storage + filtering)
//       severity:     string,  // 'CRITICAL' | 'HIGH' | 'MEDIUM' — derived by SEVERITY_MATRIX
//       message:      string,  // plain-English sentence shown to caretaker on dashboard
//       confidence:   number,  // passed through from raw (shown in dashboard detail view)
//     }
//
// HOW IT WORKS:
//   1. Destructure the four ML output fields from the raw object.
//   2. Call deriveSeverity() → looks up SEVERITY_MATRIX → gives urgency level.
//   3. Call buildMessage()   → assembles readable sentence + confidence warning.
//   4. Stamp the current time as an ISO timestamp (used for sorting alerts in DB).
//   5. Return everything as one flat object.
// ─────────────────────────────────────────────────────────────────────────────
export function buildAlert(raw) {
  // Step 1: Pull out the four ML output fields plus the optional confirmation field.
  const { fall_type, pre_activity, post_state, confidence, confirmation_window_ms } = raw;
  const location = raw.location ?? 'location_unknown';

  // Step 2: Determine how urgent this alert is using the severity matrix.
  // This drives the colour coding on the dashboard (red = CRITICAL, orange = HIGH, yellow = MEDIUM).
  // The new impact-based path is opt-in and defaults to the existing behavior.
  const severity = deriveSeverityWithImpactFallback({ fall_type, post_state, severity_proxy: raw.severity_proxy });

  // Step 3: Build the human-readable message that the caretaker will read.
  const message = buildMessage(fall_type, pre_activity, post_state, confidence, location);

  // Step 4: Record when this alert was created.
  // ISO format (e.g. "2026-06-09T14:32:01.000Z") ensures consistent sorting
  // in the database and correct display across different time zones on the dashboard.
  const timestamp = new Date().toISOString();

  // Step 5: Return the complete alert object.
  // All fields are stored in the DB. severity + message + timestamp are the three
  // fields that the caretaker dashboard prominently displays in the alerts panel.
  const alert = {
    timestamp,    // when it happened
    fall_type,    // how they fell — stored for filtering/stats in dashboard
    pre_activity, // what they were doing — stored for incident report generation
    post_state,   // condition after fall — stored for filtering/stats
    location,     // room detected by the patrol simulation, or location_unknown
    severity,     // CRITICAL / HIGH / MEDIUM — drives UI colour + sort order
    message,      // the sentence shown to the caretaker
    confidence,   // shown in the detail view so caretaker can judge reliability
  };

  // confirmation_window_ms is only present on alerts that passed through the
  // adaptive confirmation window (null/undefined for alerts from mock_emitter
  // or older clients). We include it only when explicitly provided so the
  // existing alert shape is unchanged for callers that don't send it.
  if (confirmation_window_ms != null) {
    alert.confirmation_window_ms = confirmation_window_ms;
  }

  return alert;
}
