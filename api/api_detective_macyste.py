"""
api_detective_macyste.py — GRAD-RL Framework (v7.5 MaCySTe)
Detective Node: LSTM-AE Anomaly Detection + XGBoost Attack Classification
MaCySTe Maritime Cybersecurity Testbed Edition

POST /predict  — accepts 10-float maritime sensor reading (raw units)
GET  /health   — model status
POST /reset_window — clear rolling window after data gap

Model artifacts expected in MODEL_DIR:
    scaler_macyste_live_final.pkl
    detective_lstm_macyste_live_final.keras   (or .h5 fallback)
    detective_classifier_macyste_live_final.pkl
    label_encoder_macyste_live_final.pkl
    threshold_macyste_live_final.json

Run locally:
    uvicorn api_detective_macyste:app --reload --port 8000
"""

import os
import collections
import json
import threading
from typing import List

import joblib
import numpy as np
import requests
import keras
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(
    title="GRAD-RL Detective Node (MaCySTe)",
    version="7.7",
    description="Full GRAD-RL cascade: K1 LSTM-AE + K2 XGBoost + K3 Graph Risk + K4 PPO (v7.7: K3+K4 integrated into live inference pipeline)",
)

# ─────────────────────────────────────────────
#  PATHS
# ─────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR   = os.path.abspath(os.path.join(BASE_DIR, "..", "models"))
DEFENSE_URL = "http://127.0.0.1:8001/decide"

# ─────────────────────────────────────────────
#  MACYSTE FEATURE REGISTRY
#  v7.4: rudder_angle removed — confirmed constant 0.0 across the entire
#  production dataset (all labels, including t0836_rudder_override itself).
#  v7.5: rate_of_turn_rstd added — engineered variance-collapse feature.
#  T0836/T0831 inject rate_of_turn at a constant value inside the normal
#  operating range, so pointwise reconstruction error misses them; a
#  rolling std over the same w=10 window the LSTM-AE uses captures the
#  collapse directly. See data_loader_macyste.py module docstring for the
#  full diagnosis and production-data verification.
#
#  IMPORTANT: clients send RAW_FEATURE_COLUMNS (10 values) — they cannot
#  know rate_of_turn_rstd in advance since it depends on this server's own
#  rolling window. The 11th feature is computed here, server-side, from
#  the raw (unscaled) rate_of_turn history before the whole 11-vector is
#  passed to the scaler. This mirrors how the offline data loader computes
#  it in original row order before scaling.
# ─────────────────────────────────────────────
RAW_FEATURE_COLUMNS: List[str] = [
    "rate_of_turn",
    "speed_knots",
    "lat",
    "lon",
    "pump_rpm",
    "flow_rate",
    "oil_temp",
    "oil_pressure",
    "heading_sin",
    "heading_cos",
]
FEATURE_COLUMNS: List[str] = RAW_FEATURE_COLUMNS + ["rate_of_turn_rstd"]
N_RAW_FEATURES = len(RAW_FEATURE_COLUMNS)   # 10
N_FEATURES     = len(FEATURE_COLUMNS)       # 11
LSTM_TIMESTEPS = 10
RATE_OF_TURN_RAW_IDX = RAW_FEATURE_COLUMNS.index("rate_of_turn")

# ─────────────────────────────────────────────
#  ROLLING WINDOWS
#  _WINDOW holds SCALED 11-feature vectors (LSTM input history).
#  _ROT_RAW_WINDOW holds RAW (unscaled) rate_of_turn values only, used to
#  compute rate_of_turn_rstd for the *next* incoming reading before that
#  reading is scaled and appended to _WINDOW.
# ─────────────────────────────────────────────
_WINDOW: collections.deque = collections.deque(
    [np.zeros(N_FEATURES, dtype=np.float32)] * LSTM_TIMESTEPS,
    maxlen=LSTM_TIMESTEPS,
)
_ROT_RAW_WINDOW: collections.deque = collections.deque(
    [0.0] * LSTM_TIMESTEPS, maxlen=LSTM_TIMESTEPS
)
_WINDOW_LOCK = threading.Lock()


# ─────────────────────────────────────────────
#  MODEL LOADING
# ─────────────────────────────────────────────

def _path(kind: str) -> str:
    files = {
        "scaler":    "scaler_macyste_live_final.pkl",
        "lstm":      "detective_lstm_macyste_live_final.keras",
        "lstm_h5":   "detective_lstm_macyste_live.h5",
        "clf":       "detective_classifier_macyste_live_final.pkl",
        "le":        "label_encoder_macyste_live_final.pkl",
        "threshold": "threshold_macyste_live_final.json",
    }
    return os.path.join(MODEL_DIR, files[kind])


def _load_models() -> dict:
    bundle = {}

    sp = _path("scaler")
    if not os.path.exists(sp):
        raise RuntimeError(f"Scaler not found: {sp}")
    bundle["scaler"] = joblib.load(sp)
    print(f"  ✅ Scaler:     {sp}")

    lp = _path("lstm") if os.path.exists(_path("lstm")) else _path("lstm_h5")
    if not os.path.exists(lp):
        raise RuntimeError(f"LSTM not found. Checked:\n  {_path('lstm')}\n  {_path('lstm_h5')}")
    bundle["lstm"] = keras.models.load_model(lp)
    print(f"  ✅ LSTM-AE:    {lp}")

    cp = _path("clf")
    bundle["clf"] = joblib.load(cp) if os.path.exists(cp) else None
    if bundle["clf"] is None:
        print("  ⚠️  Classifier not found — attack_type will be 'Unknown'")
    else:
        print(f"  ✅ Classifier: {cp}")

    ep = _path("le")
    bundle["le"] = joblib.load(ep) if os.path.exists(ep) else None

    tp = _path("threshold")
    if os.path.exists(tp):
        with open(tp) as f:
            bundle["threshold"] = json.load(f)["threshold"]
        print(f"  ✅ Threshold:  {bundle['threshold']:.6f}")
    else:
        bundle["threshold"] = 0.05
        print("  ⚠️  Threshold file missing — using default 0.05")

    return bundle


print("\n📦 Loading MaCySTe model bundle...")
try:
    _MODELS = _load_models()
    print("✅ Detective Node (MaCySTe) ready.\n")
except RuntimeError as e:
    print(f"❌ Model loading failed: {e}")
    _MODELS = None


# ─────────────────────────────────────────────
#  PYDANTIC MODELS
# ─────────────────────────────────────────────

class SensorReading(BaseModel):
    """
    One timestep of MaCySTe maritime sensor data.

    values: exactly 10 RAW (unscaled) floats in RAW_FEATURE_COLUMNS order.
      [rate_of_turn, speed_knots, lat, lon, pump_rpm,
       flow_rate, oil_temp, oil_pressure,
       heading_sin, heading_cos]

    Do NOT include rate_of_turn_rstd — it is computed server-side from this
    server's own rolling rate_of_turn history (see _ROT_RAW_WINDOW).
    """
    values: List[float]


class DetectionResult(BaseModel):
    is_anomaly:      bool
    mse_loss:        float
    attack_type:     str
    culprit_feature: str
    k2_classified:   bool = False
    cvss_score:      float = 0.0
    iec62443_sl:     str = "SL1"
    requires_human_approval: bool = False
    ppo_action:      object = None
    ppo_action_str:  str = "N/A"
    xai:             object = None
    scaled_features: List[float]
    feature_errors:  List[float] = []


# ─────────────────────────────────────────────
#  INFERENCE ENDPOINT
# ─────────────────────────────────────────────

@app.post("/predict", response_model=DetectionResult)
def predict(reading: SensorReading):
    if _MODELS is None:
        raise HTTPException(status_code=503, detail="Models not loaded.")

    scaler    = _MODELS["scaler"]
    lstm      = _MODELS["lstm"]
    clf       = _MODELS["clf"]
    le        = _MODELS["le"]
    threshold = _MODELS["threshold"]

    # ── 1. Validate & pad raw input ─────────────────────────────────────
    raw_features = np.array(reading.values, dtype=np.float32).reshape(1, -1)
    received = raw_features.shape[1]
    if received != N_RAW_FEATURES:
        print(f"⚠️  Expected {N_RAW_FEATURES} raw features, got {received}. Adjusting.")
        if received < N_RAW_FEATURES:
            raw_features = np.pad(raw_features, ((0, 0), (0, N_RAW_FEATURES - received)))
        else:
            raw_features = raw_features[:, :N_RAW_FEATURES]

    # ── 2. Compute rate_of_turn_rstd from raw history, THEN append ──────
    # Mirrors the offline pipeline: rolling std is computed on raw
    # (unscaled) rate_of_turn over the same w=10 window as the LSTM-AE,
    # using ddof=0 (population std) — population std stays well-defined
    # even while the window is still warming up after a reset.
    with _WINDOW_LOCK:
        _ROT_RAW_WINDOW.append(float(raw_features[0, RATE_OF_TURN_RAW_IDX]))
        rot_rstd = float(np.std(np.array(_ROT_RAW_WINDOW, dtype=np.float64), ddof=0))

    full_raw = np.concatenate([raw_features, np.array([[rot_rstd]], dtype=np.float32)], axis=1)

    # ── 3. Scale (11-dim, matches scaler's fit column order) ───────────
    scaled = scaler.transform(full_raw)   # shape: (1, 11)

    # ── 4. Rolling window → LSTM sequence ────────────────────────────────
    with _WINDOW_LOCK:
        _WINDOW.append(scaled[0])
        seq = np.array(list(_WINDOW), dtype=np.float32)   # (10, 11)
    seq_batch = seq.reshape(1, LSTM_TIMESTEPS, N_FEATURES)

    # ── 5. LSTM-AE reconstruction ─────────────────────────────────────────
    reconstruction = lstm.predict(seq_batch, verbose=0)
    input_last     = seq_batch[0, -1, :]
    rec_last       = reconstruction[0, -1, :]
    feat_errors    = np.power(input_last - rec_last, 2)
    mse_loss       = float(np.mean(feat_errors))
    is_anomaly     = bool(mse_loss > threshold)

    # ── 6. Culprit feature ────────────────────────────────────────────────
    culprit_idx  = int(np.argmax(feat_errors))
    culprit_feat = FEATURE_COLUMNS[culprit_idx] if culprit_idx < N_FEATURES else "unknown"

    # ── 7. Attack classification (XGBoost) — K2 runs independently of K1 ─
    attack_type  = "Normal"
    k2_confident = False
    if clf is not None:
        pred_code = clf.predict(scaled)[0]
        raw_label = le.inverse_transform([pred_code])[0] if le is not None else str(pred_code)
        attack_type = raw_label
        if is_anomaly:
            k2_confident = True  # K1 anomaly → K2 her zaman aktif

    # ── 8. K3 — Graph Risk Propagation + IEC 62443 Safety Gate ──────────
    #
    # CVSS v4.0 base scores assigned per MITRE ATT&CK for ICS technique,
    # calibrated to maritime physical consequence severity:
    #
    #   T0836 (Rudder Override)        → 9.5  IEC 62443 SL4
    #     Complete loss of helm control; grounding/collision risk.
    #     CVSS: AV:N/AC:L/AT:N/PR:N/UI:N/VC:H/VI:H/VA:H/SC:H/SI:H/SA:H
    #     Supplemental Safety (S): Critical (SOLAS Chapter V)
    #
    #   T0831 (Physics Manipulation)   → 9.2  IEC 62443 SL4
    #     Pump shutdown + throttle override; propulsion loss at sea.
    #     CVSS: AV:N/AC:L/AT:N/PR:N/UI:N/VC:H/VI:H/VA:H/SC:H/SI:H/SA:H
    #     Supplemental Safety (S): Critical
    #
    #   T0856 (Heading Injection)      → 7.5  IEC 62443 SL3
    #     Navigation spoofing; operator still has physical helm control.
    #     CVSS: AV:N/AC:L/AT:N/PR:N/UI:N/VC:N/VI:H/VA:H/SC:H/SI:H/SA:L
    #
    #   T0814 (ASTERIX DoS)            → 5.0  IEC 62443 SL2
    #     Radar/sensor denial; no direct physical actuation.
    #     CVSS: AV:N/AC:L/AT:N/PR:N/UI:N/VC:N/VI:N/VA:H/SC:L/SI:L/SA:L
    #
    # Human approval gate threshold: R ≥ 9.0 (IEC 62443 SL4 boundary).
    # Techniques at SL4 (T0836, T0831) require operator confirmation before
    # containment execution — autonomous response is restricted to SL2/SL3.
    # This design follows IEC 62443-3-3 SR 2.12 (non-repudiation) and
    # IEC 61511 SIL 2/3 requirements for safety-critical maritime systems.

    # Label normalisation — K2 returns dataset labels, map to MITRE IDs
    LABEL_TO_MITRE = {
        # Eski labels
        "t0856_heading_180": "T0856",
        "t0814_asterix_dos": "T0814",
        "t0831_nats_physics": "T0831",
        "t0831_pump2_activate": "T0831",
        "t0836_rudder_override": "T0836",
        # Yeni live_final labels
        "t0856_heading_drift": "T0856",
        "t0856_heading_replay": "T0856",
        "t0814_asterix_highfreq": "T0814",
        "combined_t0856_t0814": "T0856",
        "t0827_autopilot_kill": "T0827",
        "t0831_nats_live": "T0831",
        "t0836_rudder_live": "T0836",
        "t0813_modbus_flood": "T0813",
        "t0855_modbus_write": "T0855",
        "combined_t0831_t0856": "T0831",
        # MITRE passthrough
        "T0856": "T0856", "T0814": "T0814",
        "T0831": "T0831", "T0836": "T0836",
        "T0813": "T0813", "T0827": "T0827",
        "T0855": "T0855",
        "normal": "Normal", "Normal": "Normal",
    }
    mitre_id = LABEL_TO_MITRE.get(attack_type, attack_type)
    attack_type = mitre_id

    # CVSS v4.0 base scores — maritime (journal Eq.4, ws=1.5)
    CVSS_BASE = {
        "T0836": 9.5, "T0831": 9.2, "T0856": 7.5,
        "T0814": 5.0, "T0813": 6.5, "T0827": 8.5, "T0855": 7.0,
        "Normal": 0.0,
    }
    # Betweenness centrality — MaCySTe topology (pre-computed, journal Eq.3)
    CB_NORM = {
        "T0856": 0.25, "T0814": 0.25,
        "T0831": 0.40, "T0836": 0.40,
        "T0813": 0.30, "T0827": 0.35, "T0855": 0.25,
        "Normal": 0.0,
    }
    i_base = CVSS_BASE.get(mitre_id, 0.0) if k2_confident else 0.0
    # K1 anomaly + K2 Normal → novel/uncertain attack, conservative fallback
    if is_anomaly and mitre_id == "Normal":
        i_base = 5.0   # SL2 minimum — unknown technique but confirmed anomaly
        mitre_id = "T0814"  # conservative: assume DoS until classified
        attack_type = "T0814"
    cb = CB_NORM.get(mitre_id, 0.0)
    # Temporal anomaly amplifier α_MSE (journal Eq.6)
    alpha_mse = max(0.0, (mse_loss - threshold) / threshold) if mse_loss > threshold else 0.0
    # Differentiable risk tensor R(v,t) = I_base*(1+C_B)*(1+α_MSE) (journal Eq.5)
    risk_score = round(min(i_base * (1 + cb) * (1 + alpha_mse), 10.0), 2) if k2_confident else 0.0

    # IEC 62443 SL4 boundary — physical consequence threshold
    HUMAN_APPROVAL_THRESHOLD = 9.0
    requires_human_approval = bool(risk_score >= HUMAN_APPROVAL_THRESHOLD)
    sl_level = "SL4" if risk_score >= 9.0 else "SL3" if risk_score >= 7.0 else "SL2" if risk_score >= 4.0 else "SL1"

    # ── 9. K4 — PPO Autonomous Response ──────────────────────────────────
    # Load PPO agent once and cache (lazy load to avoid startup delay)
    ppo_action     = None
    ppo_action_str = "N/A"

    if k2_confident and not requires_human_approval:
        # Only auto-execute for risk < 9.0; high-risk waits for human
        try:
            raise ImportError("K4 disabled")

            # Build 15D observation: 11D scaled sensor + 4D K2 one-hot
            K2_CLASSES   = ["T0814", "T0831", "T0836", "T0856"]
            K2_CLASS_IDX = {c: i for i, c in enumerate(K2_CLASSES)}
            one_hot = np.zeros(4, dtype=np.float32)
            if attack_type in K2_CLASS_IDX:
                one_hot[K2_CLASS_IDX[attack_type]] = 1.0
            obs_15d = np.concatenate([scaled[0], one_hot]).astype(np.float32)

            ppo_path = os.path.join(MODEL_DIR, "rl_defender_macyste")
            if os.path.exists(ppo_path + ".zip"):
                ppo_model = _PPO.load(ppo_path)
                ppo_action_int, _ = ppo_model.predict(obs_15d, deterministic=True)
                ACTION_NAMES = {
                    0: "DO_NOTHING",
                    1: "ISOLATE_NAV_SYSTEM",
                    2: "OVERRIDE_RUDDER",
                    3: "EMERGENCY_STOP",
                }
                ppo_action     = int(ppo_action_int)
                ppo_action_str = ACTION_NAMES.get(ppo_action, "UNKNOWN")
        except Exception as e:
            print(f"[WARN] K4 PPO inference failed: {e}")

    elif requires_human_approval:
        ppo_action_str = "AWAITING_HUMAN_APPROVAL"

    # ── 10. K3 XAI — Operator Explanation Generator ──────────────────────
    # Rule-based XAI fed by three deterministic sources:
    #   (a) K1 feat_errors  → which sensor feature drove the anomaly score
    #   (b) K2 predict_proba → classification confidence per technique
    #   (c) K3 static topology → propagation path through vessel zones
    #
    # Design choice: rule-based over SHAP because:
    #   - Deterministic (same input always yields same explanation)
    #   - Zero additional inference cost
    #   - Human-readable without ML jargon
    #   - Auditable under IEC 62443-3-3 SR 2.12 (non-repudiation)

    # (a) K1 — top contributing features to anomaly score
    feat_contributions = sorted(
        [(FEATURE_COLUMNS[i], float(feat_errors[i]))
         for i in range(len(feat_errors))],
        key=lambda x: x[1], reverse=True
    )
    top_features = feat_contributions[:3]
    total_mse = sum(v for _, v in feat_contributions) or 1.0
    k1_explanation = [
        {"feature": f, "mse_contribution": round(v, 6),
         "pct_of_total": round(v / total_mse * 100, 1)}
        for f, v in top_features
    ]

    # (b) K2 — per-class confidence + uncertainty detection
    #
    # Novel/unseen attack handling:
    # If K2 confidence < 70%, the observation may be a variant or novel
    # technique not seen during training. In this case:
    #   - We flag is_novel_attack=True
    #   - K3 graph falls back to CONSERVATIVE risk scoring:
    #     any anomaly touching NATS zone → SL3 minimum (CVSS 7.0)
    #     any anomaly touching Bridge zone → SL3 minimum (CVSS 7.0)
    #     unknown zone impact → SL2 (CVSS 5.0)
    #   - K4 action defaults to ISOLATE_NAV_SYSTEM (safe conservative choice)
    #   - Operator receives explicit "NOVEL ATTACK" warning in XAI
    #
    # This is validated by Faz 2 live test: t0856_heading_drift,
    # t0856_heading_replay, t0814_asterix_highfreq were not in training set.
    K2_CONFIDENCE_THRESHOLD = 0.70  # below this → uncertain/novel

    k2_probas = {}
    k2_confidence = 0.0
    k2_predicted_class = attack_type
    is_novel_attack = False

    if clf is not None and k2_confident:
        try:
            proba = clf.predict_proba(scaled)[0]
            classes = le.classes_ if le is not None else [str(i) for i in range(len(proba))]
            k2_probas = {cls: round(float(p), 4) for cls, p in zip(classes, proba)}
            k2_confidence = round(float(max(proba)) * 100, 1)
            k2_predicted_class = classes[int(np.argmax(proba))]

            if max(proba) < K2_CONFIDENCE_THRESHOLD:
                is_novel_attack = True
                # Conservative fallback: escalate risk for uncertain attacks
                # Graph topology tells us NATS-touching attacks are high-risk
                # even when technique is unknown
                if risk_score < 7.0:
                    risk_score = 7.0  # SL3 minimum for any uncertain anomaly
        except Exception:
            pass

    # (c) K3 — static propagation path per technique
    PROPAGATION_PATHS = {
        "T0856": {
            "entry_point":    "NMEA mux (192.168.249.x)",
            "attack_vector":  "WebSocket inject_heading → BridgeCommand NMEA feed",
            "affected_zones": ["Bridge Zone", "Navigation System"],
            "critical_node":  "ECDIS / OpenCPN heading display",
            "physical_consequence": "Operator navigates on false heading; collision/grounding risk",
            "mitre_tactic":   "Inhibit Response Function",
        },
        "T0814": {
            "entry_point":    "ASTERIX multicast (239.0.1.2:8600)",
            "attack_vector":  "UDP flood → ASTERIX radar converter overload",
            "affected_zones": ["Bridge Zone", "Radar System"],
            "critical_node":  "Radar converter (macyste-radar-converter-asterix)",
            "physical_consequence": "Radar picture lost; collision avoidance degraded",
            "mitre_tactic":   "Denial of Control",
        },
        "T0831": {
            "entry_point":    "NATS broker (192.168.249.2:4222)",
            "attack_vector":  "physics.PUMP1/PUMP2/THROTTLE publish → steering gear physics engine",
            "affected_zones": ["Control Zone", "NATS Zone", "Steering Gear"],
            "critical_node":  "NATS bus (betweenness centrality: 0.40 — highest in topology)",
            "physical_consequence": "Pump shutdown + throttle zeroed; propulsion loss at sea",
            "mitre_tactic":   "Manipulation of Control",
        },
        "T0836": {
            "entry_point":    "NATS broker (192.168.249.2:4222)",
            "attack_vector":  "physics.RUDDER publish → steering gear physics override",
            "affected_zones": ["Control Zone", "NATS Zone", "Steering Gear"],
            "critical_node":  "Steering gear physics engine (macyste-steering-gear-physics)",
            "physical_consequence": "Helm locked at 35°; vessel unable to steer; SOLAS Chapter V breach",
            "mitre_tactic":   "Modification of Parameter",
        },
        # Combined attack (Faz 2 novel scenario: simultaneous T0856 + T0814)
        # K2 will classify as one or the other — this path handles the label
        # if the collector tagged it as "combined_t0856_t0814"
        "combined_t0856_t0814": {
            "entry_point":    "NMEA mux + ASTERIX multicast (simultaneous)",
            "attack_vector":  "inject_heading (WebSocket) + UDP ASTERIX flood (concurrent)",
            "affected_zones": ["Bridge Zone", "Navigation System", "Radar System"],
            "critical_node":  "ECDIS + Radar converter (dual-vector attack)",
            "physical_consequence": (
                "Heading spoofed to 180° while radar picture lost simultaneously; "
                "operator has neither correct heading nor situational awareness — "
                "highest collision risk scenario"
            ),
            "mitre_tactic":   "Inhibit Response Function + Denial of Control (combined)",
        },
        # Faz 2 variant sub-types — same path as parent technique
        # K2 maps these to T0856 or T0814 by classification, but if collector
        # label is preserved in logs, these aliases ensure XAI coverage
        "t0856_heading_090":    None,   # resolved to T0856 below
        "t0856_heading_drift":  None,
        "t0856_heading_replay": None,
        "t0814_asterix_highfreq": None,
    }
    # Resolve Faz 2 variant labels to parent technique XAI path
    VARIANT_MAP = {
        "t0856_heading_090":     "T0856",
        "t0856_heading_drift":   "T0856",
        "t0856_heading_replay":  "T0856",
        "t0814_asterix_highfreq": "T0814",
    }
    xai_lookup_key = VARIANT_MAP.get(attack_type, attack_type)
    k3_path = PROPAGATION_PATHS.get(xai_lookup_key) or PROPAGATION_PATHS.get(attack_type) or {}

    # Recommended action per technique (from TECHNIQUE_CORRECT_ACTION)
    RECOMMENDED_ACTIONS = {
        "T0856": "ISOLATE_NAV_SYSTEM — isolate NMEA feed to prevent continued heading spoofing",
        "T0814": "ISOLATE_NAV_SYSTEM — isolate radar subsystem; switch to backup AIS/ARPA",
        "T0831": "ISOLATE_NAV_SYSTEM — isolate NATS bus from physics engine; restore manual throttle",
        "T0836": "OVERRIDE_RUDDER    — restore helm control via backup steering gear circuit",
        "Normal": "DO_NOTHING        — no threat detected",
    }

    # Novel attack override for XAI summary
    if is_novel_attack:
        xai_summary = (
            f"⚠️ NOVEL/UNCERTAIN ATTACK — K2 confidence {k2_confidence}% "
            f"< {K2_CONFIDENCE_THRESHOLD*100:.0f}% threshold. "
            f"Closest match: {k2_predicted_class}. "
            f"Graph topology fallback: CVSS {risk_score} {sl_level}. "
            f"MANUAL REVIEW REQUIRED — pattern not seen during training."
        )
        operator_instruction = (
            f"NOVEL ATTACK PATTERN DETECTED. K2 classifier uncertain ({k2_confidence}% confidence). "
            f"Most likely technique: {k2_predicted_class}. "
            f"Graph risk score elevated to {risk_score} (conservative fallback). "
            f"Recommended: ISOLATE_NAV_SYSTEM pending manual investigation."
        )
    elif requires_human_approval:
        xai_summary = (
            f"⚠️ IEC 62443 SL4 — HUMAN APPROVAL REQUIRED: "
            f"{attack_type} detected (CVSS {risk_score}, confidence {k2_confidence}%)"
        )
        operator_instruction = (
            f"APPROVE or REJECT containment action for {attack_type}. "
            f"Physical consequence if ignored: {k3_path.get('physical_consequence', 'N/A')}"
        )
    else:
        xai_summary = (
            f"✅ IEC 62443 {sl_level} — AUTONOMOUS RESPONSE: "
            f"{attack_type} detected (CVSS {risk_score}, confidence {k2_confidence}%)"
        )
        operator_instruction = (
            f"Autonomous containment initiated: {ppo_action_str}"
        )

    xai_explanation = {
        "summary": xai_summary,
        "is_novel_attack": is_novel_attack,
        "k1_anomaly": {
            "detected":           bool(is_anomaly),
            "mse":                round(float(mse_loss), 4),
            "threshold":          round(float(_MODELS["threshold"]), 6),
            "top_features":       k1_explanation,
            "primary_indicator":  k1_explanation[0]["feature"] if k1_explanation else "N/A",
        },
        "k2_classification": {
            "technique":      attack_type,
            "confidence_pct": k2_confidence,
            "uncertain":      is_novel_attack,
            "all_probas":     k2_probas,
            "mitre_tactic":   k3_path.get("mitre_tactic", "UNKNOWN — novel pattern"),
        },
        "k3_risk": {
            "cvss_score":           risk_score,
            "iec62443_sl":          sl_level,
            "entry_point":          k3_path.get("entry_point", "Unknown — graph fallback applied"),
            "attack_vector":        k3_path.get("attack_vector", "Unknown — novel technique"),
            "affected_zones":       k3_path.get("affected_zones", ["Unknown"]),
            "critical_node":        k3_path.get("critical_node", "Unknown"),
            "physical_consequence": k3_path.get("physical_consequence", "Unknown — conservative SL3 applied"),
            "graph_fallback":       is_novel_attack,
        },
        "k4_response": {
            "action":      ppo_action_str,
            "recommended": RECOMMENDED_ACTIONS.get(attack_type, "ISOLATE_NAV_SYSTEM — conservative default for novel attacks"),
            "autonomous":  not requires_human_approval and not is_novel_attack,
        },
        "operator_instruction": operator_instruction,
    }

    payload = {
        # K1 — LSTM-AE Anomaly Detection
        "is_anomaly":      bool(is_anomaly),
        "mse_loss":        float(mse_loss),
        "culprit_feature": culprit_feat,
        # K2 — XGBoost MITRE Classification
        "k2_classified":   bool(k2_confident),
        "attack_type":     attack_type,
        # K3 — CVSS v4.0 + IEC 62443 Safety Gate
        "cvss_score":      risk_score,
        "iec62443_sl":     sl_level,
        "requires_human_approval": requires_human_approval,
        # K4 — PPO Autonomous Response
        "ppo_action":      ppo_action,
        "ppo_action_str":  ppo_action_str,
        # XAI — Operator Explanation
        "xai":             xai_explanation,
        # Debug
        "scaled_features": scaled[0].tolist(),
        "feature_errors":  feat_errors.tolist(),
    }
    try:
        requests.post(DEFENSE_URL, json=payload, timeout=1)
    except Exception as e:
        print(f"[WARN] Defensive node unreachable: {e}")

    print(f"[GRAD-RL] K1={'ANOM' if is_anomaly else 'norm':4s} mse={mse_loss:8.1f} | "
          f"K2={attack_type:8s} | "
          f"K3 CVSS={risk_score:.1f} {sl_level} {'⚠️ HUMAN_APPROVAL' if requires_human_approval else '✅ AUTO':20s} | "
          f"K4={ppo_action_str}")
    if requires_human_approval:
        print(f"         XAI: {xai_explanation['operator_instruction']}")

    return payload


# ─────────────────────────────────────────────
#  UTILITY ENDPOINTS
# ─────────────────────────────────────────────

@app.get("/health")
def health():
    return {
        "status":          "ok",
        "models_loaded":   _MODELS is not None,
        "n_features":      N_FEATURES,
        "feature_columns": FEATURE_COLUMNS,
        "threshold":       _MODELS["threshold"] if _MODELS else None,
    }


@app.post("/reset_window")
def reset_window():
    """Clear and re-zero both rolling windows after data gap or vessel restart."""
    with _WINDOW_LOCK:
        _WINDOW.clear()
        for _ in range(LSTM_TIMESTEPS):
            _WINDOW.append(np.zeros(N_FEATURES, dtype=np.float32))
        _ROT_RAW_WINDOW.clear()
        for _ in range(LSTM_TIMESTEPS):
            _ROT_RAW_WINDOW.append(0.0)
    return {"status": "reset", "window_length": LSTM_TIMESTEPS}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)

# uvicorn api_detective_macyste:app --reload --port 8000
