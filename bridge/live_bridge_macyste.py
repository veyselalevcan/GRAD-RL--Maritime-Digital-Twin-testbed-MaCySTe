#!/usr/bin/env python3
"""
live_bridge_macyste.py — GRAD-RL MaCySTe Live Inference Bridge
Veysel Alevcan, COPELABS, Lusofona University

Watches the live telemetry CSV and sends each new row to the GRAD-RL
inference API. Logs full K1-K2-K3-K4+XAI decisions to a separate JSONL
file (one JSON object per line), mirroring the telemetry_audit + xai_log
pattern used in the GRFICSv3 deployment.

Output files:
  CSV_FILE     → raw telemetry (written by macyste_collector_v2.py)
  XAI_LOG_FILE → K1/K2/K3/K4 decisions + XAI explanation (written here)
"""
import json
import math
import time
import requests
from pathlib import Path
from datetime import datetime, timezone

CSV_FILE     = "/root/MaCySTe/grad_rl_data/live_attacks_20260625.csv"
XAI_LOG_FILE = "/root/MaCySTe/grad_rl_data/live_xai_20260625.jsonl"
API_URL      = "http://127.0.0.1:8000/predict"
POLL_SEC     = 1.0

COLUMNS = ["timestamp", "heading", "rate_of_turn", "speed_knots", "lat", "lon",
           "pump_rpm", "flow_rate", "rudder_angle", "oil_temp", "oil_pressure", "label"]


def parse_row(row: dict, history: list):
    """Convert CSV row to API feature vector. Modifies history in-place."""
    try:
        heading      = float(row["heading"]      or 0)
        rot          = float(row["rate_of_turn"] or 0)
        speed        = float(row["speed_knots"]  or 0)
        lat          = float(row["lat"]          or 0)
        lon          = float(row["lon"]          or 0)
        pump_rpm     = float(row["pump_rpm"]     or 0)
        flow_rate    = float(row["flow_rate"]    or 0)
        oil_temp     = float(row["oil_temp"]     or 0)
        oil_pressure = float(row["oil_pressure"] or 0)

        heading_sin  = math.sin(math.radians(heading))
        heading_cos  = math.cos(math.radians(heading))

        # Rolling std of rate_of_turn (window=10)
        history.append(rot)
        if len(history) > 10:
            history.pop(0)
        if len(history) >= 2:
            mean = sum(history) / len(history)
            rot_rstd = (sum((x - mean)**2 for x in history) / len(history)) ** 0.5
        else:
            rot_rstd = 0.0

        return [
            rot, speed, lat, lon,
            pump_rpm, flow_rate,
            oil_temp, oil_pressure,
            heading_sin, heading_cos,
            rot_rstd
        ]
    except Exception:
        return None


def console_print(ts: str, label: str, result: dict):
    """Compact one-line console summary of the full cascade decision."""
    is_anomaly = result.get("is_anomaly", False)
    attack     = result.get("attack_type", "Normal")
    mse        = result.get("mse_loss", 0.0)
    cvss       = result.get("cvss_score", 0.0)
    sl         = result.get("iec62443_sl", "SL1")
    action     = result.get("ppo_action_str", "N/A")
    human      = result.get("requires_human_approval", False)
    novel      = result.get("xai", {}).get("is_novel_attack", False)

    flag = "⚠️ HUMAN" if human else ("🔍 NOVEL" if novel else ("🚨 ANOM" if is_anomaly else "✅ norm"))

    print(
        f"[{ts[11:19]}] {flag} "
        f"label={label:25s} "
        f"K2={attack:8s} "
        f"CVSS={cvss:.1f} {sl:3s} "
        f"K4={action:22s} "
        f"mse={mse:10.1f}",
        flush=True
    )

    # Extra line for human approval or novel attack
    xai = result.get("xai", {})
    if human or novel:
        print(f"         ↳ {xai.get('operator_instruction', '')[:120]}", flush=True)


last_line = 0
rot_history = []
print(f"[BRIDGE] Watching  : {CSV_FILE}")
print(f"[BRIDGE] XAI log   : {XAI_LOG_FILE}")
print(f"[BRIDGE] API       : {API_URL}")
print(f"[BRIDGE] Started   : {datetime.now(timezone.utc).isoformat()}")
print("-" * 90)

while True:
    try:
        lines = Path(CSV_FILE).read_text().splitlines()

        if len(lines) > last_line:
            for line in lines[last_line:]:
                if not line or line.startswith("timestamp"):
                    continue
                parts = line.split(",")
                if len(parts) < 11:
                    continue

                row    = dict(zip(COLUMNS, parts))
                values = parse_row(row, rot_history)
                if values is None:
                    continue

                ts    = row.get("timestamp", "")
                label = row.get("label", "?").strip()

                try:
                    resp   = requests.post(API_URL, json={"values": values}, timeout=30)
                    result = resp.json()
                except Exception as e:
                    print(f"[ERR] API call failed: {e}", flush=True)
                    continue

                # Console summary
                console_print(ts, label, result)

                # Write full decision to XAI JSONL log
                log_entry = {
                    "timestamp":        ts,
                    "ground_truth_label": label,
                    # K1
                    "k1_is_anomaly":    result.get("is_anomaly"),
                    "k1_mse":           result.get("mse_loss"),
                    "k1_culprit":       result.get("culprit_feature"),
                    # K2
                    "k2_attack_type":   result.get("attack_type"),
                    "k2_classified":    result.get("k2_classified"),
                    "k2_confidence":    result.get("xai", {}).get("k2_classification", {}).get("confidence_pct"),
                    "k2_all_probas":    result.get("xai", {}).get("k2_classification", {}).get("all_probas"),
                    "k2_uncertain":     result.get("xai", {}).get("k2_classification", {}).get("uncertain"),
                    # K3
                    "k3_cvss":          result.get("cvss_score"),
                    "k3_iec62443_sl":   result.get("iec62443_sl"),
                    "k3_human_approval": result.get("requires_human_approval"),
                    "k3_entry_point":   result.get("xai", {}).get("k3_risk", {}).get("entry_point"),
                    "k3_consequence":   result.get("xai", {}).get("k3_risk", {}).get("physical_consequence"),
                    # K4
                    "k4_action":        result.get("ppo_action"),
                    "k4_action_str":    result.get("ppo_action_str"),
                    "k4_recommended":   result.get("xai", {}).get("k4_response", {}).get("recommended"),
                    # XAI
                    "xai_summary":      result.get("xai", {}).get("summary"),
                    "xai_primary_feature": result.get("xai", {}).get("k1_anomaly", {}).get("primary_indicator"),
                    "xai_top_features": result.get("xai", {}).get("k1_anomaly", {}).get("top_features"),
                    "xai_novel_attack": result.get("xai", {}).get("is_novel_attack"),
                    "xai_operator_instruction": result.get("xai", {}).get("operator_instruction"),
                }

                with open(XAI_LOG_FILE, "a") as f:
                    f.write(json.dumps(log_entry) + "\n")

            last_line = len(lines)

    except FileNotFoundError:
        pass  # CSV not yet created — wait
    except Exception as e:
        print(f"[ERR] {e}", flush=True)

    time.sleep(POLL_SEC)
