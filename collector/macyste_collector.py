#!/usr/bin/env python3
"""
MaCySTe Data Collector — GRAD-RL Maritime Pipeline
Veysel Alevcan, COPELABS, Lusofona University

Collects NMEA (heading/speed/ROT) and Modbus (SGS registers) from OpenSearch,
merges into a single feature row per second, writes labeled CSV.

Usage:
    python3 macyste_collector.py --label normal --duration 2700 --output normal_baseline.csv
    python3 macyste_collector.py --label t0856_heading --duration 300 --output attack_data.csv --append
"""

import argparse
import csv
import json
import struct
import sys
import time
from datetime import datetime, timezone

import requests
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# --- Config ---
OS_BASE   = "https://192.168.249.13:9200"
OS_AUTH   = ("admin", "admin")
NMEA_IDX  = "nmea-2026-06-10"
MOD_IDX   = "modbus-2026-06-10"
POLL_SEC  = 1.0   # collection interval

# Modbus source IP (pump 1 PLC → SGS master, reg 100-109)
SGS_SRC_IP = "10.1.2.3"

# Register pairs (lo_reg, hi_reg) → decoded float name
# Live decode confirmed:
# reg100-101 = 499.7  → pump_rpm
# reg102-103 = 9.99   → flow_rate
# reg104-105 = 0.0    → rudder_angle
# reg106-107 = 41.17  → oil_temp (°C)
# reg108-109 = 50.52  → oil_pressure (bar)
REG_MAP = {
    (100, 101): "pump_rpm",
    (102, 103): "flow_rate",
    (104, 105): "rudder_angle",
    (106, 107): "oil_temp",
    (108, 109): "oil_pressure",
}

CSV_FIELDS = [
    "timestamp",
    "heading",           # HDT sentence
    "rate_of_turn",      # ROT sentence
    "speed_knots",       # VTG sentence
    "lat",               # RMC
    "lon",               # RMC
    "pump_rpm",          # Modbus reg 0,1
    "flow_rate",         # Modbus reg 2,3
    "rudder_angle",      # Modbus reg 4,5
    "oil_temp",          # Modbus reg 6,7
    "oil_pressure",      # Modbus reg 8,9
    "label",
]


def os_search(index, query, size=10):
    url = f"{OS_BASE}/{index}/_search"
    try:
        r = requests.post(
            url,
            json={"size": size, "sort": [{"@timestamp": {"order": "desc"}}], **query},
            auth=OS_AUTH,
            verify=False,
            timeout=5,
        )
        return r.json().get("hits", {}).get("hits", [])
    except Exception as e:
        print(f"[WARN] OpenSearch query failed: {e}", file=sys.stderr)
        return []


def get_latest_nmea():
    result = {"heading": None, "rate_of_turn": None, "speed_knots": None, "lat": None, "lon": None}

    # HDT — heading
    hits = os_search(NMEA_IDX, {"query": {"term": {"sentence_type.keyword": "HDT"}}}, size=1)
    if hits:
        src = hits[0]["_source"]
        result["heading"] = src.get("heading")

    # ROT — rate of turn
    hits = os_search(NMEA_IDX, {"query": {"term": {"sentence_type.keyword": "ROT"}}}, size=1)
    if hits:
        src = hits[0]["_source"]
        result["rate_of_turn"] = src.get("rate_of_turn")

    # VTG — speed
    hits = os_search(NMEA_IDX, {"query": {"term": {"sentence_type.keyword": "VTG"}}}, size=1)
    if hits:
        src = hits[0]["_source"]
        result["speed_knots"] = src.get("spd_over_grnd_kts") or src.get("true_track")

    # RMC — position + speed fallback
    hits = os_search(NMEA_IDX, {"query": {"term": {"sentence_type.keyword": "RMC"}}}, size=1)
    if hits:
        src = hits[0]["_source"]
        result["lat"] = src.get("lat")
        result["lon"] = src.get("lon")
        if result["speed_knots"] is None:
            result["speed_knots"] = src.get("spd_over_grnd")

    return result


def decode_float_pair(lo, hi):
    try:
        b = struct.pack("<HH", int(lo), int(hi))
        return round(struct.unpack("<f", b)[0], 4)
    except Exception:
        return None


def get_latest_modbus():
    result = {v: None for v in REG_MAP.values()}

    hits = os_search(
        MOD_IDX,
        {
            "query": {
                "bool": {
                    "must": [
                        {"term": {"layers.ip.ip_ip_src": SGS_SRC_IP}},
                        {"exists": {"field": "layers.modbus.modbus_modbus_regval_uint16"}},
                    ]
                }
            }
        },
        size=5,
    )

    for hit in hits:
        modbus = hit["_source"].get("layers", {}).get("modbus", {})
        regs = modbus.get("modbus_modbus_regnum16")
        vals = modbus.get("modbus_modbus_regval_uint16")
        if not regs or not vals:
            continue
        if isinstance(regs, str):
            regs = [regs]
            vals = [vals]
        reg_dict = {int(r): int(v) for r, v in zip(regs, vals)}

        for (lo_r, hi_r), name in REG_MAP.items():
            if lo_r in reg_dict and hi_r in reg_dict:
                result[name] = decode_float_pair(reg_dict[lo_r], reg_dict[hi_r])

    return result


def collect(label, duration, output_file, append):
    mode = "a" if append else "w"
    write_header = not append

    print(f"[INFO] Collecting '{label}' for {duration}s → {output_file}")

    with open(output_file, mode, newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if write_header:
            writer.writeheader()

        start = time.time()
        count = 0

        while time.time() - start < duration:
            t0 = time.time()

            nmea = get_latest_nmea()
            modbus = get_latest_modbus()

            row = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "label": label,
                **nmea,
                **modbus,
            }

            writer.writerow(row)
            f.flush()
            count += 1

            elapsed = time.time() - t0
            remaining = POLL_SEC - elapsed
            if remaining > 0:
                time.sleep(remaining)

            if count % 30 == 0:
                print(f"[INFO] {count} rows | heading={nmea.get('heading')} "
                      f"rot={nmea.get('rate_of_turn')} spd={nmea.get('speed_knots')} "
                      f"oil_p={modbus.get('oil_pressure')} oil_t={modbus.get('oil_temp')}")

    print(f"[DONE] {count} rows written to {output_file}")


def main():
    parser = argparse.ArgumentParser(description="MaCySTe GRAD-RL Collector")
    parser.add_argument("--label",    required=True, help="Row label: normal / t0856_heading / t0814_radar_dos / recovery")
    parser.add_argument("--duration", type=int, default=300, help="Collection duration in seconds")
    parser.add_argument("--output",   default="macyste_dataset.csv", help="Output CSV file")
    parser.add_argument("--append",   action="store_true", help="Append to existing file")
    args = parser.parse_args()

    # Verify OpenSearch is reachable
    try:
        r = requests.get(f"{OS_BASE}/_cluster/health", auth=OS_AUTH, verify=False, timeout=5)
        status = r.json().get("status", "unknown")
        print(f"[INFO] OpenSearch status: {status}")
    except Exception as e:
        print(f"[ERROR] Cannot reach OpenSearch: {e}", file=sys.stderr)
        sys.exit(1)

    collect(args.label, args.duration, args.output, args.append)


if __name__ == "__main__":
    main()
