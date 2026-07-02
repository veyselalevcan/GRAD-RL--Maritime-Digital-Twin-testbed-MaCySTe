#!/usr/bin/env python3
"""T0856 - Heading Injection 180° — Live Test, collector_v2 kullanan versiyon"""
import asyncio, subprocess, time

DURATION   = 180   # 3 dakika saldırı
RECOVERY   = 60    # 1 dakika recovery
CC_WS_URL  = "ws://192.168.249.9:3000"
OUTPUT     = "/root/MaCySTe/grad_rl_data/live_test_20260619.csv"
COLLECTOR  = "/root/MaCySTe/macyste_collector_v2.py"

def run_collector(label, duration):
    cmd = ["python3", COLLECTOR, "--label", label,
           "--duration", str(duration), "--output", OUTPUT, "--append"]
    print(f"[COLLECTOR] label={label} duration={duration}s")
    return subprocess.Popen(cmd)

async def main():
    import aiohttp
    print("[T0856] Heading Injection 180° başlıyor...")
    collector = run_collector("t0856_heading_180", DURATION)

    async with aiohttp.ClientSession() as session:
        async with session.ws_connect(CC_WS_URL + "/command",
                                      autoping=True, heartbeat=10.0) as ws:
            await ws.send_json({"jsonrpc":"2.0","method":"attack_start",
                                "params":["inject_heading", 180, 10],"id":1})
            resp = await asyncio.wait_for(ws.receive_json(), timeout=5)
            print(f"[WS] {resp}")

            if "error" in resp:
                print("[ERROR] Saldırı başlatılamadı")
                collector.kill()
                return

            print(f"[ATTACK] {DURATION}s bekleniyor...")
            await asyncio.sleep(DURATION)

            await ws.send_json({"jsonrpc":"2.0","method":"attack_stop",
                                "params":["inject_heading"],"id":2})
            try:
                resp = await asyncio.wait_for(ws.receive_json(), timeout=5)
                print(f"[WS] Stop: {resp}")
            except Exception:
                pass

    collector.wait()
    print(f"[RECOVERY] {RECOVERY}s...")
    run_collector("recovery", RECOVERY).wait()
    print("[DONE] T0856 tamamlandı")

asyncio.run(main())
