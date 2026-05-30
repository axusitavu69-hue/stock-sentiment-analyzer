import subprocess, time, sys, os

# Auto-restart SSH tunnel to localhost.run
# Keeps the tunnel alive even if connection drops

def get_tunnel():
    cmd = [
        "ssh", "-o", "StrictHostKeyChecking=no",
        "-o", "ServerAliveInterval=30",
        "-o", "ServerAliveCountMax=3",
        "-R", "80:localhost:8501",
        "nokey@localhost.run"
    ]
    while True:
        print(f"[{time.strftime('%H:%M:%S')}] Starting tunnel...")
        try:
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                    text=True, bufsize=1)
            for line in proc.stdout:
                line = line.strip()
                if line:
                    print(line)
                if "lhr.life" in line and "tunneled" in line:
                    print(f">>> PUBLIC URL: {line.split()[-1]}")
            proc.wait()
            print(f"[{time.strftime('%H:%M:%S')}] Tunnel disconnected. Restarting in 5s...")
        except Exception as e:
            print(f"[ERROR] {e}")
        time.sleep(5)

if __name__ == "__main__":
    get_tunnel()
