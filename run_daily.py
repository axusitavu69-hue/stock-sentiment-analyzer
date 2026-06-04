"""独立进程启动器"""
import subprocess, sys, os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
r = subprocess.run([sys.executable, 'train_model.py', '--daily'])
sys.exit(r.returncode)
