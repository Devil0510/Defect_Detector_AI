import os
import sys
import platform
import json
import subprocess
from pathlib import Path

def run_system_check():
    report = {
        "python_version": sys.version,
        "os_platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "gpu_available": False,
        "gpu_name": None,
        "cuda_version": None,
        "pytorch_version": None,
        "pytorch_cuda_available": False
    }
    
    # Check GPU via nvidia-smi
    try:
        smi_output = subprocess.check_output(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"], text=True)
        report["gpu_available"] = True
        report["gpu_name"] = smi_output.strip()
    except Exception:
        report["gpu_available"] = False
        
    # Check PyTorch
    try:
        import torch
        report["pytorch_version"] = torch.__version__
        report["pytorch_cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            report["cuda_device_name"] = torch.cuda.get_device_name(0)
    except ImportError:
        report["pytorch_version"] = "Not installed"
        
    out_dir = Path("reports")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "system_check.json"
    with open(out_file, "w") as f:
        json.dump(report, f, indent=4)
        
    print("=" * 60)
    print("SYSTEM AUDIT REPORT")
    print("=" * 60)
    for k, v in report.items():
        print(f"{k:<25}: {v}")
    print("=" * 60)
    print(f"Report saved to {out_file}")

if __name__ == "__main__":
    run_system_check()
