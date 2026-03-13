import subprocess
import sys
from pathlib import Path
from typing import Dict, Optional


def run_script(script_path: str, extra_args: Optional[list] = None, cwd: Optional[str] = None) -> int:
    script = Path(script_path)
    if not script.exists():
        raise FileNotFoundError(f"Script not found: {script}")

    cmd = [sys.executable, str(script)]
    if extra_args:
        cmd.extend(extra_args)
    proc = subprocess.run(cmd, cwd=cwd, check=False)
    return proc.returncode


def run_many(model_map: Dict[str, str], selected: Optional[list] = None, cwd: Optional[str] = None) -> Dict[str, int]:
    names = selected if selected else list(model_map.keys())
    results = {}
    for name in names:
        path = model_map[name]
        results[name] = run_script(path, cwd=cwd)
    return results
