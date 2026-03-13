from pathlib import Path
from typing import Dict


def discover_model_scripts(models_dir: str = "models") -> Dict[str, str]:
    base = Path(models_dir)
    scripts = {}
    for p in sorted(base.glob("*.py")):
        if p.name.startswith("_"):
            continue
        scripts[p.stem.upper()] = str(p)
    return scripts
