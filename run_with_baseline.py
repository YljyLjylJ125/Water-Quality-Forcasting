import argparse
from pathlib import Path

from config import load_config
from trainer import run_script
from utils import discover_model_scripts


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="*", default=None, help="Subset of model keys to run")
    parser.add_argument("--config-glob", default="config_*.yaml")
    args = parser.parse_args()

    scripts = discover_model_scripts("models")
    selected = [m.upper() for m in args.models] if args.models else None

    cfg_files = sorted(Path(".").glob(args.config_glob))
    if cfg_files:
        results = {}
        for cfg_path in cfg_files:
            cfg = load_config(str(cfg_path))
            model_cfg = cfg.get("model", {})
            model_name = model_cfg.get("name", "") if isinstance(model_cfg, dict) else str(model_cfg)
            model_key = model_name.upper()
            if selected and model_key not in selected:
                continue
            if model_key not in scripts:
                print(f"Skip {cfg_path.name}: model {model_key} not found in models/")
                continue
            results[cfg_path.name] = run_script(scripts[model_key], cwd=".")
    else:
        results = {}
        for key, path in scripts.items():
            if selected and key not in selected:
                continue
            results[key] = run_script(path, cwd=".")

    print("Run summary:")
    for k, v in results.items():
        print(f"{k}: exit_code={v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
