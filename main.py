import argparse
import sys

from config import load_config
from trainer import run_script
from utils import discover_model_scripts


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config_model.yaml")
    parser.add_argument("--model", default=None, help="Model key, e.g. model / GNN / LSTM")
    args = parser.parse_args()

    cfg = load_config(args.config)
    scripts = discover_model_scripts("models")
    cfg_model = cfg.get("model", "")
    if isinstance(cfg_model, dict):
        cfg_model_name = cfg_model.get("name", "")
    else:
        cfg_model_name = cfg_model

    model_name = (args.model or cfg_model_name or "").upper()

    if not model_name:
        print("No model selected. Use --model or set model in config.", file=sys.stderr)
        return 2
    if model_name not in scripts:
        print(f"Unknown model: {model_name}. Available: {', '.join(scripts.keys())}", file=sys.stderr)
        return 2

    return run_script(scripts[model_name], cwd=".")


if __name__ == "__main__":
    raise SystemExit(main())
