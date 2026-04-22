"""
register_model.py — Register a trained model zip into the local registry.

Usage
-----
# Register SAC (most common):
    python scripts/register_model.py --algo SAC --zip data/models/trading_bot_model_SAC.zip

# Register PPO:
    python scripts/register_model.py --algo PPO --zip data/models/trading_bot_model_PPO.zip

# Register with extra metadata:
    python scripts/register_model.py --algo SAC --zip data/models/trading_bot_model_SAC.zip \
        --run-name sac_nsei_v2 --timesteps 500000 --train-start 2018-01-01 --train-end 2023-12-31

# Just promote an already-registered run to production:
    python scripts/register_model.py --promote sac_nsei_v1
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

# Make src/ importable when running from project root
_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT / "src"))

from trading_bot.models.registry import ModelRecord, ModelRegistry  # noqa: E402

_REGISTRY_PATH = _ROOT / "data" / "models" / "registry.json"


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Register a trained model in registry.json")

    p.add_argument("--algo", default="SAC", help="Algorithm: SAC | PPO | TD3")
    p.add_argument("--zip", dest="zip_path", default="", help="Path to model .zip file")
    p.add_argument("--ticker", default="^NSEI", help="NSE ticker (default: ^NSEI)")
    p.add_argument(
        "--run-name",
        default="",
        help="Unique run name (auto-generated if omitted)",
    )
    p.add_argument("--timesteps", type=int, default=500_000, help="Total training timesteps")
    p.add_argument("--train-start", default="2018-01-01", help="Training period start (ISO date)")
    p.add_argument("--train-end", default="2023-12-31", help="Training period end (ISO date)")
    p.add_argument("--notes", default="", help="Free-text notes")
    p.add_argument(
        "--no-promote",
        action="store_true",
        help="Register without promoting to production",
    )
    p.add_argument(
        "--promote",
        metavar="RUN_NAME",
        default="",
        help="Just promote an existing run_name to production (skips registration)",
    )
    p.add_argument(
        "--registry",
        default=str(_REGISTRY_PATH),
        help="Path to registry.json (default: data/models/registry.json)",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    registry = ModelRegistry(Path(args.registry))

    # ── Promote-only mode ─────────────────────────────────────────────────────
    if args.promote:
        try:
            registry.promote_to_production(args.promote)
            print(f"[OK] Promoted '{args.promote}' to production.")
        except KeyError as exc:
            print(f"[ERROR] {exc}")
            sys.exit(1)
        return

    # ── Registration mode ─────────────────────────────────────────────────────
    if not args.zip_path:
        print("[ERROR] --zip is required when registering a model.")
        sys.exit(1)

    zip_path = Path(args.zip_path)
    if not zip_path.exists():
        print(f"[ERROR] Zip not found: {zip_path}")
        print("        Download from Colab first, then re-run.")
        sys.exit(1)

    run_name = args.run_name or (
        f"{args.algo.lower()}_{args.ticker.replace('^', '').replace('.', '_')}_v1"
    )

    record = ModelRecord(
        run_name=run_name,
        algo=args.algo.upper(),
        ticker=args.ticker,
        train_start=args.train_start,
        train_end=args.train_end,
        total_timesteps=args.timesteps,
        model_path=str(zip_path),
        vec_norm_path="",
        pipeline_path="",
        feature_names=[],
        hyperparams={},
        train_metrics={},
        val_metrics={},
        created_at=datetime.now().isoformat(),
        notes=args.notes or f"{args.algo} production model",
        is_production=False,
    )

    registry.register(record)
    print(f"[OK] Registered '{run_name}' ({args.algo} / {args.ticker})")

    if not args.no_promote:
        registry.promote_to_production(run_name)
        print(f"[OK] Promoted '{run_name}' to production.")

    # Confirm what's now in production
    prod = registry.get_production_model(args.ticker, args.algo)
    if prod:
        print(f"\nProduction {args.algo}: {prod.run_name} → {prod.model_path}")
    else:
        print(f"\n[WARN] No production {args.algo} model found after registration.")


if __name__ == "__main__":
    main()
