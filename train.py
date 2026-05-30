"""Train a Stable Baselines3 agent for the SUMO junction environment."""

from __future__ import annotations

import argparse
from pathlib import Path

from stable_baselines3 import DQN, PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor

from environment import SumoEnvironment, ensure_sumo_assets


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train RL traffic-light control in SUMO.")
    parser.add_argument("--algo", choices=("dqn", "ppo"), default="dqn")
    parser.add_argument("--timesteps", type=int, default=100_000)
    parser.add_argument("--gui", action="store_true", help="Use SUMO GUI during training.")
    parser.add_argument("--model-dir", default="models")
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    cfg = ensure_sumo_assets()

    env = Monitor(SumoEnvironment(sumo_cfg=cfg, use_gui=args.gui, seed=args.seed))
    checkpoint = CheckpointCallback(
        save_freq=10_000,
        save_path=str(model_dir),
        name_prefix=f"{args.algo}_traffic",
    )

    if args.algo == "ppo":
        model = PPO(
            "MlpPolicy",
            env,
            verbose=1,
            seed=args.seed,
            learning_rate=3e-4,
            n_steps=1024,
            batch_size=128,
            gamma=0.99,
        )
    else:
        model = DQN(
            "MlpPolicy",
            env,
            verbose=1,
            seed=args.seed,
            learning_rate=1e-4,
            buffer_size=50_000,
            learning_starts=2_000,
            batch_size=64,
            gamma=0.99,
            train_freq=4,
            target_update_interval=1_000,
            exploration_fraction=0.25,
            exploration_final_eps=0.05,
        )

    model.learn(total_timesteps=args.timesteps, callback=checkpoint, progress_bar=True)
    output_path = model_dir / f"{args.algo}_traffic_final"
    model.save(output_path)
    env.close()
    print(f"Saved trained model to {output_path}.zip")


if __name__ == "__main__":
    main()
