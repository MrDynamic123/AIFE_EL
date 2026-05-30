# SUMO RL Traffic Junction

Python application for training and running an RL traffic-light controller for a
4-way, 2-lane SUMO junction. It includes perturbations for violators,
ambulances, and lane-blocking accidents.

## Setup

1. Install SUMO from https://sumo.dlr.de/docs/Installing/index.html.
2. Set `SUMO_HOME` to your SUMO installation directory.
3. Install Python dependencies:

```powershell
pip install -r requirements.txt
```

## Train

```powershell
python train.py --algo dqn --timesteps 100000
```

Use `--gui` if you want to watch SUMO while training. The final model is saved
to `models/dqn_traffic_final.zip`.

## Run Interactive Control Panel

```powershell
python app.py --model models/dqn_traffic_final.zip --algo dqn
```

Controls:

- `Spawn Car`: injects a normal vehicle.
- `Spawn Violator`: injects a vehicle with `traci.vehicle.setSpeedMode(vehID, 0)`.
- `Spawn Ambulance`: injects an emergency vehicle with heavy reward penalties for delay.
- `Trigger Accident`: stops a random active vehicle and marks its lane as blocked.
- `Force Priority`: overrides the trained policy so the current ambulance approach gets green.

## Files

- `environment.py`: Gymnasium `SumoEnvironment`, SUMO asset generation, reward logic, accident/emergency/violation hooks.
- `train.py`: Stable Baselines3 DQN/PPO training script.
- `app.py`: Tkinter GUI and inference runner with live reward/waiting-time plots.

## Reward

The reward combines:

- `-1` per second of standard vehicle waiting time.
- `-50` per second of ambulance waiting time.
- queue-length penalty.
- pressure penalty from inbound/outbound vehicle count imbalance.
- `-1000` per collision.
- `+500` when an emergency vehicle clears the junction.
