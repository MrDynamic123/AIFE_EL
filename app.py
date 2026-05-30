"""Interactive SUMO RL control panel.

Run after training a model:
    python app.py --model models/dqn_traffic_final.zip
"""

from __future__ import annotations

import argparse
import queue
import threading
import time
import tkinter as tk
from pathlib import Path
from tkinter import ttk

from stable_baselines3 import DQN, PPO

from environment import OUTBOUND_BY_INBOUND, SPEC, SumoEnvironment, ensure_sumo_assets


class TrafficControlApp:
    def __init__(self, root: tk.Tk, model_path: Path, algo: str) -> None:
        self.root = root
        self.root.title("SUMO RL Traffic Control")
        self.commands: queue.Queue[tuple[str, object]] = queue.Queue()
        self.metrics: queue.Queue[dict] = queue.Queue()
        self.running = True
        self.spawned: list[str] = []
        self.latest_info: dict | None = None

        cfg = ensure_sumo_assets()
        self.env = SumoEnvironment(sumo_cfg=cfg, use_gui=False, min_green_seconds=10)
        model_cls = PPO if algo == "ppo" else DQN
        self.model = model_cls.load(model_path)

        self._build_ui()
        self.worker = threading.Thread(target=self._simulation_loop, daemon=True)
        self.worker.start()
        self.root.after(250, self._refresh_metrics)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _build_ui(self) -> None:
        frame = ttk.Frame(self.root, padding=10)
        frame.grid(row=0, column=0, sticky="nsew")
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)

        buttons = ttk.Frame(frame)
        buttons.grid(row=0, column=0, sticky="ew")

        self.vehicle_type = tk.StringVar(value="car")
        self.origin_lane = tk.StringVar(value="Random")
        self.target_edge = tk.StringVar(value="Random")

        ttk.Label(buttons, text="Vehicle").grid(row=0, column=0, sticky="w", padx=(0, 4))
        ttk.Combobox(
            buttons,
            textvariable=self.vehicle_type,
            values=("car", "ambulance", "violator"),
            width=12,
            state="readonly",
        ).grid(row=0, column=1, padx=(0, 10))

        ttk.Label(buttons, text="From lane").grid(row=0, column=2, sticky="w", padx=(0, 4))
        ttk.Combobox(
            buttons,
            textvariable=self.origin_lane,
            values=("Random", *SPEC.inbound_lanes),
            width=12,
            state="readonly",
        ).grid(row=0, column=3, padx=(0, 10))

        ttk.Label(buttons, text="To road").grid(row=0, column=4, sticky="w", padx=(0, 4))
        ttk.Combobox(
            buttons,
            textvariable=self.target_edge,
            values=("Random", *SPEC.outbound_edges),
            width=12,
            state="readonly",
        ).grid(row=0, column=5, padx=(0, 10))

        ttk.Button(buttons, text="Spawn Selected", command=self._spawn_selected).grid(row=0, column=6, padx=4)
        ttk.Button(buttons, text="Trigger Accident", command=lambda: self.commands.put(("accident", None))).grid(row=0, column=7, padx=4)

        self.force_priority = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            buttons,
            text="Force Priority",
            variable=self.force_priority,
            command=lambda: self.commands.put(("force_priority", self.force_priority.get())),
        ).grid(row=1, column=0, columnspan=2, sticky="w", pady=(8, 0))

        self.background_traffic = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            buttons,
            text="Background Traffic",
            variable=self.background_traffic,
            command=lambda: self.commands.put(("background_traffic", self.background_traffic.get())),
        ).grid(row=1, column=2, columnspan=2, sticky="w", pady=(8, 0))

        self.status = tk.StringVar(value="Starting SUMO...")
        ttk.Label(frame, textvariable=self.status).grid(row=1, column=0, sticky="w", pady=(10, 0))

        self.map_canvas = tk.Canvas(frame, bg="#2f3437", highlightthickness=0, width=860, height=520)
        self.map_canvas.grid(row=2, column=0, sticky="nsew", pady=(10, 0))
        frame.rowconfigure(2, weight=1)
        frame.columnconfigure(0, weight=1)
        self.map_canvas.bind("<Configure>", lambda _event: self._draw_map())

    def _spawn_selected(self) -> None:
        self.commands.put(
            (
                "spawn_selected",
                {
                    "vehicle_type": self.vehicle_type.get(),
                    "origin_lane": self.origin_lane.get(),
                    "target_edge": self.target_edge.get(),
                },
            )
        )

    def _simulation_loop(self) -> None:
        obs, _ = self.env.reset()
        while self.running:
            self._drain_commands()
            action, _ = self.model.predict(obs, deterministic=True)
            obs, reward, _, truncated, info = self.env.step(int(action))
            self.metrics.put(info)
            if truncated:
                obs, _ = self.env.reset()
            time.sleep(0.25)
        self.env.close()

    def _drain_commands(self) -> None:
        while True:
            try:
                command, value = self.commands.get_nowait()
            except queue.Empty:
                return
            if command == "spawn_car":
                self.env.spawn_car()
            elif command == "spawn_violator":
                self.env.spawn_car(violator=True)
            elif command == "spawn_ambulance":
                self.env.spawn_car(ambulance=True)
            elif command == "spawn_selected":
                payload = dict(value)
                vehicle_type = payload["vehicle_type"]
                origin_lane = payload["origin_lane"]
                target_edge = payload["target_edge"]
                if not self._target_is_valid(origin_lane, target_edge):
                    self.metrics.put({"message": f"{origin_lane} cannot route to {target_edge}. Pick Random or a valid road."})
                    continue
                veh_id = self.env.spawn_car(
                    violator=vehicle_type == "violator",
                    ambulance=vehicle_type == "ambulance",
                    origin_lane=origin_lane,
                    target_edge=target_edge,
                )
                self.spawned.append(veh_id)
            elif command == "accident":
                self.env.trigger_accident()
            elif command == "force_priority":
                self.env.set_force_priority(bool(value))
            elif command == "background_traffic":
                self.env.set_background_traffic(bool(value))

    def _refresh_metrics(self) -> None:
        latest = None
        while True:
            try:
                latest = self.metrics.get_nowait()
            except queue.Empty:
                break
        if latest:
            if "message" in latest:
                self.status.set(latest["message"])
                if self.running:
                    self.root.after(250, self._refresh_metrics)
                return
            self.latest_info = latest
            signal = "North/South green" if latest["phase"] == 0 else "East/West green"
            priority = latest.get("ambulance_priority") or "none"
            self.status.set(
                "Vehicles: {vehicles} | Signal: {signal} | Ambulance priority: {priority} | Accidents: {blocked}".format(
                    vehicles=latest["vehicles"],
                    signal=signal,
                    priority=priority,
                    blocked=len(latest["blocked_lanes"]),
                )
            )
            self._draw_map()
        if self.running:
            self.root.after(80, self._refresh_metrics)

    def _draw_map(self) -> None:
        canvas = self.map_canvas
        canvas.delete("all")
        width = max(canvas.winfo_width(), 1)
        height = max(canvas.winfo_height(), 1)
        cx = width / 2
        cy = height / 2
        scale = min(width, height) / 620
        road = 92 * scale
        lane = road / 4
        extent = min(width, height) * 0.48

        canvas.create_rectangle(0, 0, width, height, fill="#2f3437", outline="")
        canvas.create_rectangle(cx - road / 2, cy - extent, cx + road / 2, cy + extent, fill="#4a4f52", outline="")
        canvas.create_rectangle(cx - extent, cy - road / 2, cx + extent, cy + road / 2, fill="#4a4f52", outline="")
        canvas.create_rectangle(cx - road / 2, cy - road / 2, cx + road / 2, cy + road / 2, fill="#555b5f", outline="")

        for offset in (-lane, lane):
            canvas.create_line(cx + offset, cy - extent, cx + offset, cy + extent, fill="#d8dde0", dash=(10, 10), width=1)
            canvas.create_line(cx - extent, cy + offset, cx + extent, cy + offset, fill="#d8dde0", dash=(10, 10), width=1)

        canvas.create_text(cx, cy - extent + 18, text="N2J", fill="#eef2f3", font=("Segoe UI", 10, "bold"))
        canvas.create_text(cx, cy + extent - 18, text="S2J", fill="#eef2f3", font=("Segoe UI", 10, "bold"))
        canvas.create_text(cx + extent - 24, cy, text="E2J", fill="#eef2f3", font=("Segoe UI", 10, "bold"))
        canvas.create_text(cx - extent + 24, cy, text="W2J", fill="#eef2f3", font=("Segoe UI", 10, "bold"))

        info = self.latest_info or {}
        phase = info.get("phase", 0)
        self._draw_lane_signals(cx, cy, road, scale, phase)

        for accident in info.get("accidents", []):
            self._draw_accident(accident, cx, cy, scale)

        for vehicle in info.get("vehicle_states", []):
            self._draw_vehicle(vehicle, cx, cy, scale)

    def _draw_lane_signals(self, cx: float, cy: float, road: float, scale: float, phase: int) -> None:
        groups = [
            ("N", cx - road * 0.82, cy - road * 1.32, phase == 0),
            ("S", cx + road * 0.82, cy + road * 1.32, phase == 0),
            ("E", cx + road * 1.32, cy - road * 0.82, phase == 2),
            ("W", cx - road * 1.32, cy + road * 0.82, phase == 2),
        ]
        for label, x, y, allowed in groups:
            self.map_canvas.create_text(x, y - 18 * scale, text=label, fill="#eef2f3", font=("Segoe UI", 9, "bold"))
            for index, move in enumerate(("S", "L", "R")):
                color = "#22c55e" if allowed else "#ef4444"
                px = x + (index - 1) * 18 * scale
                self.map_canvas.create_oval(px - 6, y - 6, px + 6, y + 6, fill=color, outline="#111827", width=1)
                self.map_canvas.create_text(px, y + 15 * scale, text=move, fill="#f8fafc", font=("Segoe UI", 8, "bold"))

    def _draw_accident(self, accident: dict, cx: float, cy: float, scale: float) -> None:
        x = cx + (float(accident["x"]) - 250.0) * scale
        y = cy - (float(accident["y"]) - 250.0) * scale
        size = 13 * scale
        self.map_canvas.create_polygon(
            x,
            y - size,
            x + size,
            y + size,
            x - size,
            y + size,
            fill="#f59e0b",
            outline="#111827",
            width=2,
        )
        self.map_canvas.create_text(x, y + size * 0.35, text="!", fill="#111827", font=("Segoe UI", 10, "bold"))

    def _draw_vehicle(self, vehicle: dict, cx: float, cy: float, scale: float) -> None:
        x = cx + (float(vehicle["x"]) - 250.0) * scale
        y = cy - (float(vehicle["y"]) - 250.0) * scale
        length = 18 * scale
        width = 10 * scale
        fill = {"car": "#38bdf8", "ambulance": "#f8fafc", "violator": "#ef4444"}.get(vehicle["type"], "#a78bfa")
        outline = "#f59e0b" if vehicle.get("accident") else "#dc2626" if vehicle["type"] == "ambulance" else "#111827"
        self.map_canvas.create_rectangle(x - width / 2, y - length / 2, x + width / 2, y + length / 2, fill=fill, outline=outline, width=2)
        if vehicle["type"] == "ambulance":
            self.map_canvas.create_line(x - width / 3, y, x + width / 3, y, fill="#dc2626", width=2)
            self.map_canvas.create_line(x, y - width / 3, x, y + width / 3, fill="#dc2626", width=2)
        self._draw_vehicle_intent(vehicle, x, y, scale)

    def _draw_vehicle_intent(self, vehicle: dict, x: float, y: float, scale: float) -> None:
        directions = {
            "J2N": (0, -1),
            "J2S": (0, 1),
            "J2E": (1, 0),
            "J2W": (-1, 0),
        }
        dx, dy = directions.get(vehicle.get("target", ""), (0, -1))
        start = 10 * scale
        end = 28 * scale
        x1 = x + dx * start
        y1 = y + dy * start
        x2 = x + dx * end
        y2 = y + dy * end
        color = "#fde047" if vehicle["type"] == "ambulance" else "#f8fafc"
        self.map_canvas.create_line(x1, y1, x2, y2, fill=color, width=2, arrow=tk.LAST)
        label = {"straight": "S", "left": "L", "right": "R"}.get(vehicle.get("maneuver"), "?")
        radius = 8 * scale
        self.map_canvas.create_oval(x2 - radius, y2 - radius, x2 + radius, y2 + radius, fill="#111827", outline=color, width=1)
        self.map_canvas.create_text(x2, y2, text=label, fill=color, font=("Segoe UI", 8, "bold"))

    def _on_close(self) -> None:
        self.running = False
        self.root.after(300, self.root.destroy)

    @staticmethod
    def _target_is_valid(origin_lane: str, target_edge: str) -> bool:
        if origin_lane == "Random" or target_edge == "Random":
            return True
        from_edge = origin_lane.rsplit("_", 1)[0]
        return target_edge in OUTBOUND_BY_INBOUND[from_edge]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run trained SUMO RL agent with interactive controls.")
    parser.add_argument("--model", default="models/dqn_traffic_final.zip")
    parser.add_argument("--algo", choices=("dqn", "ppo"), default="dqn")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_path = Path(args.model)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}. Train one with `python train.py` first.")
    root = tk.Tk()
    root.geometry("920x620")
    TrafficControlApp(root, model_path, args.algo)
    root.mainloop()


if __name__ == "__main__":
    main()
