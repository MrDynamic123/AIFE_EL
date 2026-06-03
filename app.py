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
import customtkinter as ctk

from stable_baselines3 import DQN, PPO

from environment import OUTBOUND_BY_INBOUND, SPEC, SumoEnvironment, ensure_sumo_assets


class TrafficControlApp:
    def __init__(self, root: ctk.CTk, model_path: Path, algo: str) -> None:
        self.root = root
        self.root.title("SUMO RL Traffic Control")
        self.commands: queue.Queue[tuple[str, object]] = queue.Queue()
        self.metrics: queue.Queue[dict] = queue.Queue()
        self.running = True
        self.spawned: list[str] = []
        self.latest_info: dict | None = None

        self._build_ui()

        try:
            cfg = ensure_sumo_assets()
            self.env = SumoEnvironment(sumo_cfg=cfg, use_gui=False, min_green_seconds=10)
            model_cls = PPO if algo == "ppo" else DQN
            self.model = model_cls.load(model_path)
            self.worker = threading.Thread(target=self._simulation_loop, daemon=True)
            self.worker.start()
        except Exception as e:
            self.env = None
            print(f"Warning: Starting without SUMO backend due to error: {e}")
            self.status.set("SUMO missing. Frontend running in Preview Mode.")
            self._draw_map()

        self.root.after(250, self._refresh_metrics)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _build_ui(self) -> None:
        self.root.columnconfigure(0, weight=0, minsize=280)
        self.root.columnconfigure(1, weight=1)
        self.root.rowconfigure(0, weight=1)

        sidebar = ctk.CTkFrame(self.root, corner_radius=0, fg_color="#1e1e24")
        sidebar.grid(row=0, column=0, sticky="nsew")
        
        ctk.CTkLabel(sidebar, text="CONTROL PANEL", font=ctk.CTkFont(size=20, weight="bold")).pack(pady=(25, 25))

        self.vehicle_type = ctk.StringVar(value="car")
        self.origin_lane = ctk.StringVar(value="Random")
        self.target_edge = ctk.StringVar(value="Random")

        ctk.CTkLabel(sidebar, text="Vehicle Type").pack(anchor="w", padx=20, pady=(0, 2))
        ctk.CTkComboBox(sidebar, variable=self.vehicle_type, values=["car", "ambulance", "violator"], state="readonly").pack(fill="x", padx=20, pady=(0, 15))

        ctk.CTkLabel(sidebar, text="From Lane").pack(anchor="w", padx=20, pady=(0, 2))
        ctk.CTkComboBox(sidebar, variable=self.origin_lane, values=["Random", *SPEC.inbound_lanes], state="readonly").pack(fill="x", padx=20, pady=(0, 15))

        ctk.CTkLabel(sidebar, text="To Road").pack(anchor="w", padx=20, pady=(0, 2))
        ctk.CTkComboBox(sidebar, variable=self.target_edge, values=["Random", *SPEC.outbound_edges], state="readonly").pack(fill="x", padx=20, pady=(0, 20))

        ctk.CTkButton(sidebar, text="Spawn Selected", command=self._spawn_selected, height=40, font=ctk.CTkFont(weight="bold")).pack(fill="x", padx=20, pady=(0, 10))
        ctk.CTkButton(sidebar, text="Trigger Accident", command=lambda: self.commands.put(("accident", None)), height=40, fg_color="#ef4444", hover_color="#dc2626", font=ctk.CTkFont(weight="bold")).pack(fill="x", padx=20, pady=(0, 25))

        self.force_priority = ctk.BooleanVar(value=True)
        ctk.CTkCheckBox(sidebar, text="Force Ambulance Priority", variable=self.force_priority, command=lambda: self.commands.put(("force_priority", self.force_priority.get()))).pack(anchor="w", padx=20, pady=(0, 15))

        self.background_traffic = ctk.BooleanVar(value=False)
        ctk.CTkCheckBox(sidebar, text="Enable Background Traffic", variable=self.background_traffic, command=lambda: self.commands.put(("background_traffic", self.background_traffic.get()))).pack(anchor="w", padx=20, pady=(0, 10))

        self.status = ctk.StringVar(value="System Ready")
        ctk.CTkLabel(sidebar, textvariable=self.status, font=ctk.CTkFont(size=13, slant="italic"), text_color="#a1a1aa", wraplength=240).pack(side="bottom", pady=20, padx=20)

        main_view = ctk.CTkFrame(self.root, corner_radius=0, fg_color="#2b2b36")
        main_view.grid(row=0, column=1, sticky="nsew")
        main_view.rowconfigure(1, weight=1)
        main_view.columnconfigure(0, weight=1)

        dashboard = ctk.CTkFrame(main_view, fg_color="transparent")
        dashboard.grid(row=0, column=0, sticky="ew", padx=20, pady=20)
        for i in range(4):
            dashboard.columnconfigure(i, weight=1)

        self.val_vehicles = ctk.StringVar(value="0")
        self.val_signal = ctk.StringVar(value="N/A")
        self.val_priority = ctk.StringVar(value="None")
        self.val_blocked = ctk.StringVar(value="0")

        self._create_metric_card(dashboard, 0, "Active Vehicles", self.val_vehicles)
        self._create_metric_card(dashboard, 1, "Signal Phase", self.val_signal, text_color="#22c55e")
        self._create_metric_card(dashboard, 2, "Ambulance Priority", self.val_priority)
        self._create_metric_card(dashboard, 3, "Blocked Lanes", self.val_blocked, text_color="#ef4444")

        canvas_frame = ctk.CTkFrame(main_view, fg_color="#2b2b36")
        canvas_frame.grid(row=1, column=0, sticky="nsew", padx=20, pady=(0, 20))
        canvas_frame.rowconfigure(0, weight=1)
        canvas_frame.columnconfigure(0, weight=1)
        
        self.map_canvas = tk.Canvas(canvas_frame, bg="#2b2b36", highlightthickness=0)
        self.map_canvas.grid(row=0, column=0, sticky="nsew")
        self.map_canvas.bind("<Configure>", lambda _event: self._draw_map())

    def _create_metric_card(self, parent, col, title, variable, text_color="#ffffff"):
        card = ctk.CTkFrame(parent, fg_color="#363645", corner_radius=10)
        card.grid(row=0, column=col, sticky="ew", padx=5)
        ctk.CTkLabel(card, text=title, font=ctk.CTkFont(size=13, weight="normal"), text_color="#a1a1aa").pack(pady=(12, 0))
        ctk.CTkLabel(card, textvariable=variable, font=ctk.CTkFont(size=26, weight="bold"), text_color=text_color).pack(pady=(0, 12))

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
            signal = "N/S Green" if latest["phase"] == 0 else "E/W Green"
            priority = latest.get("ambulance_priority") or "None"
            self.val_vehicles.set(str(latest["vehicles"]))
            self.val_signal.set(signal)
            self.val_priority.set(priority)
            self.val_blocked.set(str(len(latest["blocked_lanes"])))
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

        canvas.create_rectangle(0, 0, width, height, fill="#2b2b36", outline="")
        canvas.create_rectangle(cx - road / 2, cy - extent, cx + road / 2, cy + extent, fill="#3f3f4e", outline="")
        canvas.create_rectangle(cx - extent, cy - road / 2, cx + extent, cy + road / 2, fill="#3f3f4e", outline="")
        canvas.create_rectangle(cx - road / 2, cy - road / 2, cx + road / 2, cy + road / 2, fill="#4c4c5e", outline="")

        for offset in (-lane, lane):
            canvas.create_line(cx + offset, cy - extent, cx + offset, cy + extent, fill="#7a7a8c", dash=(10, 10), width=1)
            canvas.create_line(cx - extent, cy + offset, cx + extent, cy + offset, fill="#7a7a8c", dash=(10, 10), width=1)

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
    
    ctk.set_appearance_mode("Dark")
    ctk.set_default_color_theme("blue")
    
    root = ctk.CTk()
    root.geometry("1150x700")
    TrafficControlApp(root, model_path, args.algo)
    root.mainloop()


if __name__ == "__main__":
    main()
