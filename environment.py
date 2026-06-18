"""Gymnasium wrapper for a SUMO 4-way traffic junction.

The environment exposes a compact state vector for reinforcement learning and
keeps perturbation hooks available for an interactive controller.
"""

from __future__ import annotations

import os
import random
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

# Auto-detect SUMO installation on Windows if SUMO_HOME is missing
if "SUMO_HOME" not in os.environ:
    for default_path in [r"C:\Program Files (x86)\Eclipse\Sumo", r"C:\Program Files\Eclipse\Sumo"]:
        if os.path.exists(default_path):
            os.environ["SUMO_HOME"] = default_path
            os.environ["PATH"] += os.pathsep + os.path.join(default_path, "bin")
            break

import gymnasium as gym
import numpy as np
from gymnasium import spaces

try:
    import traci
    from sumolib import checkBinary
except ImportError:  # pragma: no cover - handled at runtime with a clearer error.
    traci = None
    checkBinary = None


ROOT = Path(__file__).resolve().parent
SUMO_DIR = ROOT / "sumo_assets"


@dataclass(frozen=True)
class JunctionSpec:
    junction_id: str = "J0"
    inbound_edges: tuple[str, ...] = ("N2J", "S2J", "E2J", "W2J")
    outbound_edges: tuple[str, ...] = ("J2N", "J2S", "J2E", "J2W")
    inbound_lanes: tuple[str, ...] = (
        "N2J_0",
        "N2J_1",
        "S2J_0",
        "S2J_1",
        "E2J_0",
        "E2J_1",
        "W2J_0",
        "W2J_1",
    )
    outbound_by_direction: dict[str, str] = None

    @property
    def inbound_lane_labels(self) -> tuple[str, ...]:
        return self.inbound_lanes


SPEC = JunctionSpec()
OUTBOUND_BY_INBOUND = {
    "N2J": ("J2S", "J2E", "J2W"),
    "S2J": ("J2N", "J2W", "J2E"),
    "E2J": ("J2W", "J2S", "J2N"),
    "W2J": ("J2E", "J2N", "J2S"),
}
ROUTE_IDS = {
    (src, dst): f"{src}_{dst}".replace("2J", "").replace("J2", "")
    for src, destinations in OUTBOUND_BY_INBOUND.items()
    for dst in destinations
}
ROUTE_EDGES = {route_id: edges for edges, route_id in ROUTE_IDS.items()}
MANEUVERS = {
    ("N2J", "J2S"): "straight",
    ("N2J", "J2E"): "left",
    ("N2J", "J2W"): "right",
    ("S2J", "J2N"): "straight",
    ("S2J", "J2W"): "left",
    ("S2J", "J2E"): "right",
    ("E2J", "J2W"): "straight",
    ("E2J", "J2S"): "left",
    ("E2J", "J2N"): "right",
    ("W2J", "J2E"): "straight",
    ("W2J", "J2N"): "left",
    ("W2J", "J2S"): "right",
}


def _require_sumo() -> None:
    if traci is None or checkBinary is None:
        raise RuntimeError(
            "SUMO TraCI bindings are missing. Install SUMO, set SUMO_HOME, then run "
            "`pip install -r requirements.txt`."
        )


def ensure_sumo_assets(base_dir: Path = SUMO_DIR) -> Path:
    """Generate a simple 4-way, 2-lane junction if assets are absent."""
    _require_sumo()
    base_dir.mkdir(parents=True, exist_ok=True)
    net_file = base_dir / "junction.net.xml"
    routes_file = base_dir / "routes.rou.xml"
    cfg_file = base_dir / "junction.sumocfg"

    node_file = base_dir / "nodes.nod.xml"
    edge_file = base_dir / "edges.edg.xml"
    connection_file = base_dir / "connections.con.xml"
    tl_file = base_dir / "tls.add.xml"

    node_file.write_text(
        """<nodes>
    <node id="J0" x="0" y="0" type="traffic_light"/>
    <node id="N" x="0" y="250" type="priority"/>
    <node id="S" x="0" y="-250" type="priority"/>
    <node id="E" x="250" y="0" type="priority"/>
    <node id="W" x="-250" y="0" type="priority"/>
</nodes>
""",
        encoding="utf-8",
    )
    edge_file.write_text(
        """<edges>
    <edge id="N2J" from="N" to="J0" numLanes="2" speed="13.9"/>
    <edge id="J2N" from="J0" to="N" numLanes="2" speed="13.9"/>
    <edge id="S2J" from="S" to="J0" numLanes="2" speed="13.9"/>
    <edge id="J2S" from="J0" to="S" numLanes="2" speed="13.9"/>
    <edge id="E2J" from="E" to="J0" numLanes="2" speed="13.9"/>
    <edge id="J2E" from="J0" to="E" numLanes="2" speed="13.9"/>
    <edge id="W2J" from="W" to="J0" numLanes="2" speed="13.9"/>
    <edge id="J2W" from="J0" to="W" numLanes="2" speed="13.9"/>
</edges>
""",
        encoding="utf-8",
    )
    connection_lines = ["<connections>"]
    for from_edge, to_edges in OUTBOUND_BY_INBOUND.items():
        for from_lane in range(2):
            for to_edge in to_edges:
                connection_lines.append(
                    f'    <connection from="{from_edge}" to="{to_edge}" fromLane="{from_lane}" toLane="{from_lane}"/>'
                )
    connection_lines.append("</connections>")
    connection_file.write_text("\n".join(connection_lines) + "\n", encoding="utf-8")
    tl_file.write_text(
        """<additional>
    <tlLogic id="J0" type="static" programID="rl" offset="0">
        <phase duration="60" state="GGGGGGrrrrrrGGGGGGrrrrrr"/>
        <phase duration="3" state="yyyyyyrrrrrryyyyyyrrrrrr"/>
        <phase duration="60" state="rrrrrrGGGGGGrrrrrrGGGGGG"/>
        <phase duration="3" state="rrrrrryyyyyyrrrrrryyyyyy"/>
    </tlLogic>
</additional>
""",
        encoding="utf-8",
    )
    subprocess.run(
        [
            checkBinary("netconvert"),
            "--node-files",
            str(node_file),
            "--edge-files",
            str(edge_file),
            "--connection-files",
            str(connection_file),
            "--tllogic-files",
            str(tl_file),
            "--offset.disable-normalization",
            "true",
            "--output-file",
            str(net_file),
        ],
        check=True,
    )

    route_lines = [
        "<routes>",
        '    <vType id="car" accel="1.4" decel="3.8" sigma="0.5" length="5" maxSpeed="6.0" color="0,0.45,1"/>',
        '    <vType id="violator" accel="1.8" decel="4.0" sigma="0.2" length="5" maxSpeed="7.0" color="1,0,0"/>',
        '    <vType id="ambulance" accel="2.0" decel="4.5" sigma="0.1" length="6.5" maxSpeed="8.0" color="1,1,1" guiShape="emergency"/>',
    ]
    for (from_edge, to_edge), route_id in ROUTE_IDS.items():
        route_lines.append(f'    <route id="{route_id}" edges="{from_edge} {to_edge}"/>')
    route_lines.append("</routes>")
    routes_file.write_text("\n".join(route_lines) + "\n", encoding="utf-8")
    cfg_file.write_text(
        f"""<configuration>
    <input>
        <net-file value="{net_file.name}"/>
        <route-files value="{routes_file.name}"/>
    </input>
    <time>
        <begin value="0"/>
        <step-length value="1"/>
    </time>
    <report>
        <no-step-log value="true"/>
        <no-warnings value="true"/>
    </report>
    <gui_only>
        <start value="true"/>
        <tracker-interval value="1"/>
    </gui_only>
</configuration>
""",
        encoding="utf-8",
    )
    return cfg_file


class SumoEnvironment(gym.Env):
    """Single-agent traffic-light control environment."""

    metadata = {"render_modes": ["human", "sumo-gui", None]}

    green_phases = (0, 2)
    yellow_after_green = {0: 1, 2: 3}

    def __init__(
        self,
        sumo_cfg: str | Path | None = None,
        use_gui: bool = False,
        max_steps: int = 3600,
        min_green_seconds: int = 3,
        seed: int | None = None,
    ) -> None:
        super().__init__()
        self.sumo_cfg = Path(sumo_cfg) if sumo_cfg else ensure_sumo_assets()
        self.use_gui = use_gui
        self.max_steps = max_steps
        self.min_green_seconds = min_green_seconds
        self.step_length = 0.5
        self.rng = random.Random(seed)
        self.step_count = 0
        self.current_green = 0
        self.last_switch_step = 0
        self.vehicle_counter = 0
        self.blocked_lanes: set[str] = set()
        self.accidents: dict[str, dict] = {}
        self.emergency_ids: set[str] = set()
        self.force_priority = True
        self.background_traffic = False
        self.last_waiting_time = 0.0
        self.cleared_emergencies = 0

        # 8 queues + phase + time_since_switch + 8 occupancies + 8 speeds + emergency + blocked
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(28,), dtype=np.float32)
        self.action_space = spaces.Discrete(len(self.green_phases))

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng.seed(seed)
        self.close()
        self._start_sumo()
        self.step_count = 0
        self.current_green = 0
        self.last_switch_step = 0
        self.vehicle_counter = 0
        self.blocked_lanes.clear()
        self.accidents.clear()
        self.emergency_ids.clear()
        self.force_priority = True
        self.background_traffic = False
        self.last_waiting_time = 0.0
        self.cleared_emergencies = 0
        traci.trafficlight.setProgram(SPEC.junction_id, "rl")
        traci.trafficlight.setPhase(SPEC.junction_id, self.current_green)
        return self._get_obs(), {}

    def step(self, action: int):
        if self.force_priority and self.emergency_ids:
            action = self._priority_action_for_emergency()
        self._apply_action(int(action))
        self._spawn_background_traffic()
        self._enforce_accident_blocks()
        traci.simulationStep()
        self.step_count += 1
        
        active_vehicles = set(traci.vehicle.getIDList())
        self.cleared_emergencies = len(self.emergency_ids - active_vehicles)
        self.emergency_ids.intersection_update(active_vehicles)
        
        self._clear_expired_accidents()
        self._issue_red_light_fines()
        obs = self._get_obs()
        reward = self._reward()
        terminated = False
        truncated = self.step_count >= self.max_steps
        info = self._info(reward)
        return obs, reward, terminated, truncated, info

    def close(self) -> None:
        if traci is not None and traci.isLoaded():
            traci.close(False)

    def spawn_car(
        self,
        violator: bool = False,
        ambulance: bool = False,
        origin_lane: str | None = None,
        target_edge: str | None = None,
    ) -> str:
        route_id, depart_lane = self._resolve_spawn_route(origin_lane, target_edge)
        prefix = "amb" if ambulance else "vio" if violator else "car"
        veh_id = f"{prefix}_{self.vehicle_counter}"
        self.vehicle_counter += 1
        veh_type = "ambulance" if ambulance else "violator" if violator else "car"
        traci.vehicle.add(
            veh_id, 
            route_id, 
            typeID=veh_type, 
            departLane=depart_lane, 
            departSpeed="max",
            departPos="185"
        )
        traci.vehicle.setMaxSpeed(veh_id, 8.0 if ambulance else 7.0 if violator else 6.0)
        if ambulance:
            self.emergency_ids.add(veh_id)
        self._set_turn_signal(veh_id, route_id)
        return veh_id

    def set_background_traffic(self, enabled: bool) -> None:
        self.background_traffic = enabled

    def trigger_accident(self) -> str | None:
        vehicle_ids = list(traci.vehicle.getIDList())
        candidates = [veh_id for veh_id in vehicle_ids if veh_id not in self.emergency_ids]
        vehicles = sorted(candidates or vehicle_ids, key=self._distance_to_junction)
        if vehicles:
            veh_id = vehicles[0]
        else:
            veh_id = self.spawn_car(origin_lane=self.rng.choice(SPEC.inbound_lanes), target_edge="Random")
            traci.simulationStep()
            self.step_count += 1
        lane_id = traci.vehicle.getLaneID(veh_id)
        lane_len = traci.lane.getLength(lane_id)
        pos = min(max(lane_len - 18.0, 5.0), lane_len - 2.0)
        road_id = traci.vehicle.getRoadID(veh_id)
        lane_index = traci.vehicle.getLaneIndex(veh_id)
        traci.vehicle.setStop(veh_id, road_id, pos=pos, laneIndex=lane_index, duration=48)
        traci.vehicle.setColor(veh_id, (255, 185, 28, 255))
        self.blocked_lanes.add(lane_id)
        x, y = traci.vehicle.getPosition(veh_id)
        self.accidents[veh_id] = {
            "id": veh_id,
            "lane": lane_id,
            "road": road_id,
            "lane_index": lane_index,
            "pos": pos,
            "x": x,
            "y": y,
            "clear_step": self.step_count + 48,
        }
        return veh_id

    def set_force_priority(self, enabled: bool) -> None:
        self.force_priority = enabled

    def _start_sumo(self) -> None:
        _require_sumo()
        binary = checkBinary("sumo-gui" if self.use_gui else "sumo")
        traci.start(
            [
                binary,
                "-c",
                str(self.sumo_cfg),
                "--start",
                "--quit-on-end",
                "false",
                "--step-length",
                str(self.step_length),
                "--delay",
                "160",
            ]
        )
        if self.use_gui:
            view_id = traci.gui.DEFAULT_VIEW
            traci.gui.setBoundary(view_id, -280, -280, 280, 280)
            traci.gui.setZoom(view_id, 900)

    def _apply_action(self, action: int) -> None:
        target_green = self.green_phases[action]
        min_green_steps = max(1, int(self.min_green_seconds / self.step_length))
        can_switch = self.step_count - self.last_switch_step >= min_green_steps
        if target_green == self.current_green or not can_switch:
            return
        traci.trafficlight.setPhase(SPEC.junction_id, self.yellow_after_green[self.current_green])
        traci.simulationStep()
        self.step_count += 1
        traci.trafficlight.setPhase(SPEC.junction_id, target_green)
        self.current_green = target_green
        self.last_switch_step = self.step_count

    def _spawn_background_traffic(self) -> None:
        if self.background_traffic and self.rng.random() < 0.1:
            self.spawn_car()


    def _enforce_accident_blocks(self) -> None:
        for accident in self.accidents.values():
            lane_id = accident["lane"]
            stop_pos = max(float(accident["pos"]) - 10.0, 1.0)
            for veh_id in traci.lane.getLastStepVehicleIDs(lane_id):
                if veh_id == accident["id"]:
                    continue
                if traci.vehicle.getLanePosition(veh_id) < stop_pos:
                    traci.vehicle.setStop(
                        veh_id,
                        accident["road"],
                        pos=stop_pos,
                        laneIndex=int(accident["lane_index"]),
                        duration=1,
                    )

    def _clear_expired_accidents(self) -> None:
        expired = [veh_id for veh_id, accident in self.accidents.items() if self.step_count >= accident["clear_step"]]
        for veh_id in expired:
            accident = self.accidents.pop(veh_id)
            self.blocked_lanes.discard(accident["lane"])
            if veh_id not in traci.vehicle.getIDList():
                continue
            try:
                traci.vehicle.setColor(veh_id, (0, 114, 255, 255))
            except traci.TraCIException:
                pass

    def _resolve_spawn_route(self, origin_lane: str | None, target_edge: str | None) -> tuple[str, str]:
        if origin_lane is None or origin_lane == "Random":
            origin_lane = self.rng.choice(SPEC.inbound_lanes)
        from_edge, lane_index = origin_lane.rsplit("_", 1)
        choices = OUTBOUND_BY_INBOUND[from_edge]
        if target_edge is None or target_edge == "Random" or target_edge not in choices:
            target_edge = self.rng.choice(choices)
        return ROUTE_IDS[(from_edge, target_edge)], lane_index

    def _get_obs(self) -> np.ndarray:
        queues = [min(traci.lane.getLastStepHaltingNumber(lane) / 20.0, 1.0) for lane in SPEC.inbound_lanes]
        phase = [self.current_green / 3.0]
        time_since_switch = [min((self.step_count - self.last_switch_step) / 100.0, 1.0)]
        occupancies = [min(traci.lane.getLastStepOccupancy(lane) / 100.0, 1.0) for lane in SPEC.inbound_lanes]
        speeds = [min(traci.lane.getLastStepMeanSpeed(lane) / 20.0, 1.0) for lane in SPEC.inbound_lanes]
        flags = [float(bool(self.emergency_ids & set(traci.vehicle.getIDList()))), float(bool(self.blocked_lanes))]
        return np.array(queues + phase + time_since_switch + occupancies + speeds + flags, dtype=np.float32)

    def _reward(self) -> float:
        vehicle_ids = traci.vehicle.getIDList()
        waiting = sum(traci.vehicle.getWaitingTime(v) for v in vehicle_ids)
        waiting_diff = waiting - self.last_waiting_time
        self.last_waiting_time = waiting
        
        ambulance_waiting = sum(traci.vehicle.getWaitingTime(v) for v in vehicle_ids if v in self.emergency_ids)
        queues = sum(traci.lane.getLastStepHaltingNumber(lane) for lane in SPEC.inbound_lanes)
        pressure = abs(
            sum(traci.edge.getLastStepVehicleNumber(edge) for edge in SPEC.inbound_edges)
            - sum(traci.edge.getLastStepVehicleNumber(edge) for edge in SPEC.outbound_edges)
        )
        collisions = traci.simulation.getCollidingVehiclesNumber()
        cleared = self.cleared_emergencies
        return -waiting_diff - queues - pressure - (50.0 * ambulance_waiting) - (1000.0 * collisions) + (500.0 * cleared)


    def _info(self, reward: float) -> dict:
        vehicle_ids = traci.vehicle.getIDList()
        avg_waiting = 0.0
        if vehicle_ids:
            avg_waiting = sum(traci.vehicle.getWaitingTime(v) for v in vehicle_ids) / len(vehicle_ids)
        return {
            "reward": reward,
            "avg_waiting_time": avg_waiting,
            "vehicles": len(vehicle_ids),
            "vehicle_states": [self._vehicle_state(veh_id) for veh_id in vehicle_ids],
            "phase": self.current_green,
            "blocked_lanes": list(self.blocked_lanes),
            "accidents": list(self.accidents.values()),
            "ambulance_priority": self._active_ambulance_priority(),
            "emergency_present": bool(self.emergency_ids),
        }

    def _vehicle_state(self, veh_id: str) -> dict:
        x, y = traci.vehicle.getPosition(veh_id)
        vehicle_type = traci.vehicle.getTypeID(veh_id).split("@", 1)[0]
        return {
            "id": veh_id,
            "type": vehicle_type,
            "x": x,
            "y": y,
            "angle": traci.vehicle.getAngle(veh_id),
            "speed": traci.vehicle.getSpeed(veh_id),
            "lane": traci.vehicle.getLaneID(veh_id),
            "route": traci.vehicle.getRouteID(veh_id),
            "target": self._target_for_route(traci.vehicle.getRouteID(veh_id)),
            "maneuver": self._maneuver_for_route(traci.vehicle.getRouteID(veh_id)),
            "accident": veh_id in self.accidents,
        }

    def _distance_to_junction(self, veh_id: str) -> float:
        x, y = traci.vehicle.getPosition(veh_id)
        return abs(x) + abs(y)


    @staticmethod
    def _target_for_route(route_id: str) -> str:
        return ROUTE_EDGES.get(route_id, ("", ""))[1]

    @staticmethod
    def _maneuver_for_route(route_id: str) -> str:
        return MANEUVERS.get(ROUTE_EDGES.get(route_id, ("", "")), "unknown")

    def _priority_action_for_emergency(self) -> int:
        for veh_id in list(self.emergency_ids):
            if veh_id not in traci.vehicle.getIDList():
                continue
            edge = traci.vehicle.getRoadID(veh_id)
            if edge in ("N2J", "S2J"):
                return 0
            if edge in ("E2J", "W2J"):
                return 1
        return self.green_phases.index(self.current_green)

    def _active_ambulance_priority(self) -> str | None:
        for veh_id in list(self.emergency_ids):
            if veh_id not in traci.vehicle.getIDList():
                continue
            edge = traci.vehicle.getRoadID(veh_id)
            if edge in ("N2J", "S2J"):
                return "North/South"
            if edge in ("E2J", "W2J"):
                return "East/West"
        return None

    def _edge_has_green(self, edge: str) -> bool:
        return (edge in ("N2J", "S2J") and self.current_green == 0) or (
            edge in ("E2J", "W2J") and self.current_green == 2
        )

    def _issue_red_light_fines(self) -> None:
        phase = traci.trafficlight.getPhase(SPEC.junction_id)
        green_ns = phase == 0
        green_ew = phase == 2
        for veh_id in traci.vehicle.getIDList():
            edge = traci.vehicle.getRoadID(veh_id)
            pos = traci.vehicle.getLanePosition(veh_id)
            lane_len = traci.lane.getLength(traci.vehicle.getLaneID(veh_id))
            beyond_stop_line = pos > lane_len - 8.0
            red_for_edge = (edge in ("N2J", "S2J") and not green_ns) or (edge in ("E2J", "W2J") and not green_ew)
            if beyond_stop_line and red_for_edge:
                self._issue_fine(veh_id)

    def _issue_fine(self, veh_id: str) -> None:
        traci.vehicle.setParameter(veh_id, "fine:red_light", str(self.step_count))

    @staticmethod
    def _set_turn_signal(veh_id: str, route_id: str) -> None:
        if route_id in {"N_E", "S_W", "E_S", "W_N"}:
            traci.vehicle.setSignals(veh_id, 0x02)
        elif route_id in {"N_W", "S_E", "E_N", "W_S"}:
            traci.vehicle.setSignals(veh_id, 0x01)


if __name__ == "__main__":
    cfg = ensure_sumo_assets()
    print(f"SUMO configuration ready: {cfg}")
