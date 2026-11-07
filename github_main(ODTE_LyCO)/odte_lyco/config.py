"""Fixed paper configuration; no legacy formats or validation layer."""

from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import yaml


@dataclass(frozen=True)
class DAGNodeConfig:
    node_id: str
    compute_ratio: float


@dataclass(frozen=True)
class DAGEdgeConfig:
    source: str
    target: str
    data_ratio: float


@dataclass(frozen=True)
class DAGTemplateConfig:
    nodes: tuple[DAGNodeConfig, ...]
    edges: tuple[DAGEdgeConfig, ...]
    source_node: str
    sink_node: str

    def predecessors(self, node_id: str) -> tuple[str, ...]:
        return tuple((edge.source for edge in self.edges if edge.target == node_id))

    def successors(self, node_id: str) -> tuple[str, ...]:
        return tuple((edge.target for edge in self.edges if edge.source == node_id))

    def incoming_data_ratio(self, node_id: str) -> float:
        return sum((edge.data_ratio for edge in self.edges if edge.target == node_id))

    def outgoing_data_ratio(self, node_id: str) -> float:
        return sum((edge.data_ratio for edge in self.edges if edge.source == node_id))


@dataclass(frozen=True)
class ServiceClassConfig:
    name: str
    arrival_rate: float
    arrival_scope: str
    input_bits: int
    cycles_per_bit: float
    output_ratio: float
    deadline_slots: int
    odte_target: float
    dag: DAGTemplateConfig


@dataclass(frozen=True)
class ScenarioConfig:
    name: str
    arrival_multiplier: float
    se_loss: float
    ec_loss: float
    edge_gateway_availability: float
    edge_compute_availability: float
    cloud_availability: float
    ec_capacity_multiplier: float
    edge_capacity_multiplier: float
    cloud_capacity_multiplier: float
    sensor_energy_budget_multiplier: float
    initial_twin_location: str
    evaluation_event_start: float | None
    evaluation_event_end: float | None
    burst_windows: tuple[dict[str, Any], ...]
    se_outage_windows: tuple[dict[str, Any], ...]
    ec_outage_windows: tuple[dict[str, Any], ...]
    edge_gateway_outage_windows: tuple[dict[str, Any], ...]
    edge_compute_outage_windows: tuple[dict[str, Any], ...]
    cloud_outage_windows: tuple[dict[str, Any], ...]
    sensor_groups: tuple[dict[str, Any], ...]
    resource_windows: tuple[dict[str, Any], ...] = ()
    initial_twin_locations: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class ExperimentConfig:
    simulation: dict[str, Any]
    resources: dict[str, Any]
    energy: dict[str, Any]
    controller: dict[str, Any]
    placement: dict[str, Any]
    service_classes: dict[str, ServiceClassConfig]
    scenarios: dict[str, ScenarioConfig]

    @property
    def slot_seconds(self) -> float:
        return float(self.simulation["slot_seconds"])

    @property
    def monitor_window(self) -> int:
        return int(self.simulation["monitor_window"])

    @property
    def slow_layer_period_slots(self):
        return int(self.simulation["slow_layer_period_slots"])


def load_config(path):
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    services = {}
    for name, item in data.pop("service_classes").items():
        dag = item.pop("dag")
        dag["nodes"] = tuple((DAGNodeConfig(**value) for value in dag["nodes"]))
        dag["edges"] = tuple((DAGEdgeConfig(**value) for value in dag["edges"]))
        services[name] = ServiceClassConfig(
            name=name, dag=DAGTemplateConfig(**dag), **item
        )
    scenarios = {
        name: ScenarioConfig(
            name=name,
            **{
                key: (
                    tuple(value)
                    if key.endswith("_windows") or key == "sensor_groups"
                    else value
                )
                for key, value in item.items()
            },
        )
        for name, item in data.pop("scenarios").items()
    }
    return ExperimentConfig(service_classes=services, scenarios=scenarios, **data)
