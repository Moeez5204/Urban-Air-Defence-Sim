from dataclasses import dataclass
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
import math
import json

import matplotlib.pyplot as plt


@dataclass
class TopologicalContext:
    in_canyon: bool = False
    near_obstacle: bool = False
    in_void: bool = False
    in_radar_shadow: bool = False
    canyon_persistence: float = 0.0
    obstacle_threat: float = 0.0
    shadow_strength: float = 0.0


@dataclass
class FilterSettings:
    time_step: float = 0.1
    state_size: int = 7
    measurement_size: int = 3
    initial_canyon_affinity: float = 0.5
    position_uncertainty: float = 1.0
    velocity_uncertainty: float = 2.0
    affinity_uncertainty: float = 0.01
    measurement_uncertainty: float = 10.0


@dataclass
class TrackingResult:
    timestamp: float
    position: List[float]
    velocity: List[float]
    canyon_affinity: float
    topological_context: str
    filter_confidence: List[float]
    measurement_used: List[float]


class UrbanAwareTracker:


    def __init__(self, settings: FilterSettings = None, city_map_data=None):
        self.settings = settings or FilterSettings()
        self.city_map_data = city_map_data or {}
        self.current_urban_context: Optional[TopologicalContext] = None
        self.tracking_history: List[TrackingResult] = []

        self.total_time: float = 0.0
        self.setup_tracking_system()

    def setup_tracking_system(self):
        self.current_state = np.zeros(self.settings.state_size)
        self.current_state[6] = self.settings.initial_canyon_affinity
        self.movement_predictor = np.eye(self.settings.state_size)
        self.movement_predictor[0, 3] = self.settings.time_step
        self.movement_predictor[1, 4] = self.settings.time_step
        self.movement_predictor[2, 5] = self.settings.time_step

        self.measurement_extractor = np.zeros((self.settings.measurement_size, self.settings.state_size))
        self.measurement_extractor[0, 0] = 1
        self.measurement_extractor[1, 1] = 1
        self.measurement_extractor[2, 2] = 1

        self.confidence_matrix = np.eye(self.settings.state_size) * 100  #  start confident
        self.confidence_matrix[6, 6] = 0.1

        self.base_prediction_uncertainty = self.create_base_uncertainty()
        self.base_measurement_uncertainty = np.eye(
            self.settings.measurement_size) * self.settings.measurement_uncertainty
        self.radar_history: List[np.ndarray] = []

    def create_base_uncertainty(self) -> np.ndarray:
        uncertainty_matrix = np.eye(self.settings.state_size)
        uncertainty_matrix[0:3, 0:3] *= self.settings.position_uncertainty
        uncertainty_matrix[3:6, 3:6] *= self.settings.velocity_uncertainty
        uncertainty_matrix[6, 6] *= self.settings.affinity_uncertainty
        return uncertainty_matrix

    def predict_next_position(self, urban_context: TopologicalContext = None) -> np.ndarray:
        self.current_urban_context = urban_context
        self.update_canyon_behavior_dynamics()
        self.current_state = self.movement_predictor @ self.current_state
        self.confidence_matrix = self.movement_predictor @ self.confidence_matrix @ self.movement_predictor.T + self.get_urban_aware_prediction_uncertainty()
        return self.current_state.copy()

    def update_with_measurement(self, radar_measurement: np.ndarray) -> np.ndarray:
        self.radar_history.append(radar_measurement)

        adjusted_radar_trust = self.get_urban_aware_measurement_uncertainty()

        measurement_error = radar_measurement - self.measurement_extractor @ self.current_state
        innovation_covariance = self.measurement_extractor @ self.confidence_matrix @ self.measurement_extractor.T + adjusted_radar_trust
        kalman_gain = self.confidence_matrix @ self.measurement_extractor.T @ np.linalg.inv(innovation_covariance)

        self.current_state += kalman_gain @ measurement_error
        self.confidence_matrix = (np.eye(
            self.settings.state_size) - kalman_gain @ self.measurement_extractor) @ self.confidence_matrix

        self.learn_canyon_preference_from_behavior(radar_measurement)
        self.save_tracking_snapshot(radar_measurement)
        return self.current_state.copy()

    def save_tracking_snapshot(self, radar_measurement: np.ndarray):
        context_description = self.get_context_description()
        position_confidence = np.diag(self.confidence_matrix)[:3].tolist()
        snapshot = TrackingResult(
            timestamp=self.total_time,
            position=self.current_state[:3].tolist(),
            velocity=self.current_state[3:6].tolist(),
            canyon_affinity=float(self.current_state[6]),
            topological_context=context_description,
            filter_confidence=position_confidence,
            measurement_used=radar_measurement.tolist()
        )
        self.tracking_history.append(snapshot)
        self.total_time += self.settings.time_step

    def get_context_description(self) -> str:
        if not self.current_urban_context:
            return "unknown_area"
        context = self.current_urban_context
        if context.in_canyon:
            return f"canyon_strength_{context.canyon_persistence:.2f}"
        elif context.near_obstacle:
            return f"near_obstacle_danger_{context.obstacle_threat:.2f}"
        elif context.in_void:
            return "open_void_area"
        elif context.in_radar_shadow:
            return f"radar_shadow_{context.shadow_strength:.2f}"
        else:
            return "open_clear_area"

    def update_canyon_behavior_dynamics(self):
        self.movement_predictor[6, 6] = 0.95  # Slowly forget canyon preference (decay toward neutral)

    def get_urban_aware_prediction_uncertainty(self) -> np.ndarray:
        prediction_uncertainty = self.base_prediction_uncertainty.copy()

        if not self.current_urban_context:
            return prediction_uncertainty

        context = self.current_urban_context

        if context.in_canyon:
            canyon_strength = min(1.0, context.canyon_persistence / 1000)
            prediction_uncertainty[0:3, 0:3] *= (0.3 + 0.5 * (1 - canyon_strength))
            prediction_uncertainty[3:6, 3:6] *= (1.0 + 0.3 * canyon_strength)

        if context.near_obstacle:
            prediction_uncertainty *= (1.0 + 0.4 * context.obstacle_threat)


        if context.in_void:
            prediction_uncertainty *= 1.2

        return prediction_uncertainty

    def get_urban_aware_measurement_uncertainty(self) -> np.ndarray:
        radar_uncertainty = self.base_measurement_uncertainty.copy()
        if not self.current_urban_context:
            return radar_uncertainty

        context = self.current_urban_context

        if context.in_radar_shadow:
            radar_uncertainty *= (1.0 + 2.0 * context.shadow_strength)

        if context.in_canyon:
            radar_uncertainty *= 1.5

        return radar_uncertainty

    def learn_canyon_preference_from_behavior(self, radar_measurement: np.ndarray):
        if len(self.radar_history) < 3:
            return

        recent_positions = np.array(self.radar_history[-5:])
        if len(recent_positions) < 3:
            return

        straightness_score = self.calculate_movement_straightness(recent_positions)

        if self.current_urban_context and self.current_urban_context.in_canyon:
            if straightness_score < 0.7:
                self.current_state[6] = min(1.0, self.current_state[6] + 0.1)
            else:
                self.current_state[6] = max(0.0, self.current_state[6] - 0.05)

    def calculate_movement_straightness(self, positions: np.ndarray) -> float:
        if len(positions) < 3:
            return 1.0
        movement_vectors = np.diff(positions, axis=0)
        movement_lengths = np.linalg.norm(movement_vectors, axis=1)
        normalized_directions = movement_vectors / movement_lengths[:, np.newaxis]

        if len(normalized_directions) < 2:
            return 1.0

        direction_similarities = [np.dot(normalized_directions[i], normalized_directions[i + 1])
                                  for i in range(len(normalized_directions) - 1)]

        return np.mean(direction_similarities) if direction_similarities else 1.0

    def analyze_urban_terrain(self, position: np.ndarray) -> TopologicalContext:
        terrain_info = TopologicalContext()

        if not self.city_map_data:
            return terrain_info

        position_2d = position[:2]

        for canyon in self.city_map_data.get('canyons', []):
            if self.is_position_near_canyon_center(position_2d, canyon.get('centerline', [])):
                terrain_info.in_canyon = True
                terrain_info.canyon_persistence = canyon.get('persistence', 0) / 1000
                break

        for obstacle in self.city_map_data.get('obstacles', []):
            if self.is_position_near_obstacle(position_2d, obstacle):
                terrain_info.near_obstacle = True
                terrain_info.obstacle_threat = obstacle.get('threat_score', 0)
                terrain_info.in_radar_shadow = True
                terrain_info.shadow_strength = obstacle.get('concealment_value', 0)
                break

        if not terrain_info.in_canyon and not terrain_info.near_obstacle:
            terrain_info.in_void = True

        return terrain_info

    def is_position_near_canyon_center(self, position_2d: np.ndarray, canyon_centerline: List) -> bool:
        if len(canyon_centerline) < 2:
            return False

        centerline_points = np.array(canyon_centerline)[:, :2]
        distances_to_centerline = np.linalg.norm(centerline_points - position_2d, axis=1)
        return np.min(distances_to_centerline) < 50.0  # Within 50 meters of canyon center

    def is_position_near_obstacle(self, position_2d: np.ndarray, obstacle: Dict) -> bool:
        birth_radius = np.sqrt(obstacle.get('birth', 100))
        death_radius = np.sqrt(obstacle.get('death', 400)) if obstacle.get('death') != float(
            'inf') else birth_radius * 2
        average_obstacle_radius = (birth_radius + death_radius) / 2
        obstacle_center = np.array([obstacle.get('birth', 0), obstacle.get('death', 0)])
        distance_to_obstacle = np.linalg.norm(position_2d - obstacle_center)
        return distance_to_obstacle < average_obstacle_radius * 1.5

    def export_complete_tracking_data(self, filename: str = 'urban_tracking_data.json'):
        results_package = {
            'summary_info': {
                'total_tracking_steps': len(self.tracking_history),
                'final_canyon_preference': float(self.current_state[6]),
                'total_tracking_time': self.total_time,
                'tracker_settings': {
                    'time_step_size': self.settings.time_step,
                    'starting_canyon_guess': self.settings.initial_canyon_affinity
                }
            },
            'complete_tracking_history': [{
                'time': result.timestamp,
                'estimated_position': result.position,
                'estimated_velocity': result.velocity,
                'canyon_preference': result.canyon_affinity,
                'urban_terrain_type': result.topological_context,
                'position_confidence': result.filter_confidence,
                'radar_measurement': result.measurement_used
            } for result in self.tracking_history],
            'performance_stats': {
                'average_canyon_preference': np.mean([r.canyon_affinity for r in self.tracking_history]),
                'different_terrains_encountered': len(set([r.topological_context for r in self.tracking_history])),
                'final_position_confidence': self.tracking_history[
                    -1].filter_confidence if self.tracking_history else []
            }
        }

        with open(filename, 'w') as f:
            json.dump(results_package, f, indent=2)

        print(f"Saved {len(self.tracking_history)} tracking steps to {filename}")
        return results_package


def plot_canyon_preference_learning(preference_history, positions, terrain_types):
    print("\nCreating advanced urban tracking visualization...")

    fig = plt.figure(figsize=(16, 8))
    ax1 = fig.add_subplot(121, projection='3d')
    pos_array = np.array(positions)
    colors = plt.cm.viridis(preference_history)
    for i in range(len(positions) - 1):
        ax1.plot([pos_array[i, 0], pos_array[i + 1, 0]],
                 [pos_array[i, 1], pos_array[i + 1, 1]],
                 [pos_array[i, 2], pos_array[i + 1, 2]],
                 color=colors[i], linewidth=3, alpha=0.8)

    scatter = ax1.scatter(pos_array[:, 0], pos_array[:, 1], pos_array[:, 2],
                          c=preference_history, cmap='viridis', s=100, alpha=0.8)

    ax1.set_xlabel('X Position (m)')
    ax1.set_ylabel('Y Position (m)')
    ax1.set_zlabel('Altitude (m)')
    ax1.set_title('3D Target Trajectory\n(Color = Canyon Preference)')
    plt.colorbar(scatter, ax=ax1, label='Canyon Preference')

    ax2 = fig.add_subplot(122, polar=True)
    metrics = ['Canyon\nUsage', 'Altitude\nStability', 'Speed\nConsistency',
               'Urban\nAdaptation', 'Path\nStraightness']

    canyon_usage = np.mean(preference_history)
    alt_stability = max(0, 1.0 - (np.std(pos_array[:, 2]) / 50))
    speed_consistency = 0.7
    urban_adaptation = min(1.0, len(set(terrain_types)) / 3)
    path_straightness = 0.6

    values = [canyon_usage, alt_stability, speed_consistency, urban_adaptation, path_straightness]
    values += values[:1]

    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]

    ax2.plot(angles, values, 'o-', linewidth=2, label='Target Behavior')
    ax2.fill(angles, values, alpha=0.25)
    ax2.set_xticks(angles[:-1])
    ax2.set_xticklabels(metrics)
    ax2.set_ylim(0, 1)
    ax2.set_title('Behavioral Profile Radar', size=14, pad=20)
    ax2.grid(True)
    plt.suptitle('Topology-Enhanced Kalman Filter: Urban Behavior Analysis', y=0.95)

    plt.tight_layout()
    plt.show()


def demonstrate_urban_tracker():
    urban_map = {
        'canyons': [
            {
                'centerline': [[0, 0, 0], [50, 0, 0], [100, 0, 0], [150, 0, 0], [200, 0, 0]],
                'persistence': 600
            }
        ],
        'obstacles': [
            {'birth': 120, 'death': 180, 'threat_score': 0.8, 'concealment_value': 0.7}
        ]
    }
    settings = FilterSettings(time_step=0.1, initial_canyon_affinity=0.3)
    tracker = UrbanAwareTracker(settings=settings, city_map_data=urban_map)

    target_positions = [
        [10, 5, 50],
        [30, 3, 50],
        [60, 8, 50],
        [90, 6, 50],
        [120, 15, 50],
        [150, 20, 50],
        [180, 25, 50],
        [220, 30, 50],
        [250, 35, 50]
    ]

    preference_history = []
    terrain_history = []

    for step, position in enumerate(target_positions):
        terrain = tracker.analyze_urban_terrain(np.array(position))
        predicted_state = tracker.predict_next_position(urban_context=terrain)
        updated_state = tracker.update_with_measurement(np.array(position))
        preference_history.append(updated_state[6])

        if terrain.in_canyon:
            terrain_type = "Canyon"
        elif terrain.near_obstacle:
            terrain_type = "Obstacle"
        elif terrain.in_void:
            terrain_type = "Void"
        else:
            terrain_type = "Open"
        terrain_history.append(terrain_type)

        print(f"Step {step + 1}: Position={position}, Canyon Preference={updated_state[6]:.3f}")
        active_terrain = [feature for feature, value in terrain.__dict__.items()
                          if value and feature not in ['canyon_persistence', 'obstacle_threat', 'shadow_strength']]
        print(f"Urban Area: {active_terrain}")

    print(f"Final canyon preference: {tracker.current_state[6]:.3f}")

    export_results = tracker.export_complete_tracking_data('urban_tracking_data.json')
    plot_canyon_preference_learning(preference_history, target_positions, terrain_history)

    return tracker


if __name__ == "__main__":
    demonstrate_urban_tracker()