import random
import math
import numpy as np
import time
import queue
import numpy
import os
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional
from AdaptiveSectorDefenseAllocation import SimpleSectorAllocator
import torch
from LTSM import ImprovedLSTMPredictor, BetterTopologyAwareLSTM
from bad_drone_controller import EnemyDrone3D
from bad_drone_controller import EnemyDrone3D

REAL_LSTM_AVAILABLE = True


# data strucutres

@dataclass
class GoodDrone3D:
    # 3D rep of a defensive drone with radar
    id: str
    position: Tuple[float, float, float]
    velocity: Tuple[float, float, float]
    color: Tuple[float, float, float] = (0.0, 0.8, 0.0)  # Green
    size: float = 8.0
    sector: str = "Unknown"

    # radar system
    radar_range: float = 300.0  # meters
    radar_fov: float = 120.0  # degrees
    incoming_queue: queue.Queue = field(default_factory=queue.Queue)
    outgoing_queues: List[queue.Queue] = field(default_factory=list)
    estimated_positions: Dict[str, Tuple[float, float, float]] = field(default_factory=dict)
    radar_targets: List = field(default_factory=list)
    last_message: Optional[str] = None

    # Movement
    patrol_points: List[Tuple[float, float, float]] = field(default_factory=list)
    target_position: Tuple[float, float, float] = (0, 0, 0)
    patrol_index: int = 0


@dataclass
class UrbanFeature:
    id: str
    type: str
    position: Tuple[float, float, float]
    dimensions: Tuple[float, float, float]


class EnemyTrackingHistory:

    def __init__(self, max_history=10):
        self.max_history = max_history
        self.positions = {}
        self.velocities = {}
        self.timestamps = {}
        self.last_update_time = {}

    def add_measurement(self, enemy_id: str, position: Tuple, velocity: Tuple, timestamp: float):

        if enemy_id not in self.positions:
            self.positions[enemy_id] = []
            self.velocities[enemy_id] = []
            self.timestamps[enemy_id] = []

        self.positions[enemy_id].append(position)
        self.velocities[enemy_id].append(velocity)
        self.timestamps[enemy_id].append(timestamp)
        self.last_update_time[enemy_id] = timestamp

        if len(self.positions[enemy_id]) > self.max_history:
            self.positions[enemy_id].pop(0)
            self.velocities[enemy_id].pop(0)
            self.timestamps[enemy_id].pop(0)

    def get_history(self, enemy_id: str) -> Optional[Dict]:
        if enemy_id not in self.positions:
            return None

        return {
            'positions': self.positions[enemy_id],
            'velocities': self.velocities[enemy_id],
            'timestamps': self.timestamps[enemy_id]
        }

    def get_current_state(self, enemy_id: str) -> Optional[Tuple]:
        # Current pos
        if enemy_id not in self.positions or not self.positions[enemy_id]:
            return None

        return (self.positions[enemy_id][-1], self.velocities[enemy_id][-1])

    def clear_old_entries(self, max_age_seconds: float = 30.0):
        current_time = time.time()
        to_remove = []
        for enemy_id, last_time in self.last_update_time.items():
            if current_time - last_time > max_age_seconds:
                to_remove.append(enemy_id)

        for enemy_id in to_remove:
            if enemy_id in self.positions:
                del self.positions[enemy_id]
            if enemy_id in self.velocities:
                del self.velocities[enemy_id]
            if enemy_id in self.timestamps:
                del self.timestamps[enemy_id]
            if enemy_id in self.last_update_time:
                del self.last_update_time[enemy_id]


class LSTMPredictor:
    def __init__(self):
        self.predictor = None
        self.is_ready = False

        if not REAL_LSTM_AVAILABLE:
            print("Real LSTM not available")
            return

        try:
            self.predictor = ImprovedLSTMPredictor()

            # Check if model exists
            if os.path.exists('best_lstm_model.pth') and os.path.exists('lstm_normalization.npz'):
                # Load existing model
                self.predictor.model = BetterTopologyAwareLSTM(input_size=11, output_size=3)
                self.predictor.model.load_state_dict(torch.load('best_lstm_model.pth', map_location='cpu'))
                self.predictor.model.eval()

                # Load normalization params from NPZ file
                norm_data = np.load('lstm_normalization.npz')
                self.predictor.pos_mean = norm_data['pos_mean']
                self.predictor.pos_std = norm_data['pos_std']
                self.predictor.feat_mean = norm_data['feat_mean']
                self.predictor.feat_std = norm_data['feat_std']

                # Also try to load sequence length from JSON if available
                if os.path.exists('best_lstm_model_norm.json'):
                    import json
                    with open('best_lstm_model_norm.json', 'r') as f:
                        json_data = json.load(f)
                        self.predictor.sequence_length = json_data.get('sequence_length', 5)

                self.is_ready = True
                print("Loaded pre-trained LSTM model from npz file")
            else:
                print("Model files not found. Run LTSM.py first to train the model.")

                return

        except Exception as e:
            print(f"⚠ Error initializing LSTM: {e}")
            self.is_ready = False

        except Exception as e:
            print(f"⚠ Error initializing LSTM: {e}")
            self.is_ready = False

    def predict_enemy_movement(self, enemy_id: str, current_pos: Tuple, current_vel: Tuple,
                               history: Dict = None, timestamp: float = 0) -> Tuple:

        if not self.is_ready or self.predictor is None:
            # Fallback
            x, y, z = current_pos
            vx, vy, vz = current_vel
            return (x + vx * 2.0, y + vy * 2.0, z + vz * 2.0)

        try:

            canyon_affinity = 0.5  # Default, could be learned
            topo_features = self.predictor.context_mapper.get_current_features(current_pos, timestamp)
            features = np.array([
                current_pos[0], current_pos[1], current_pos[2],
                current_vel[0], current_vel[1], current_vel[2],
                canyon_affinity,
                topo_features.normalized_persistence,
                topo_features.distance_to_centerline / 100.0,
                topo_features.threat_level,
                topo_features.concealment_value
            ])

            # Normalize features
            if hasattr(self.predictor, 'feat_mean') and hasattr(self.predictor, 'feat_std'):
                features = (features - self.predictor.feat_mean) / self.predictor.feat_std

            sequence = [features] * 5
            input_tensor = torch.FloatTensor([sequence])

            # Predict
            with torch.no_grad():
                normalized_pred = self.predictor.model(input_tensor)

                # denormaliza
                if hasattr(self.predictor, 'pos_mean') and hasattr(self.predictor, 'pos_std'):
                    predicted_pos = normalized_pred.numpy().flatten() * self.predictor.pos_std + self.predictor.pos_mean
                else:
                    predicted_pos = normalized_pred.numpy().flatten()

            return tuple(predicted_pos)

        except Exception as e:
            # fallback
            x, y, z = current_pos
            vx, vy, vz = current_vel
            return (x + vx * 2.0, y + vy * 2.0, z + vz * 2.0)

    def is_available(self) -> bool:
        return self.is_ready


# Urban radar system

class UrbanDroneRadar:

    def __init__(self, drone):
        self.drone = drone
        self.max_range = drone.radar_range
        self.fov = drone.radar_fov
        self.update_rate = 0.1
        self.radar_pulse_radii = []
        self.pulse_timer = random.uniform(0, 2)
        self.reflect_timers = {}
        self.frame_count = 0
        self.detection_history = {}

    def update(self, dt, enemies, urban_features):
        # update radar pulses
        self.pulse_timer -= dt
        if self.pulse_timer <= 0:
            self.radar_pulse_radii.append(0)
            self.pulse_timer = 0.5

        self.radar_pulse_radii = [r + 200 * dt for r in self.radar_pulse_radii if r + 200 * dt < self.max_range]
        detections = self.get_detections(enemies, urban_features)

        for enemy_id in detections:
            self.reflect_timers[enemy_id] = 0.5

        self.frame_count += 1
        return detections

    def get_detections(self, enemies, urban_features):
        detections = {}

        for enemy in enemies:
            if self._has_line_of_sight(enemy.position, urban_features):
                distance = self._calculate_distance(enemy.position)

                if self._in_radar_fov(enemy.position) and distance < self.max_range:
                    urban_error = self._calculate_urban_error(enemy.position, urban_features)
                    confidence = self._calculate_confidence(enemy.position, urban_features)

                    detections[enemy.id] = {
                        'distance': distance + urban_error,
                        'position': enemy.position,
                        'confidence': confidence,
                        'timestamp': time.time()
                    }

        return detections

    def _has_line_of_sight(self, target_pos, urban_features):
        drone_pos = self.drone.position

        for feature in urban_features:
            if self._ray_intersects_feature(drone_pos, target_pos, feature):
                return False
        return True

    def _ray_intersects_feature(self, start, end, feature):
        fx, fy, fz = feature.position
        fw, fl, fh = feature.dimensions

        if (max(start[0], end[0]) < fx - fw / 2 or min(start[0], end[0]) > fx + fw / 2 or
                max(start[1], end[1]) < fy - fl / 2 or min(start[1], end[1]) > fy + fl / 2 or
                max(start[2], end[2]) < fz - fh / 2 or min(start[2], end[2]) > fz + fh / 2):
            return False

        return True

    def _calculate_distance(self, target_pos):
        dx = target_pos[0] - self.drone.position[0]
        dy = target_pos[1] - self.drone.position[1]
        dz = target_pos[2] - self.drone.position[2]
        return math.sqrt(dx * dx + dy * dy + dz * dz)

    def _in_radar_fov(self, target_pos):
        return True

    def _calculate_urban_error(self, target_pos, urban_features):
        error = 0.0

        for feature in urban_features:
            if feature.type == "building":
                distance_to_feature = self._calculate_distance(feature.position)
                if distance_to_feature < 100:
                    error += random.uniform(-5.0, 5.0)

        error += random.uniform(-2.0, 2.0)
        return error

    def _calculate_confidence(self, target_pos, urban_features):
        confidence = 1.0

        for feature in urban_features:
            if self._ray_near_feature(self.drone.position, target_pos, feature):
                if feature.type == "building":
                    confidence *= 0.7
                elif feature.type == "canyon_wall":
                    confidence *= 0.8

        return max(0.1, min(1.0, confidence))

    def _ray_near_feature(self, start, end, feature, threshold=20.0):
        fx, fy, fz = feature.position

        line_dir = (end[0] - start[0], end[1] - start[1], end[2] - start[2])
        line_length = math.sqrt(line_dir[0] ** 2 + line_dir[1] ** 2 + line_dir[2] ** 2)

        if line_length == 0:
            return False

        t = ((fx - start[0]) * line_dir[0] + (fy - start[1]) * line_dir[1] + (fz - start[2]) * line_dir[2]) / (
                line_length ** 2)
        t = max(0, min(1, t))

        closest_x = start[0] + t * line_dir[0]
        closest_y = start[1] + t * line_dir[1]
        closest_z = start[2] + t * line_dir[2]

        distance = math.sqrt((fx - closest_x) ** 2 + (fy - closest_y) ** 2 + (fz - closest_z) ** 2)
        return distance < threshold


# Communication Network

class DroneCommunicationNetwork:
    def __init__(self, communication_interval=2.0):
        self.drones = []
        self.communication_interval = communication_interval
        self.last_communication = 0
        self.shared_detections = {}

    def add_drone(self, drone):
        self.drones.append(drone)

        for other_drone in self.drones:
            if other_drone != drone:
                drone.outgoing_queues.append(other_drone.incoming_queue)

    def update(self, current_time, enemies, urban_features):
        individual_detections = {}
        for drone in self.drones:
            if hasattr(drone, 'radar_system'):
                detections = drone.radar_system.update(0.016, enemies, urban_features)
                individual_detections[drone.id] = detections

        if current_time - self.last_communication >= self.communication_interval:
            self._share_detections(individual_detections)
            self._fuse_detections()
            self._triangulate_positions()
            self.last_communication = current_time

            for drone in self.drones:
                drone.estimated_positions = self.shared_detections.get(drone.id, {})

    def _share_detections(self, individual_detections):
        for drone_id, detections in individual_detections.items():
            message = {
                "sender": drone_id,
                "pos": self._get_drone_position(drone_id),
                "detections": detections,
                "timestamp": time.time()
            }

            for drone in self.drones:
                if drone.id != drone_id:
                    drone.incoming_queue.put(message)

    def _fuse_detections(self):
        fused = {}

        for drone in self.drones:
            while not drone.incoming_queue.empty():
                try:
                    msg = drone.incoming_queue.get_nowait()
                    drone.last_message = f"From {msg['sender']}: {len(msg['detections'])} detections"

                    for target_id, detection in msg['detections'].items():
                        if target_id not in fused:
                            fused[target_id] = []
                        fused[target_id].append({
                            'position': detection['position'],
                            'distance': detection['distance'],
                            'confidence': detection['confidence'],
                            'sender': msg['sender']
                        })

                except queue.Empty:
                    break

        self.shared_detections = {}
        for target_id, measurements in fused.items():
            if len(measurements) >= 2:
                total_weight = sum(m['confidence'] for m in measurements)
                if total_weight > 0:
                    avg_x = sum(m['position'][0] * m['confidence'] for m in measurements) / total_weight
                    avg_y = sum(m['position'][1] * m['confidence'] for m in measurements) / total_weight
                    avg_z = sum(m['position'][2] * m['confidence'] for m in measurements) / total_weight

                    for drone in self.drones:
                        if drone.id not in self.shared_detections:
                            self.shared_detections[drone.id] = {}
                        self.shared_detections[drone.id][target_id] = (avg_x, avg_y, avg_z)

    def _triangulate_positions(self):
        for drone in self.drones:
            if not drone.estimated_positions:
                continue

            for target_id, estimated_pos in drone.estimated_positions.items():
                other_measurements = []
                for other_drone in self.drones:
                    if other_drone.id != drone.id and target_id in other_drone.estimated_positions:
                        other_measurements.append(other_drone.estimated_positions[target_id])

                if len(other_measurements) >= 1:
                    all_positions = [estimated_pos] + other_measurements
                    avg_x = sum(p[0] for p in all_positions) / len(all_positions)
                    avg_y = sum(p[1] for p in all_positions) / len(all_positions)
                    avg_z = sum(p[2] for p in all_positions) / len(all_positions)
                    drone.estimated_positions[target_id] = (avg_x, avg_y, avg_z)

    def _get_drone_position(self, drone_id):
        for drone in self.drones:
            if drone.id == drone_id:
                return drone.position
        return (0, 0, 0)



class GoodDroneController:
    def __init__(self):

        self.drones: List[GoodDrone3D] = []
        self.enemies: List[EnemyDrone3D] = []  # Enemies will be provided externally
        self.urban_features: List[UrbanFeature] = []
        self.radar_network = DroneCommunicationNetwork()
        self.lstm_predictor = LSTMPredictor()



        self.tracking_history = EnemyTrackingHistory(max_history=10)

        sectors_config = [
            {"name": "Walled_City", "base_priority": 0.9, "assets": []},
            {"name": "Central_Lahore", "base_priority": 1.0, "assets": []},
            {"name": "Gulberg", "base_priority": 0.8, "assets": []},
            {"name": "Cantonment", "base_priority": 0.7, "assets": []},
            {"name": "Other_Sector", "base_priority": 0.6, "assets": []},
        ]
        self.adaptive_sector = SimpleSectorAllocator(sectors_config)

        self.current_time = 0.0
        self.map_bounds = {
            'x_min': -600, 'x_max': 600,
            'y_min': -600, 'y_max': 600,
            'z_min': 30, 'z_max': 400
        }

        self.enemy_predictions = {}

    def set_enemies(self, enemies):
        #Set enemies from controller
        self.enemies = enemies

    def initialize_drones(self, num_drones: int = 8):

        print(f"\nInitializing {num_drones} integrated defense drones")

        self.drones = []
        self._generate_urban_features()

        enemy_positions = [e.position for e in self.enemies] if self.enemies else []

        # Use ASDA for allocation
        allocations = self.adaptive_sector.get_resources_allocation(num_drones, epsilon=0.1)

        # Sector centers for drone placement
        sector_centers = {
            'Walled_City': (-200, 100, 100),
            'Central_Lahore': (0, 0, 120),
            'Gulberg': (150, -50, 110),
            'Cantonment': (-100, 150, 90),
            'Other_Sector': (100, 200, 100)
        }

        sector_radii = {
            'Walled_City': 200,
            'Central_Lahore': 250,
            'Gulberg': 180,
            'Cantonment': 220,
            'Other_Sector': 200
        }

        drone_id = 0
        for sector_name, drone_count in allocations.items():
            for i in range(drone_count):
                if drone_id >= num_drones:
                    break

                center = sector_centers.get(sector_name, (0, 0, 100))
                radius = sector_radii.get(sector_name, 200)

                angle = random.uniform(0, 2 * math.pi)
                distance = random.uniform(0, radius * 0.6)

                x = center[0] + distance * math.cos(angle)
                y = center[1] + distance * math.sin(angle)
                z = center[2] + random.uniform(-40, 40)

                drone = GoodDrone3D(
                    id=f"Drone_{drone_id:02d}",
                    position=(x, y, z),
                    velocity=(0.0, 0.0, 0.0),
                    sector=sector_name
                )

                drone.radar_system = UrbanDroneRadar(drone)
                drone.radar_targets = self.enemies
                self._create_patrol_path(drone, center, radius)

                self.drones.append(drone)
                self.radar_network.add_drone(drone)

                drone_id += 1

        print(f" {len(self.drones)} drones initialized")
        return self.drones

    def _generate_urban_features(self):
        self.urban_features = []

        for i in range(20):
            x = random.uniform(-500, 500)
            y = random.uniform(-500, 500)
            z = 0
            width = random.uniform(20, 60)
            length = random.uniform(20, 60)
            height = random.uniform(30, 120)

            self.urban_features.append(UrbanFeature(
                id=f"Building_{i:02d}",
                type="building",
                position=(x, y, z),
                dimensions=(width, length, height)
            ))

        print(f"Generated {len(self.urban_features)} urban features")

    def _create_patrol_path(self, drone, center, radius):
        cx, cy, cz = center
        patrol_points = []

        for i in range(6):
            angle = (i / 6) * 2 * math.pi
            x = cx + radius * 0.7 * math.cos(angle)
            y = cy + radius * 0.7 * math.sin(angle)
            z = cz + random.uniform(-30, 30)
            patrol_points.append((x, y, z))

        drone.patrol_points = patrol_points
        drone.target_position = patrol_points[0]

    def _update_enemy_predictions(self):
       #Update LTSM
        if not self.enemies:
            return

        current_timestamp = self.current_time

        for enemy in self.enemies:
            history = self.tracking_history.get_history(enemy.id)

            # Get current velocity
            if history and len(history['velocities']) > 0:
                current_vel = history['velocities'][-1]
            else:
                current_vel = enemy.velocity

            # Make prediction using real LSTM
            predicted_pos = self.lstm_predictor.predict_enemy_movement(
                enemy.id,
                enemy.position,
                current_vel,
                history,
                current_timestamp
            )

            self.enemy_predictions[enemy.id] = predicted_pos

    def update_drones(self, delta_time: float = 0.016):
        """Update complete drone system with LSTM predictions"""
        if not self.drones or not self.enemies:
            return

        self.current_time += delta_time
        self.radar_network.update(self.current_time, self.enemies, self.urban_features)
        for enemy in self.enemies:
            self.tracking_history.add_measurement(
                enemy.id,
                enemy.position,
                enemy.velocity,
                self.current_time
            )

        self._update_enemy_predictions()
        for drone in self.drones:
            self._update_single_drone(drone, delta_time)

        threat_positions = [e.position for e in self.enemies]
        sector_centers = {
            'Walled_City': (-200, 100),
            'Central_Lahore': (0, 0),
            'Gulberg': (150, -50),
            'Cantonment': (-100, 150),
            'Other_Sector': (100, 200)
        }

        sector_threats = {sector: 0 for sector in sector_centers.keys()}
        for threat_pos in threat_positions:
            closest_sector = None
            min_distance = float('inf')
            for sector_name, center in sector_centers.items():
                distance = math.sqrt((threat_pos[0] - center[0]) ** 2 + (threat_pos[1] - center[1]) ** 2)
                if distance < min_distance and distance < 300:
                    min_distance = distance
                    closest_sector = sector_name
            if closest_sector:
                sector_threats[closest_sector] += 1

        threat_data = [{'asset_sector': sector, 'overall_score': min(10, count * 2)}
                       for sector, count in sector_threats.items() if count > 0]
        if threat_data:
            self.adaptive_sector.update_from_threat_assessment(threat_data)

    def _update_single_drone(self, drone, delta_time):
        #Update single drone's movement
        tx, ty, tz = drone.target_position
        dx = tx - drone.position[0]
        dy = ty - drone.position[1]
        dz = tz - drone.position[2]

        distance = math.sqrt(dx * dx + dy * dy + dz * dz)

        if distance < 30:
            drone.patrol_index = (drone.patrol_index + 1) % len(drone.patrol_points)
            drone.target_position = drone.patrol_points[drone.patrol_index]

            tx, ty, tz = drone.target_position
            dx = tx - drone.position[0]
            dy = ty - drone.position[1]
            dz = tz - drone.position[2]
            distance = math.sqrt(dx * dx + dy * dy + dz * dz)

        if distance > 0:
            speed = 80.0
            drone.velocity = (
                (dx / distance) * speed,
                (dy / distance) * speed,
                (dz / distance) * speed * 0.5
            )

        new_x = drone.position[0] + drone.velocity[0] * delta_time
        new_y = drone.position[1] + drone.velocity[1] * delta_time
        new_z = drone.position[2] + drone.velocity[2] * delta_time

        drone.position = (new_x, new_y, new_z)

    def get_detection_data(self):
        detections = []

        for drone in self.drones:
            if hasattr(drone, 'estimated_positions'):
                for target_id, pos in drone.estimated_positions.items():
                    detections.append({
                        'drone_id': drone.id,
                        'target_id': target_id,
                        'position': pos,
                        'confidence': 0.8
                    })

        return detections

    def get_predictions(self):
        return self.enemy_predictions

    def get_lstm_status(self):
        return {
            'using_real_lstm': self.lstm_predictor.is_available(),
            'tracked_enemies': len(self.tracking_history.positions),
            'active_predictions': len(self.enemy_predictions)
        }


# TEST FUNCTION

def test_integrated_system():

    controller = GoodDroneController()

    @dataclass
    class TestEnemy:
        id: str
        position: Tuple[float, float, float]
        velocity: Tuple[float, float, float]
        color: Tuple[float, float, float] = (1.0, 0.0, 0.0)
        size: float = 8.0

    test_enemies = [
        TestEnemy("Enemy_01", (100, 100, 150), (20, 10, 5)),
        TestEnemy("Enemy_02", (-100, -50, 120), (-15, 20, -3)),
        TestEnemy("Enemy_03", (50, -100, 180), (10, -20, 2)),
    ]
    controller.set_enemies(test_enemies)

    controller.initialize_drones(num_drones=4)

    print(f"System Status:")
    print(f"  Drones: {len(controller.drones)}")
    print(f"  Enemies: {len(controller.enemies)}")
    print(f"  Urban Features: {len(controller.urban_features)}")

    lstm_status = controller.get_lstm_status()
    print(f"LSTM Status:")
    print(f"  Using Real LSTM: {lstm_status['using_real_lstm']}")
    print(f"  Tracked Enemies: {lstm_status['tracked_enemies']}")

    # Simulate
    for i in range(10):
        controller.update_drones()
        if i % 2 == 0:
            print(f"Frame {i + 1}: Updated ")

    # Show predictions
    predictions = controller.get_predictions()
    print(f"\nLSTM Predictions: {len(predictions)} enemies")
    for enemy_id, pred_pos in predictions.items():
        print(f"  {enemy_id}: ({pred_pos[0]:.1f}, {pred_pos[1]:.1f}, {pred_pos[2]:.1f})")

    return controller


if __name__ == "__main__":
    controller = test_integrated_system()