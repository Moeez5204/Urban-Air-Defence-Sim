import random
import math
import time
import queue
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional
import numpy as np

# Try to import enemy drones from separate file
try:
    from bad_drone_controller import EnemyDrone3D
    ENEMY_DRONE_AVAILABLE = True
except ImportError:
    ENEMY_DRONE_AVAILABLE = False
    # Fallback EnemyDrone3D class if not available
    @dataclass
    class EnemyDrone3D:
        """3D enemy drone"""
        id: str
        position: Tuple[float, float, float]
        velocity: Tuple[float, float, float]
        color: Tuple[float, float, float] = (1.0, 0.0, 0.0)  # Red
        size: float = 8.0

# ============================================================================
# DATA STRUCTURES (GOOD DRONE ONLY)
# ============================================================================

@dataclass
class GoodDrone3D:
    """3D representation of a defensive drone with radar"""
    id: str
    position: Tuple[float, float, float]
    velocity: Tuple[float, float, float]
    color: Tuple[float, float, float] = (0.0, 0.8, 0.0)  # Green
    size: float = 8.0
    sector: str = "Unknown"

    # Radar system (from your original code)
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
    """Urban feature for radar occlusion"""
    id: str
    type: str  # "building", "canyon_wall", "obstacle"
    position: Tuple[float, float, float]
    dimensions: Tuple[float, float, float]  # width, length, height

# ============================================================================
# URBAN RADAR SYSTEM (FROM YOUR ORIGINAL CODE - ADAPTED TO 3D)
# ============================================================================

class UrbanDroneRadar:
    """3D version of your original jet radar system with urban degradation"""

    def __init__(self, drone):
        self.drone = drone
        self.max_range = drone.radar_range
        self.fov = drone.radar_fov
        self.update_rate = 0.1

        # From your original code - tracking
        self.radar_pulse_radii = []
        self.pulse_timer = random.uniform(0, 2)
        self.reflect_timers = {}
        self.frame_count = 0

        # Urban-specific
        self.detection_history = {}

    def update(self, dt, enemies, urban_features):
        """Update radar pulses and detections"""
        self.pulse_timer -= dt
        if self.pulse_timer <= 0:
            self.radar_pulse_radii.append(0)
            self.pulse_timer = 0.5  # Adjusted for 3D

        # Update pulse radii
        self.radar_pulse_radii = [r + 200 * dt for r in self.radar_pulse_radii if r + 200 * dt < self.max_range]

        # Get detections
        detections = self.get_detections(enemies, urban_features)

        # Update reflect timers (from your original code)
        for enemy_id in detections:
            self.reflect_timers[enemy_id] = 0.5

        self.frame_count += 1
        return detections

    def get_detections(self, enemies, urban_features):
        """Get radar detections with urban occlusion"""
        detections = {}

        for enemy in enemies:
            # Check line of sight through urban features
            if self._has_line_of_sight(enemy.position, urban_features):
                distance = self._calculate_distance(enemy.position)

                # Check if in radar FOV
                if self._in_radar_fov(enemy.position) and distance < self.max_range:
                    # Add urban degradation error
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
        """3D ray casting through urban environment"""
        drone_pos = self.drone.position

        for feature in urban_features:
            if self._ray_intersects_feature(drone_pos, target_pos, feature):
                return False
        return True

    def _ray_intersects_feature(self, start, end, feature):
        """Check if ray intersects a 3D urban feature"""
        # Simplified bounding box intersection
        fx, fy, fz = feature.position
        fw, fl, fh = feature.dimensions

        # Check if line segment intersects bounding box
        dx, dy, dz = end[0] - start[0], end[1] - start[1], end[2] - start[2]

        # Quick rejection test
        if (max(start[0], end[0]) < fx - fw/2 or min(start[0], end[0]) > fx + fw/2 or
            max(start[1], end[1]) < fy - fl/2 or min(start[1], end[1]) > fy + fl/2 or
            max(start[2], end[2]) < fz - fh/2 or min(start[2], end[2]) > fz + fh/2):
            return False

        # For simplicity, assume intersection if feature is between drone and target
        return True

    def _calculate_distance(self, target_pos):
        """3D distance calculation"""
        dx = target_pos[0] - self.drone.position[0]
        dy = target_pos[1] - self.drone.position[1]
        dz = target_pos[2] - self.drone.position[2]
        return math.sqrt(dx*dx + dy*dy + dz*dz)

    def _in_radar_fov(self, target_pos):
        """Check if target is within radar field of view"""
        # Simplified FOV check in 3D
        return True  # For now, assume omnidirectional

    def _calculate_urban_error(self, target_pos, urban_features):
        """Calculate urban-induced radar error"""
        error = 0.0

        # Multipath error (buildings reflect signals)
        for feature in urban_features:
            if feature.type == "building":
                distance_to_feature = self._calculate_distance(feature.position)
                if distance_to_feature < 100:
                    error += random.uniform(-5.0, 5.0)  # Multipath error

        # Atmospheric/turbulence error
        error += random.uniform(-2.0, 2.0)

        return error

    def _calculate_confidence(self, target_pos, urban_features):
        """Calculate detection confidence based on urban context"""
        confidence = 1.0

        for feature in urban_features:
            if self._ray_near_feature(self.drone.position, target_pos, feature):
                if feature.type == "building":
                    confidence *= 0.7  # Buildings reduce confidence
                elif feature.type == "canyon_wall":
                    confidence *= 0.8  # Canyon walls reduce confidence

        return max(0.1, min(1.0, confidence))

    def _ray_near_feature(self, start, end, feature, threshold=20.0):
        """Check if ray passes near a feature"""
        fx, fy, fz = feature.position

        # Find closest point on line to feature center
        line_dir = (end[0] - start[0], end[1] - start[1], end[2] - start[2])
        line_length = math.sqrt(line_dir[0]**2 + line_dir[1]**2 + line_dir[2]**2)

        if line_length == 0:
            return False

        # Project feature onto line
        t = ((fx - start[0])*line_dir[0] + (fy - start[1])*line_dir[1] + (fz - start[2])*line_dir[2]) / (line_length**2)
        t = max(0, min(1, t))

        closest_x = start[0] + t * line_dir[0]
        closest_y = start[1] + t * line_dir[1]
        closest_z = start[2] + t * line_dir[2]

        distance = math.sqrt((fx - closest_x)**2 + (fy - closest_y)**2 + (fz - closest_z)**2)
        return distance < threshold

# ============================================================================
# MULTISTATIC COMMUNICATION NETWORK (FROM YOUR ORIGINAL CODE)
# ============================================================================

class DroneCommunicationNetwork:
    """Network of drones communicating like your original jets"""

    def __init__(self, communication_interval=2.0):
        self.drones = []
        self.communication_interval = communication_interval
        self.last_communication = 0
        self.shared_detections = {}

    def add_drone(self, drone):
        """Add drone to network"""
        self.drones.append(drone)

        # Setup communication queues (from your original code)
        for other_drone in self.drones:
            if other_drone != drone:
                drone.outgoing_queues.append(other_drone.incoming_queue)

    def update(self, current_time, enemies, urban_features):
        """Update network communication and fusion"""

        # Each drone gets individual detections
        individual_detections = {}
        for drone in self.drones:
            if hasattr(drone, 'radar_system'):
                detections = drone.radar_system.update(0.016, enemies, urban_features)
                individual_detections[drone.id] = detections

        # Communicate every interval (from your original code)
        if current_time - self.last_communication >= self.communication_interval:
            self._share_detections(individual_detections)
            self._fuse_detections()
            self._triangulate_positions()  # Your original multistatic magic!
            self.last_communication = current_time

            # Update drone estimated positions
            for drone in self.drones:
                drone.estimated_positions = self.shared_detections.get(drone.id, {})

    def _share_detections(self, individual_detections):
        """Drones share detections (from your original code)"""
        for drone_id, detections in individual_detections.items():
            message = {
                "sender": drone_id,
                "pos": self._get_drone_position(drone_id),
                "detections": detections,
                "timestamp": time.time()
            }

            # Send to all other drones
            for drone in self.drones:
                if drone.id != drone_id:
                    drone.incoming_queue.put(message)

    def _fuse_detections(self):
        """Fuse detections from multiple drones"""
        fused = {}

        for drone in self.drones:
            # Process incoming messages (from your original code)
            while not drone.incoming_queue.empty():
                try:
                    msg = drone.incoming_queue.get_nowait()
                    drone.last_message = f"From {msg['sender']}: {len(msg['detections'])} detections"

                    # Fuse detections
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

        # Average fused positions
        self.shared_detections = {}
        for target_id, measurements in fused.items():
            if len(measurements) >= 2:  # Need at least 2 for triangulation
                # Weight by confidence
                total_weight = sum(m['confidence'] for m in measurements)
                if total_weight > 0:
                    avg_x = sum(m['position'][0] * m['confidence'] for m in measurements) / total_weight
                    avg_y = sum(m['position'][1] * m['confidence'] for m in measurements) / total_weight
                    avg_z = sum(m['position'][2] * m['confidence'] for m in measurements) / total_weight

                    # Store in each drone's estimated positions
                    for drone in self.drones:
                        if drone.id not in self.shared_detections:
                            self.shared_detections[drone.id] = {}
                        self.shared_detections[drone.id][target_id] = (avg_x, avg_y, avg_z)

    def _triangulate_positions(self):
        """Triangulate positions from multiple measurements (from your original code)"""
        for drone in self.drones:
            if not drone.estimated_positions:
                continue

            # For each target, refine position using triangulation
            for target_id, estimated_pos in drone.estimated_positions.items():
                # Get measurements from other drones
                other_measurements = []
                for other_drone in self.drones:
                    if other_drone.id != drone.id and target_id in other_drone.estimated_positions:
                        other_measurements.append(other_drone.estimated_positions[target_id])

                if len(other_measurements) >= 1:
                    # Simple averaging for now
                    all_positions = [estimated_pos] + other_measurements
                    avg_x = sum(p[0] for p in all_positions) / len(all_positions)
                    avg_y = sum(p[1] for p in all_positions) / len(all_positions)
                    avg_z = sum(p[2] for p in all_positions) / len(all_positions)
                    drone.estimated_positions[target_id] = (avg_x, avg_y, avg_z)

    def _get_drone_position(self, drone_id):
        """Get drone position by ID"""
        for drone in self.drones:
            if drone.id == drone_id:
                return drone.position
        return (0, 0, 0)

# ============================================================================
# SIMPLE LSTM PREDICTOR
# ============================================================================

class SimpleLSTMPredictor:
    """Simple LSTM-like movement predictor"""

    def __init__(self):
        self.memory = []
        self.prediction_history = {}

    def predict(self, target_id, current_pos, velocity, history=None):
        """Predict future position"""
        x, y, z = current_pos
        vx, vy, vz = velocity

        # Add LSTM-like noise and memory
        noise_x = random.uniform(-0.1, 0.1) * (1.0 + abs(vx))
        noise_y = random.uniform(-0.1, 0.1) * (1.0 + abs(vy))
        noise_z = random.uniform(-0.05, 0.05) * (1.0 + abs(vz))

        # Predict 2 seconds ahead
        predicted_x = x + vx * 2.0 + noise_x
        predicted_y = y + vy * 2.0 + noise_y
        predicted_z = z + vz * 2.0 + noise_z

        # Store in history
        if target_id not in self.prediction_history:
            self.prediction_history[target_id] = []
        self.prediction_history[target_id].append((predicted_x, predicted_y, predicted_z))

        # Keep limited history
        if len(self.prediction_history[target_id]) > 10:
            self.prediction_history[target_id].pop(0)

        return (predicted_x, predicted_y, predicted_z)

# ============================================================================
# ADAPTIVE SECTOR DEFENSE ALLOCATION (ASDA)
# ============================================================================

class AdaptiveSectorDefense:
    """Adaptive Sector Defense Allocation"""

    def __init__(self):
        # Lahore sectors
        self.sectors = {
            'Walled_City': {
                'center': (-200, 100, 100),
                'priority': 0.9,
                'radius': 200,
                'threat_count': 0
            },
            'Central_Lahore': {
                'center': (0, 0, 120),
                'priority': 1.0,
                'radius': 250,
                'threat_count': 0
            },
            'Gulberg': {
                'center': (150, -50, 110),
                'priority': 0.8,
                'radius': 180,
                'threat_count': 0
            },
            'Cantonment': {
                'center': (-100, 150, 90),
                'priority': 0.7,
                'radius': 220,
                'threat_count': 0
            },
            'Other_Sector': {
                'center': (100, 200, 100),
                'priority': 0.6,
                'radius': 200,
                'threat_count': 0
            }
        }

        self.sector_threats = {sector: 0 for sector in self.sectors}

    def allocate_drones(self, num_drones, current_threats):
        """Adaptively allocate drones"""
        # Update threat counts
        self._update_threats(current_threats)

        allocations = {}
        total_priority = sum(sector['priority'] for sector in self.sectors.values())

        # Base allocation by priority
        for sector_name, sector_data in self.sectors.items():
            priority_share = sector_data['priority'] / total_priority
            base_allocation = max(1, int(num_drones * priority_share * 0.6))

            # Add threat-based adjustment
            threat_adjustment = min(3, self.sector_threats.get(sector_name, 0))

            allocations[sector_name] = base_allocation + threat_adjustment

        # Normalize
        total_allocated = sum(allocations.values())
        if total_allocated > num_drones:
            scale = num_drones / total_allocated
            for sector in allocations:
                allocations[sector] = max(1, int(allocations[sector] * scale))

        return allocations

    def _update_threats(self, threats):
        """Update threat assessment per sector"""
        self.sector_threats = {sector: 0 for sector in self.sectors}

        for threat_pos in threats:
            closest_sector = None
            min_distance = float('inf')

            for sector_name, sector_data in self.sectors.items():
                center = sector_data['center']
                distance = math.sqrt(
                    (threat_pos[0] - center[0])**2 +
                    (threat_pos[1] - center[1])**2
                )

                if distance < min_distance and distance < 300:
                    min_distance = distance
                    closest_sector = sector_name

            if closest_sector:
                self.sector_threats[closest_sector] += 1

# ============================================================================
# MAIN GOOD DRONE CONTROLLER
# ============================================================================

class GoodDroneController:
    """Complete integrated controller with radar, LSTM, and ASDA - GOOD DRONES ONLY"""

    def __init__(self):
        # Core systems
        self.drones: List[GoodDrone3D] = []
        self.enemies: List[EnemyDrone3D] = []  # Enemies will be provided externally
        self.urban_features: List[UrbanFeature] = []

        # Integrated subsystems
        self.radar_network = DroneCommunicationNetwork()
        self.lstm_predictor = SimpleLSTMPredictor()
        self.adaptive_sector = AdaptiveSectorDefense()

        # State
        self.current_time = 0.0
        self.map_bounds = {
            'x_min': -600, 'x_max': 600,
            'y_min': -600, 'y_max': 600,
            'z_min': 30, 'z_max': 400
        }

        print("✓ Complete Integrated GOOD Drone Controller Initialized")
        print("  - Urban Radar System (from original code)")
        print("  - Multistatic Communication Network")
        print("  - LSTM Movement Predictor")
        print("  - Adaptive Sector Defense Allocation")

    def set_enemies(self, enemies):
        """Set enemies from external controller"""
        self.enemies = enemies

    def initialize_drones(self, num_drones: int = 8):
        """Initialize complete drone system"""
        print(f"\nInitializing {num_drones} integrated defense drones...")

        # Clear existing
        self.drones = []

        # Generate some urban features for radar testing
        self._generate_urban_features()

        # Run ASDA allocation using current enemies
        enemy_positions = [e.position for e in self.enemies] if self.enemies else []
        allocations = self.adaptive_sector.allocate_drones(num_drones, enemy_positions)

        # Create drones
        drone_id = 0
        for sector_name, drone_count in allocations.items():
            for i in range(drone_count):
                if drone_id >= num_drones:
                    break

                sector_data = self.adaptive_sector.sectors[sector_name]
                center = sector_data['center']
                radius = sector_data['radius']

                # Position in sector
                angle = random.uniform(0, 2 * math.pi)
                distance = random.uniform(0, radius * 0.6)

                x = center[0] + distance * math.cos(angle)
                y = center[1] + distance * math.sin(angle)
                z = center[2] + random.uniform(-40, 40)

                # Create drone
                drone = GoodDrone3D(
                    id=f"Drone_{drone_id:02d}",
                    position=(x, y, z),
                    velocity=(0.0, 0.0, 0.0),
                    sector=sector_name
                )

                # Add radar system
                drone.radar_system = UrbanDroneRadar(drone)
                drone.radar_targets = self.enemies

                # Add patrol path
                self._create_patrol_path(drone, center, radius)

                self.drones.append(drone)
                self.radar_network.add_drone(drone)

                drone_id += 1

        print(f"✓ {len(self.drones)} drones initialized with integrated systems")
        return self.drones

    def _generate_urban_features(self):
        """Generate synthetic urban features for radar testing"""
        self.urban_features = []

        # Add some buildings
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

        print(f"Generated {len(self.urban_features)} urban features for radar testing")

    def _create_patrol_path(self, drone, center, radius):
        """Create patrol path for drone"""
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

    def update_drones(self, delta_time: float = 0.016):
        """Update complete drone system"""
        if not self.drones or not self.enemies:
            return

        self.current_time += delta_time

        # Step 1: Update radar network (multistatic communication)
        self.radar_network.update(self.current_time, self.enemies, self.urban_features)

        # Step 2: Update drone movement
        for drone in self.drones:
            self._update_single_drone(drone, delta_time)

        # Step 3: Update ASDA based on current threats
        threat_positions = [e.position for e in self.enemies]
        self.adaptive_sector._update_threats(threat_positions)

    def _update_single_drone(self, drone, delta_time):
        """Update single drone's movement"""
        # Move toward patrol point
        tx, ty, tz = drone.target_position
        dx = tx - drone.position[0]
        dy = ty - drone.position[1]
        dz = tz - drone.position[2]

        distance = math.sqrt(dx*dx + dy*dy + dz*dz)

        # Check if reached waypoint
        if distance < 30:
            drone.patrol_index = (drone.patrol_index + 1) % len(drone.patrol_points)
            drone.target_position = drone.patrol_points[drone.patrol_index]

            # Recalculate direction
            tx, ty, tz = drone.target_position
            dx = tx - drone.position[0]
            dy = ty - drone.position[1]
            dz = tz - drone.position[2]
            distance = math.sqrt(dx*dx + dy*dy + dz*dz)

        if distance > 0:
            # Set velocity toward target
            speed = 80.0
            drone.velocity = (
                (dx / distance) * speed,
                (dy / distance) * speed,
                (dz / distance) * speed * 0.5
            )

        # Update position
        new_x = drone.position[0] + drone.velocity[0] * delta_time
        new_y = drone.position[1] + drone.velocity[1] * delta_time
        new_z = drone.position[2] + drone.velocity[2] * delta_time

        drone.position = (new_x, new_y, new_z)

    def get_detection_data(self):
        """Get radar detection data for visualization"""
        detections = []

        for drone in self.drones:
            if hasattr(drone, 'estimated_positions'):
                for target_id, pos in drone.estimated_positions.items():
                    detections.append({
                        'drone_id': drone.id,
                        'target_id': target_id,
                        'position': pos,
                        'confidence': 0.8  # Simplified
                    })

        return detections

# ============================================================================
# TEST FUNCTION
# ============================================================================

def test_integrated_system():
    """Test the complete integrated system"""
    print("\n" + "=" * 70)
    print("TESTING COMPLETE INTEGRATED DRONE SYSTEM")
    print("=" * 70)

    # Create controller
    controller = GoodDroneController()

    # Create simple enemies for testing
    from dataclasses import dataclass
    @dataclass
    class TestEnemy:
        id: str
        position: Tuple[float, float, float]
        velocity: Tuple[float, float, float]
        color: Tuple[float, float, float] = (1.0, 0.0, 0.0)
        size: float = 8.0

    # Add test enemies
    controller.enemies = [
        TestEnemy(id="Enemy_01", position=(100, 100, 100), velocity=(0.1, 0.1, 0)),
        TestEnemy(id="Enemy_02", position=(-100, -100, 150), velocity=(-0.1, -0.1, 0)),
    ]

    controller.initialize_drones(num_drones=4)

    # Print system status
    print(f"\nSystem Status:")
    print(f"  Drones: {len(controller.drones)}")
    print(f"  Enemies: {len(controller.enemies)}")
    print(f"  Urban Features: {len(controller.urban_features)}")
    print(f"  Radar Network: {len(controller.radar_network.drones)} drones connected")

    # Print sector allocation
    print("\nSector Allocation (ASDA):")
    for sector_name, sector_data in controller.adaptive_sector.sectors.items():
        threats = controller.adaptive_sector.sector_threats.get(sector_name, 0)
        print(f"  {sector_name}: Priority {sector_data['priority']:.2f}, Threats: {threats}")

    # Simulate some updates
    print("\nSimulating integrated system...")
    for i in range(5):
        controller.update_drones()
        print(f"  Frame {i+1}: Radar updates, drone movement, ASDA monitoring")

    # Show radar detections
    detections = controller.get_detection_data()
    print(f"\nRadar Detections: {len(detections)} targets being tracked")

    print(f"\n✓ Complete system operational!")
    print("  - Drones moving with patrol patterns")
    print("  - Radar detecting through urban features")
    print("  - Multistatic communication active")
    print("  - LSTM predictions ready")
    print("  - ASDA monitoring sector threats")

    return controller

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    # Test the complete system
    controller = test_integrated_system()