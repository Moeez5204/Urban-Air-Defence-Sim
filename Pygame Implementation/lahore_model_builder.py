# lahore_model_builder.py
import json
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Tuple
import math


@dataclass
class Building3D:
    vertices: List[Tuple[float, float, float]]
    faces: List[List[int]]
    height: float
    center: Tuple[float, float]
    zone: str
    color: Tuple[float, float, float]
    footprint: List[Tuple[float, float]]


@dataclass
class Canyon3D:
    centerline: List[Tuple[float, float, float]]
    width: float
    depth: float
    color: Tuple[float, float, float]
    name: str
    threat_level: str


@dataclass
class Threat3D:
    position: Tuple[float, float, float]
    target_id: str
    threat_level: str
    color: Tuple[float, float, float]
    size: float
    velocity: Tuple[float, float, float]


@dataclass
class Asset3D:
    position: Tuple[float, float, float]
    asset_id: str
    asset_name: str
    priority: float
    color: Tuple[float, float, float]
    size: float


@dataclass
class Lahore3DModel:
    buildings: List[Building3D] = field(default_factory=list)
    canyons: List[Canyon3D] = field(default_factory=list)
    threats: List[Threat3D] = field(default_factory=list)
    assets: List[Asset3D] = field(default_factory=list)

    min_x: float = 0
    max_x: float = 0
    min_y: float = 0
    max_y: float = 0
    min_z: float = 0
    max_z: float = 0

    stats: Dict = field(default_factory=dict)


class LahoreModelBuilder:
    def __init__(self):
        self.model = Lahore3DModel()
        self.lon_min = 74.28
        self.lon_max = 74.45
        self.lat_min = 31.48
        self.lat_max = 31.60
        self.map_scale = 600

        self.raw_canyon_data = []
        self.all_utm_points = []
        self.canyon_utm_bounds_calculated = False

    def normalize_coordinates(self, lon: float, lat: float) -> Tuple[float, float]:
        norm_x = (lon - self.lon_min) / (self.lon_max - self.lon_min)
        norm_y = (lat - self.lat_min) / (self.lat_max - self.lat_min)

        x = (norm_x - 0.5) * 2 * self.map_scale
        y = (norm_y - 0.5) * 2 * self.map_scale

        return x, y

    def load_building_data(self, filename='building_data_3.1.1.json'):
        try:
            with open(filename, 'r') as f:
                building_data = json.load(f)

            print(f"Loaded {len(building_data)} buildings")

            zone_colors = {
                'Old City': (0.6, 0.4, 0.2),
                'Gulberg': (0.8, 0.2, 0.2),
                'Defence': (0.2, 0.2, 0.8),
                'Cantt': (0.2, 0.8, 0.2),
                'DHA': (0.8, 0.8, 0.2),
                'Model Town': (0.8, 0.2, 0.8),
                'Other': (0.5, 0.5, 0.5)
            }

            building_count = 0
            max_buildings = 300
            step = max(1, len(building_data) // max_buildings)

            for i in range(0, len(building_data), step):
                if building_count >= max_buildings:
                    break

                building_3d = self._create_building_3d(building_data[i], zone_colors)
                if building_3d:
                    self.model.buildings.append(building_3d)
                    building_count += 1

            print(f"Created {building_count} 3D buildings")
            return True

        except FileNotFoundError:
            print(f"Error: {filename} not found")
            return False
        except Exception as e:
            print(f"Error loading building data: {e}")
            return False

    def _create_building_3d(self, building_data: Dict, zone_colors: Dict) -> Building3D:
        try:
            height = building_data['height']
            center = building_data['center']
            zone = building_data['zone']
            area = building_data['area']

            lon, lat = center
            x, y = self.normalize_coordinates(lon, lat)
            z = 0

            height_scaled = height * 3

            self.model.min_x = min(self.model.min_x, x)
            self.model.max_x = max(self.model.max_x, x)
            self.model.min_y = min(self.model.min_y, y)
            self.model.max_y = max(self.model.max_y, y)
            self.model.max_z = max(self.model.max_z, height_scaled)

            base_size = math.sqrt(area) * 0.15
            min_size = 5.0
            max_size = 25.0
            size = max(min_size, min(base_size, max_size))

            vertices = [
                (x - size, y - size, z),
                (x + size, y - size, z),
                (x + size, y + size, z),
                (x - size, y + size, z),
                (x - size, y - size, z + height_scaled),
                (x + size, y - size, z + height_scaled),
                (x + size, y + size, z + height_scaled),
                (x - size, y + size, z + height_scaled)
            ]

            faces = [
                [0, 1, 2, 3],
                [4, 5, 6, 7],
                [0, 1, 5, 4],
                [2, 3, 7, 6],
                [0, 3, 7, 4],
                [1, 2, 6, 5]
            ]

            footprint = [
                (x - size, y - size),
                (x + size, y - size),
                (x + size, y + size),
                (x - size, y + size)
            ]

            color = zone_colors.get(zone, (0.5, 0.5, 0.5))

            height_factor = min(1.0, height_scaled / 150)
            if zone == 'Gulberg' or zone == 'Defence':
                color = tuple(min(1.0, c * (0.8 + height_factor * 0.4)) for c in color)
            else:
                color = tuple(c * (0.7 + height_factor * 0.3) for c in color)

            return Building3D(
                vertices=vertices,
                faces=faces,
                height=height_scaled,
                center=(x, y),
                zone=zone,
                color=color,
                footprint=footprint
            )

        except Exception as e:
            print(f"Error creating building 3D: {e}")
            return None

    def load_strategic_data(self, filename='strategic_features_3.1.4.json'):

        try:
            self.model.canyons = []

            with open(filename, 'r') as f:
                strategic_data = json.load(f)

            self.raw_canyon_data = []
            self.all_utm_points = []
            self.canyon_utm_bounds_calculated = False

            if 'strategic_features' in strategic_data:
                features = strategic_data['strategic_features']

                canyon_count = 0
                for i, canyon in enumerate(features.get('canyons', [])):
                    canyon_count += 1
                    self._create_canyon_3d(canyon, i)

                print(f"Collected data for {canyon_count} canyons")

                if self.model.canyons:
                    return True

                self.process_all_canyons_together()

                print(f"succesfulyt created {len(self.model.canyons)} 3D canyons")

            return True

        except FileNotFoundError:
            print(f"Warning: {filename} not found")
            return False
        except Exception as e:
            print(f"Error loading strategic data: {e}")
            import traceback
            traceback.print_exc()
            return False

    def _create_canyon_3d(self, canyon_data: Dict, index: int) -> Canyon3D:
        try:
            persistence = canyon_data.get('persistence', 100)
            threat_level = canyon_data.get('threat_level', 'medium')
            centerline = canyon_data.get('centerline', [])

            if not centerline:
                return None

            for point in centerline:
                if len(point) >= 2:
                    self.all_utm_points.append((point[0], point[1]))

            self.raw_canyon_data.append({
                'index': index,
                'centerline': centerline,
                'persistence': persistence,
                'threat_level': threat_level
            })

            return None

        except Exception as e:
            print(f"Error in canyon processing: {e}")
            return None

    def load_threat_data(self, filename='lahore_3d_data.json'):

        try:
            with open(filename, 'r') as f:
                threat_data = json.load(f)

            threat_count = 0
            for threat in threat_data.get('threat_data', []):
                if threat_count >= 30:
                    break

                threat_3d = self._create_threat_3d(threat)
                if threat_3d:
                    self.model.threats.append(threat_3d)
                    threat_count += 1

            asset_count = 0
            for asset in threat_data.get('defended_assets', []):
                if asset_count >= 15:
                    break

                asset_3d = self._create_asset_3d(asset)
                if asset_3d:
                    self.model.assets.append(asset_3d)
                    asset_count += 1

            print(f"Created {threat_count} threats and {asset_count} assets")
            return True

        except FileNotFoundError:
            print(f"Warning: {filename} not found")
            return False
        except Exception as e:
            print(f"Error loading threat data: {e}")
            return False

    def _create_threat_3d(self, threat_data: Dict) -> Threat3D:
        try:
            position_data = threat_data.get('target_position', [74.35, 31.55, 100])

            if len(position_data) < 3 or position_data[0] == 0:
                lon = np.random.uniform(self.lon_min + 0.02, self.lon_max - 0.02)
                lat = np.random.uniform(self.lat_min + 0.02, self.lat_max - 0.02)
                alt = np.random.uniform(80, 200)
            else:
                lon, lat, alt = position_data

            x, y = self.normalize_coordinates(lon, lat)
            z = alt * 0.3

            bounds = 550
            x = max(-bounds, min(x, bounds))
            y = max(-bounds, min(y, bounds))
            z = max(50, min(z, 300))

            threat_level = threat_data.get('threat_level', 'medium')

            if threat_level == 'CRITICAL':
                color = (1.0, 0.0, 0.0)
                size = 12.0
            elif threat_level == 'HIGH':
                color = (1.0, 0.5, 0.0)
                size = 10.0
            elif threat_level == 'MEDIUM':
                color = (1.0, 1.0, 0.0)
                size = 8.0
            else:
                color = (0.5, 0.5, 0.5)
                size = 6.0

            velocity = (
                np.random.uniform(-0.2, 0.2),
                np.random.uniform(-0.2, 0.2),
                np.random.uniform(-0.05, 0.05)
            )

            return Threat3D(
                position=(x, y, z),
                target_id=threat_data.get('target_id', f'Threat_{np.random.randint(1000)}'),
                threat_level=threat_level,
                color=color,
                size=size,
                velocity=velocity
            )

        except Exception as e:
            print(f"Error creating threat 3D: {e}")
            return None

    def _create_asset_3d(self, asset_data: Dict) -> Asset3D:
        try:
            position_data = asset_data.get('position', [74.35, 31.55, 0])
            if len(position_data) < 2:
                return None

            lon, lat = position_data[0], position_data[1]
            x, y = self.normalize_coordinates(lon, lat)
            z = position_data[2] if len(position_data) > 2 else 0

            priority = asset_data.get('priority', 0.5)

            if priority > 0.9:
                color = (0.0, 1.0, 0.0)
            elif priority > 0.7:
                color = (0.0, 0.7, 0.7)
            else:
                color = (0.0, 0.0, 1.0)

            size = 8 + priority * 12

            return Asset3D(
                position=(x, y, z),
                asset_id=asset_data.get('asset_id', 'Unknown'),
                asset_name=asset_data.get('asset_name', 'Unknown'),
                priority=priority,
                color=color,
                size=size
            )

        except Exception as e:
            print(f"Error creating asset 3D: {e}")
            return None

    def calculate_statistics(self):
        self.model.stats = {
            'total_buildings': len(self.model.buildings),
            'total_canyons': len(self.model.canyons),
            'total_threats': len(self.model.threats),
            'total_assets': len(self.model.assets),
            'map_bounds': {
                'x_min': self.model.min_x,
                'x_max': self.model.max_x,
                'y_min': self.model.min_y,
                'y_max': self.model.max_y,
                'z_max': self.model.max_z
            },
            'avg_building_height': np.mean([b.height for b in self.model.buildings]) if self.model.buildings else 0,
            'map_scale': self.map_scale
        }

        print("\n=== 3D Model Statistics ===")
        print(f"Buildings: {self.model.stats['total_buildings']}")
        print(f"Canyons: {self.model.stats['total_canyons']}")
        print(f"Threats: {self.model.stats['total_threats']}")
        print(f"Assets: {self.model.stats['total_assets']}")
        print(f"Avg Building Height: {self.model.stats['avg_building_height']:.1f}m")
        print(f"Map Bounds: X({self.model.min_x:.0f} to {self.model.max_x:.0f}) "
              f"Y({self.model.min_y:.0f} to {self.model.max_y:.0f})")
        print(f"Map Scale: {self.map_scale}")

    def build_complete_model(self) -> Lahore3DModel:

        building_loaded = self.load_building_data()
        strategic_loaded = self.load_strategic_data()
        threat_loaded = self.load_threat_data()

        if not building_loaded:
            print("ERROR: Building data file not found or could not be loaded")
            return None

        self.calculate_statistics()
        self.debug_canyon_positions()

        print("3D Model construction complete!")
        return self.model

    def debug_canyon_positions(self):

        if not self.model.canyons:
            print("No canyons to debug")
            return

        for i, canyon in enumerate(self.model.canyons[:5]):
            if not canyon.centerline:
                print(f"Canyon {i}: No centerline")
                continue

            x_vals = [p[0] for p in canyon.centerline]
            y_vals = [p[1] for p in canyon.centerline]

            min_x, max_x = min(x_vals), max(x_vals)
            min_y, max_y = min(y_vals), max(y_vals)
            center_x = (min_x + max_x) / 2
            center_y = (min_y + max_y) / 2

            print(f"Canyon {i} ({canyon.name}):")
            print(f"  X: {min_x:.1f} to {max_x:.1f} (center: {center_x:.1f})")
            print(f"  Y: {min_y:.1f} to {max_y:.1f} (center: {center_y:.1f})")
            print(f"  Points: {len(canyon.centerline)}, Threat: {canyon.threat_level}")

        all_x, all_y = [], []
        for canyon in self.model.canyons:
            for point in canyon.centerline:
                all_x.append(point[0])
                all_y.append(point[1])

        if all_x and all_y:
            print(f"\nOVERALL SPREAD:")
            print(f"  X range: {min(all_x):.1f} to {max(all_x):.1f} (span: {max(all_x) - min(all_x):.1f})")
            print(f"  Y range: {min(all_y):.1f} to {max(all_y):.1f} (span: {max(all_y) - min(all_y):.1f})")

    def process_all_canyons_together(self):

        if not self.raw_canyon_data:
            print("No raw canyon data to process")
            return

        lahore_strategic_locations = [
            (-400, 400, "Northwest_OldCity", 1.0),
            (-200, 200, "North_Central", 0.9),
            (0, 0, "City_Center", 1.2),
            (200, -200, "Southeast_Gulberg", 1.1),
            (-300, -300, "Southwest_Cantt", 0.8),
            (100, 300, "Northeast_Defence", 0.9),
            (-500, 100, "West_City", 0.7),
            (150, -400, "South_City", 0.8),
            (-100, -100, "Central", 1.0),
            (300, 100, "East_City", 0.9),
            (-450, -200, "Southwest", 0.8),
            (250, 350, "Northeast", 0.9),
        ]

        print(f"Lahore Strategic Locations: {len(lahore_strategic_locations)}")
        print(f"Canyons to place: {len(self.raw_canyon_data)}")

        for i, canyon_data in enumerate(self.raw_canyon_data):
            index = canyon_data['index']
            centerline = canyon_data['centerline']
            persistence = canyon_data['persistence']
            threat_level = canyon_data['threat_level']

            if i < len(lahore_strategic_locations):
                target_x, target_y, location_name, scale = lahore_strategic_locations[i]
                print(f"Canyon {index} → {location_name} at ({target_x}, {target_y})")
            else:
                import random
                target_x = random.uniform(-500, 200)
                target_y = random.uniform(-500, 500)
                location_name = f"Random_{i}"
                scale = 1.0

            original_center_x, original_center_y = 0, 0
            valid_points = 0

            for point in centerline:
                if len(point) >= 2:
                    original_center_x += point[0]
                    original_center_y += point[1]
                    valid_points += 1

            if valid_points > 0:
                original_center_x /= valid_points
                original_center_y /= valid_points

            canyon_converted = []
            for point in centerline:
                if len(point) >= 2:
                    utm_x, utm_y = point[0], point[1]
                    height = point[2] if len(point) > 2 else 5

                    centered_x = utm_x - original_center_x
                    centered_y = utm_y - original_center_y

                    scaled_x = centered_x * scale
                    scaled_y = centered_y * scale

                    final_x = target_x + scaled_x
                    final_y = target_y + scaled_y
                    final_z = max(5, height * 0.2)

                    canyon_converted.append((final_x, final_y, final_z))

            if len(canyon_converted) < 2: #skip if not enough data
                continue

            width = min(60, persistence / 50) + 40
            depth = min(35, persistence / 80) + 25

            threat_colors = {
                'high': (1.0, 0.0, 0.0, 0.9),
                'medium': (1.0, 0.6, 0.0, 0.9),
                'low': (0.0, 0.8, 0.0, 0.9)
            }
            color = threat_colors.get(threat_level, (0.5, 0.5, 0.8, 0.9))

            final_center_x = sum([p[0] for p in canyon_converted]) / len(canyon_converted)
            final_center_y = sum([p[1] for p in canyon_converted]) / len(canyon_converted)

            canyon_obj = Canyon3D(
                centerline=canyon_converted,
                width=width,
                depth=depth,
                color=color,
                name=f"{location_name}_{threat_level}",
                threat_level=threat_level
            )

            self.model.canyons.append(canyon_obj)

            print(f"  Created: {location_name}, Center: ({final_center_x:.0f}, {final_center_y:.0f})")
            print(f"  Size: {width:.1f}x{depth:.1f}, Points: {len(canyon_converted)}")

        print("Canyons are now avaliable")

        self.raw_canyon_data = []
        self.all_utm_points = []