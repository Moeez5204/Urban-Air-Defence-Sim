"""
BAD_DRONE_CONTROLLER.py - Enemy drone system
All enemy drones are RED for easy identification
"""

import random
import math
from dataclasses import dataclass, field
from typing import List, Tuple

@dataclass
class EnemyDrone3D:
    id: str
    position: Tuple[float, float, float]
    velocity: Tuple[float, float, float]
    color: Tuple[float, float, float] = (1.0, 0.0, 0.0)  # RED - ALL ENEMIES ARE RED
    size: float = 8.0
    attack_mode: str = "patrol"  # "patrol", "attack", "evade"
    target_position: Tuple[float, float, float] = (0, 0, 0)
    health: int = 100

class BadDroneController:

    def __init__(self):
        self.enemies: List[EnemyDrone3D] = []

        #map bounds
        self.map_bounds = {
            'x_min': -600, 'x_max': 600,
            'y_min': -600, 'y_max': 600,
            'z_min': 30, 'z_max': 400
        }
        print("✓ Bad drone controller initialized - ALL ENEMIES ARE RED")
        print(f"  City bounds: X({self.map_bounds['x_min']} to {self.map_bounds['x_max']})")
        print(f"               Y({self.map_bounds['y_min']} to {self.map_bounds['y_max']})")
        print(f"               Z({self.map_bounds['z_min']} to {self.map_bounds['z_max']})")

    def generate_enemies(self, num_enemies: int = 4): #make the enemies

        self.enemies = []

        for i in range(num_enemies):
            # GENERATE WITHIN CITY BOUNDS:
            x = random.uniform(self.map_bounds['x_min'] + 50, self.map_bounds['x_max'] - 50)
            y = random.uniform(self.map_bounds['y_min'] + 50, self.map_bounds['y_max'] - 50)
            z = random.uniform(self.map_bounds['z_min'] + 80, self.map_bounds['z_max'] - 50)  # Keep in air

            #Velocity scale matching:
            vx = random.uniform(-40, 40)
            vy = random.uniform(-40, 40)
            vz = random.uniform(-10, 10)
            #Enemies are red
            red_intensity = random.uniform(0.8, 1.0)
            color = (red_intensity, 0.0, 0.0)

            size = random.uniform(7.5, 9.5)

            self.enemies.append(EnemyDrone3D(
                id=f"Enemy_{i:02d}",
                position=(x, y, z),
                velocity=(vx, vy, vz),
                color=color,
                size=size
            ))

        print(f"✓ Generated {len(self.enemies)} RED enemy drones within city limits")
        return self.enemies

    def update_enemies(self, delta_time: float = 0.016):
        for enemy in self.enemies:
            x, y, z = enemy.position
            vx, vy, vz = enemy.velocity

            new_x = x + vx * delta_time
            new_y = y + vy * delta_time
            new_z = z + vz * delta_time

            if new_x < self.map_bounds['x_min']:
                vx = -vx * 0.8
                new_x = self.map_bounds['x_min'] + 5
            elif new_x > self.map_bounds['x_max']:
                vx = -vx * 0.8
                new_x = self.map_bounds['x_max'] - 5

            if new_y < self.map_bounds['y_min']:
                vy = -vy * 0.8
                new_y = self.map_bounds['y_min'] + 5
            elif new_y > self.map_bounds['y_max']:
                vy = -vy * 0.8
                new_y = self.map_bounds['y_max'] - 5

            if new_z < self.map_bounds['z_min']:
                vz = -vz * 0.8
                new_z = self.map_bounds['z_min'] + 10
            elif new_z > self.map_bounds['z_max']:
                vz = -vz * 0.8
                new_z = self.map_bounds['z_max'] - 10

            if random.random() < 0.015:
                vx += random.uniform(-15, 15)
                vy += random.uniform(-15, 15)
                vz += random.uniform(-5, 5)
                speed = math.sqrt(vx*vx + vy*vy + vz*vz)
                max_speed = 80
                if speed > max_speed:
                    scale = max_speed / speed
                    vx *= scale
                    vy *= scale
                    vz *= scale


            enemy.position = (new_x, new_y, new_z)
            enemy.velocity = (vx, vy, vz)

        return self.enemies

    def add_enemy(self, position=None):
        enemy_id = f"Enemy_{len(self.enemies):02d}"

        if position:
            x, y, z = position
        else:
            x = random.uniform(self.map_bounds['x_min'] + 50, self.map_bounds['x_max'] - 50)
            y = random.uniform(self.map_bounds['y_min'] + 50, self.map_bounds['y_max'] - 50)
            z = random.uniform(self.map_bounds['z_min'] + 80, self.map_bounds['z_max'] - 50)

        vx = random.uniform(-40, 40)
        vy = random.uniform(-40, 40)
        vz = random.uniform(-10, 10)

        red_intensity = random.uniform(0.8, 1.0)
        color = (red_intensity, 0.0, 0.0)

        enemy = EnemyDrone3D(
            id=enemy_id,
            position=(x, y, z),
            velocity=(vx, vy, vz),
            color=color,
            size=random.uniform(7.5, 9.5)
        )

        self.enemies.append(enemy)
        print(f"✓ Added RED enemy drone {enemy_id} at ({x:.0f}, {y:.0f}, {z:.0f})")
        return enemy

    def remove_enemy(self, enemy_id):
        self.enemies = [e for e in self.enemies if e.id != enemy_id]

    def get_enemy_positions(self):
        return [enemy.position for enemy in self.enemies]

    def get_enemies_near_position(self, position, radius=100):
        x, y, z = position
        nearby = []

        for enemy in self.enemies:
            ex, ey, ez = enemy.position
            distance = math.sqrt((ex - x)**2 + (ey - y)**2 + (ez - z)**2)
            if distance <= radius:
                nearby.append(enemy)

        return nearby