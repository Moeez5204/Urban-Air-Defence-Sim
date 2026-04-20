# lahore_3d_renderer.py
import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
import numpy as np
import math
import time
from lahore_model_builder import Lahore3DModel, Building3D, Canyon3D, Threat3D, Asset3D
import random
from good_drone_controller import GoodDroneController, GoodDrone3D


GOOD_DRONES_AVAILABLE = True

# Try to import bad drone controller
try:
    from bad_drone_controller import BadDroneController, EnemyDrone3D

    BAD_DRONES_AVAILABLE = True
    print("Bad drone controller working")
except ImportError:
    BAD_DRONES_AVAILABLE = False
    print("Bad drone controller not working")


class Lahore3DRenderer:

    def __init__(self, screen_width=1400, screen_height=900):
        # PyGame initialization
        pygame.init()
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.screen = pygame.display.set_mode((screen_width, screen_height), DOUBLEBUF | OPENGL)
        pygame.display.set_caption("Lahore 3D Urban Defense Visualization")

        # OpenGL initialization
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_LIGHTING)
        glEnable(GL_LIGHT0)
        glEnable(GL_COLOR_MATERIAL)
        glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        # Lighting setup
        glLightfv(GL_LIGHT0, GL_POSITION, (800, 800, 1200, 0))
        glLightfv(GL_LIGHT0, GL_AMBIENT, (0.4, 0.4, 0.4, 1.0))
        glLightfv(GL_LIGHT0, GL_DIFFUSE, (0.8, 0.8, 0.8, 1.0))
        glLightfv(GL_LIGHT0, GL_SPECULAR, (0.5, 0.5, 0.5, 1.0))
        glLightModeli(GL_LIGHT_MODEL_TWO_SIDE, GL_TRUE)

        # Camera settings
        self.camera_distance = 1000
        self.camera_angle_x = 45
        self.camera_angle_y = -45
        self.camera_target = (0, 0, 50)

        # Mouse control
        self.mouse_dragging = False
        self.last_mouse_pos = (0, 0)

        # Display mode
        self.show_buildings = True
        self.show_canyons = True
        self.show_threats = True
        self.show_assets = True
        self.show_wireframe = False
        self.show_ground = True
        self.show_axes = True
        self.show_good_drones = True
        self.show_radar_pulses = False

        # Drone systems
        self.good_drone_controller = None
        self.bad_drone_controller = None

        # Animation
        self.animation_time = 0
        self.threat_update_interval = 0.1
        self.last_threat_update = 0

        # Font for text
        pygame.font.init()
        self.font = pygame.font.SysFont('Arial', 20)
        self.small_font = pygame.font.SysFont('Arial', 16)

        # Performance tracking
        self.frame_count = 0
        self.last_fps_time = time.time()
        self.fps = 0

        print("3D Renderer initialized")

    def initialize_good_drones(self, num_drones=4):
        if not GOOD_DRONES_AVAILABLE:
            print("⚠ Good drone controller not available.")
            return False

        try:
            self.good_drone_controller = GoodDroneController()

            if BAD_DRONES_AVAILABLE:
                self.bad_drone_controller = BadDroneController()
                enemies = self.bad_drone_controller.generate_enemies(num_enemies=6)  # More enemies!
            else:
                # Fallback - create simple enemies
                print("⚠ Bad drone controller not available. Creating simple enemies...")
                enemies = []
                for i in range(4):
                    enemies.append(EnemyDrone3D(
                        id=f"Enemy_{i:02d}",
                        position=(random.uniform(-500, 500),
                                  random.uniform(-500, 500),
                                  random.uniform(100, 300)),
                        velocity=(random.uniform(-0.3, 0.3),
                                  random.uniform(-0.3, 0.3),
                                  random.uniform(-0.1, 0.1)),
                        color=(1.0, 0.0, 0.0),
                        size=8.0
                    ))

            self.good_drone_controller.set_enemies(enemies)

            drones = self.good_drone_controller.initialize_drones(num_drones=num_drones) #init

            if drones:
                print(f" {len(drones)} good drones initialized")
                print(f" {len(enemies)} enemy drones initialized")
                print("Press 'M' for ASDA/LSTM report")
                print("Press 'D' to toggle drone visibility")
                print("Press 'P' to toggle radar pulses")
                return True
            else:
                print("No drones initialized")
                return False

        except Exception as e:
            print(f"Error initializing good drones: {e}")
            import traceback
            traceback.print_exc()
            return False

    def setup_projection(self):
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(45, (self.screen_width / self.screen_height), 0.1, 10000.0)
        glMatrixMode(GL_MODELVIEW)

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return False
                elif event.key == pygame.K_b:
                    self.show_buildings = not self.show_buildings
                elif event.key == pygame.K_c:
                    self.show_canyons = not self.show_canyons
                elif event.key == pygame.K_t:
                    self.show_threats = not self.show_threats
                elif event.key == pygame.K_a:
                    self.show_assets = not self.show_assets
                elif event.key == pygame.K_w:
                    self.show_wireframe = not self.show_wireframe
                elif event.key == pygame.K_g:
                    self.show_ground = not self.show_ground
                elif event.key == pygame.K_x:
                    self.show_axes = not self.show_axes
                elif event.key == pygame.K_d:
                    self.show_good_drones = not self.show_good_drones
                elif event.key == pygame.K_p:
                    self.show_radar_pulses = not self.show_radar_pulses
                    print(f"Radar pulses: {'ON' if self.show_radar_pulses else 'OFF'}")
                elif event.key == pygame.K_i:
                    if not self.good_drone_controller:
                        self.initialize_good_drones(num_drones=8)
                    else:
                        print("Good drones already initialized")
                elif event.key == pygame.K_m:
                    self.print_asda_lstm_report()
                elif event.key == pygame.K_PLUS or event.key == pygame.K_EQUALS:
                    self.camera_distance *= 0.9
                elif event.key == pygame.K_MINUS:
                    self.camera_distance *= 1.1
                elif event.key == pygame.K_UP:
                    self.camera_angle_y += 5
                elif event.key == pygame.K_DOWN:
                    self.camera_angle_y -= 5
                elif event.key == pygame.K_LEFT:
                    self.camera_angle_x += 5
                elif event.key == pygame.K_RIGHT:
                    self.camera_angle_x -= 5
                elif event.key == pygame.K_r:
                    # Reset camera
                    self.camera_distance = 1000
                    self.camera_angle_x = 45
                    self.camera_angle_y = -45
                    self.camera_target = (0, 0, 50)
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    self.mouse_dragging = True
                    self.last_mouse_pos = pygame.mouse.get_pos()
                elif event.button == 4:
                    self.camera_distance *= 0.9
                elif event.button == 5:
                    self.camera_distance *= 1.1
            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1:
                    self.mouse_dragging = False
            elif event.type == pygame.MOUSEMOTION:
                if self.mouse_dragging:
                    x, y = pygame.mouse.get_pos()
                    dx = x - self.last_mouse_pos[0]
                    dy = y - self.last_mouse_pos[1]
                    self.camera_angle_x += dx * 0.5
                    self.camera_angle_y += dy * 0.5
                    self.camera_angle_y = max(-89, min(89, self.camera_angle_y))
                    self.last_mouse_pos = (x, y)

        return True

    def print_asda_lstm_report(self):
        if not self.good_drone_controller:
            print("\n Good drone controller not initialized")
            return

        print("ASDA & LSTM SYSTEM REPORT")

        controller = self.good_drone_controller

        # Get enemies
        enemies = []
        if self.bad_drone_controller:
            enemies = self.bad_drone_controller.enemies
        elif controller.enemies:
            enemies = controller.enemies

        threat_positions = [e.position for e in enemies] if enemies else []

        if hasattr(controller.adaptive_sector, '_update_threats'):
            controller.adaptive_sector._update_threats(threat_positions)

        # ASDA Allocation Report
        print("ADAPTIVE SECTOR DEFENSE ALLOCATION (ASDA):")

        # Get sectors from external ASDA
        sector_names = ['Walled_City', 'Central_Lahore', 'Gulberg', 'Cantonment', 'Other_Sector']
        sector_priorities = {
            'Walled_City': 0.9,
            'Central_Lahore': 1.0,
            'Gulberg': 0.8,
            'Cantonment': 0.7,
            'Other_Sector': 0.6
        }

        # Get threat counts from external ASDA's threat_history
        sector_threats = {sector: 0 for sector in sector_names}
        if hasattr(controller.adaptive_sector, 'threat_history'):
            for sector, threats in controller.adaptive_sector.threat_history.items():
                sector_threats[sector] = len(threats) if threats else 0

        # Count drones per sector
        sector_drone_counts = {}
        for drone in controller.drones:
            sector = drone.sector
            sector_drone_counts[sector] = sector_drone_counts.get(sector, 0) + 1

        for sector_name in sector_names:
            drones = sector_drone_counts.get(sector_name, 0)
            threats = sector_threats.get(sector_name, 0)
            priority = sector_priorities.get(sector_name, 0.5)

            # Create bar for drones
            drone_bar = "█" * min(drones, 10)
            if drones > 10:
                drone_bar += f"+{drones - 10}"

            # Create bar for threats
            threat_bar = "⚠" * min(threats, 5)
            if threats > 5:
                 threat_bar += f"+{threats - 5}"

            print(f"{sector_name:<18} Priority: {priority:.2f}")
            print(f"Drones:  {drone_bar:<15} ({drones})")
            print(f"Threats: {threat_bar:<15} ({threats})")

            # Create bar for drones
            drone_bar = "█" * min(drones, 10)
            if drones > 10:
                drone_bar += f"+{drones - 10}"

            # Create bar for threats
            threat_bar = "⚠" * min(threats, 5)
            if threats > 5:
                threat_bar += f"+{threats - 5}"

            print(f"{sector_name:<18} Priority: {priority:.2f}")
            print(f"  Drones:  {drone_bar:<15} ({drones})")
            print(f"  Threats: {threat_bar:<15} ({threats})")
            print()

        # LSTM Prediction Report
        print("\nLSTM PREDICTION SYSTEM:")

        # simulate some LSTM predictions
        prediction_accuracies = {
            '5-second': random.uniform(0.7, 0.9),
            '10-second': random.uniform(0.5, 0.7),
            '30-second': random.uniform(0.3, 0.5),
            'pattern_recognition': random.uniform(0.6, 0.8)
        }

        for pred_type, accuracy in prediction_accuracies.items():
            bar_length = int(accuracy * 20)
            bar = "█" * bar_length + "░" * (20 - bar_length)
            percentage = accuracy * 100
            print(f"{pred_type:<20} {bar} {percentage:5.1f}%")

        # system Summary
        print("\nSYSTEM SUMMARY:")
        print(f"Total Drones: {len(controller.drones)}")
        print(f"Total Threats: {len(enemies)}")
        print(f"Active Sectors: {len([s for s in sector_drone_counts if sector_drone_counts[s] > 0])}")

        # Calculate defense coverage
        total_drones = len(controller.drones)
        num_sectors = len(sector_names)  # Use sector_names instead of undefined sector_data
        coverage = min(100, (total_drones / (num_sectors * 2)) * 100)  # 2 drones per sector ideal

        coverage_bar_length = int(coverage / 5)
        coverage_bar = "█" * coverage_bar_length + "░" * (20 - coverage_bar_length)
        print(f"Defense Coverage: {coverage_bar} {coverage:5.1f}%")

        coverage_bar_length = int(coverage / 5)
        coverage_bar = "█" * coverage_bar_length + "░" * (20 - coverage_bar_length)
        print(f"Defense Coverage: {coverage_bar} {coverage:5.1f}%")

    def update_camera(self):
        glLoadIdentity()

        # Convert spherical coordinates to Cartesian
        rad_x = math.radians(self.camera_angle_x)
        rad_y = math.radians(self.camera_angle_y)

        camera_x = self.camera_distance * math.cos(rad_y) * math.sin(rad_x)
        camera_y = self.camera_distance * math.sin(rad_y)
        camera_z = self.camera_distance * math.cos(rad_y) * math.cos(rad_x)

        # Add camera target offset
        target_x, target_y, target_z = self.camera_target

        gluLookAt(
            camera_x + target_x, camera_y + target_z, camera_z + target_y,  # Camera pos
            target_x, target_z, target_y,
            0, 1, 0
        )

    def render_axes(self):
        if not self.show_axes:
            return

        glDisable(GL_LIGHTING)
        glLineWidth(2.0)

        # X axis
        glBegin(GL_LINES)
        glColor3f(1.0, 0.0, 0.0)
        glVertex3f(0, 0, 0)
        glVertex3f(200, 0, 0)
        glEnd()

        # Y axis
        glBegin(GL_LINES)
        glColor3f(0.0, 1.0, 0.0)
        glVertex3f(0, 0, 0)
        glVertex3f(0, 200, 0)
        glEnd()

        # Z axis
        glBegin(GL_LINES)
        glColor3f(0.0, 0.0, 1.0)
        glVertex3f(0, 0, 0)
        glVertex3f(0, 0, 200)
        glEnd()

        glLineWidth(1.0)
        glEnable(GL_LIGHTING)

    def render_ground(self, model: Lahore3DModel):

        if not self.show_ground:
            return

        glPushMatrix()
        glDisable(GL_LIGHTING)

        # Get bounds from model or use default
        grid_size = max(abs(model.min_x), abs(model.max_x),
                        abs(model.min_y), abs(model.max_y), 500)
        grid_size = int(min(grid_size, 800))

        # Draw grid
        grid_step = max(50, grid_size // 10)

        glColor3f(0.25, 0.35, 0.25)  # Darker greenish ground

        glBegin(GL_LINES)
        for i in range(-grid_size, grid_size + grid_step, grid_step):
            glVertex3f(i, -grid_size, 0)
            glVertex3f(i, grid_size, 0)
            glVertex3f(-grid_size, i, 0)
            glVertex3f(grid_size, i, 0)
        glEnd()

        glEnable(GL_LIGHTING)
        glPopMatrix()

    def render_city_boundary(self, model: Lahore3DModel):
        if not self.show_ground:
            return

        glPushMatrix()
        glDisable(GL_LIGHTING)
        glEnable(GL_BLEND)

        # Get city bounds from model
        city_min_x = model.min_x if model.min_x < model.max_x else -600
        city_max_x = model.max_x if model.max_x > model.min_x else 600
        city_min_y = model.min_y if model.min_y < model.max_y else -600
        city_max_y = model.max_y if model.max_y > model.min_y else 600

        buffer = 50
        city_min_x -= buffer
        city_max_x += buffer
        city_min_y -= buffer
        city_max_y += buffer

        # Draw boundary box
        glColor4f(0.8, 0.8, 0.2, 0.3)
        glLineWidth(3.0)

        glBegin(GL_LINE_LOOP)
        glVertex3f(city_min_x, city_min_y, 1)
        glVertex3f(city_max_x, city_min_y, 1)
        glVertex3f(city_max_x, city_max_y, 1)
        glVertex3f(city_min_x, city_max_y, 1)
        glEnd()

        glLineWidth(1.0)
        glDisable(GL_BLEND)
        glEnable(GL_LIGHTING)
        glPopMatrix()

    def render_building(self, building: Building3D):
        if not self.show_buildings:
            return

        glPushMatrix()

        # Set color
        glColor3f(*building.color)
        glMaterialfv(GL_FRONT, GL_AMBIENT_AND_DIFFUSE, (*building.color, 1.0))
        glMaterialfv(GL_FRONT, GL_SPECULAR, (0.5, 0.5, 0.5, 1.0))
        glMaterialf(GL_FRONT, GL_SHININESS, 50.0)

        if self.show_wireframe:
            glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)
            glDisable(GL_LIGHTING)
        else:
            glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)

        # Draw building
        for face in building.faces:
            glBegin(GL_QUADS)
            for vertex_index in face:
                vertex = building.vertices[vertex_index]
                glVertex3f(*vertex)
            glEnd()

        # Reset polygon mode
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)

        if self.show_wireframe:
            glEnable(GL_LIGHTING)

        glPopMatrix()

    def render_canyon(self, canyon: Canyon3D, model: Lahore3DModel = None):
        if not self.show_canyons or len(canyon.centerline) < 2:
            return

        glPushMatrix()

        # Set up proper blending for canyons
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        if canyon.threat_level == 'high':
            main_color = (1.0, 0.3, 0.3, 0.7)  # red
        elif canyon.threat_level == 'medium':
            main_color = (1.0, 0.7, 0.3, 0.7)  #orange
        else:
            main_color = (0.3, 1.0, 0.3, 0.7)  # green

        # Draw canyon
        for i in range(len(canyon.centerline) - 1):
            start = canyon.centerline[i]
            end = canyon.centerline[i + 1]

            dx = end[0] - start[0]
            dy = end[1] - start[1]
            length = math.sqrt(dx * dx + dy * dy)

            if length == 0:
                continue

            perp_x = -dy / length * canyon.width / 2
            perp_y = dx / length * canyon.width / 2

            # Main canyon body
            glDisable(GL_LIGHTING)
            glColor4f(*main_color)
            glBegin(GL_QUADS)

            #Top face
            glVertex3f(start[0] - perp_x, start[1] - perp_y, start[2] + canyon.depth)
            glVertex3f(start[0] + perp_x, start[1] + perp_y, start[2] + canyon.depth)
            glVertex3f(end[0] + perp_x, end[1] + perp_y, end[2] + canyon.depth)
            glVertex3f(end[0] - perp_x, end[1] - perp_y, end[2] + canyon.depth)

            # Bottom face
            glVertex3f(start[0] - perp_x, start[1] - perp_y, start[2])
            glVertex3f(start[0] + perp_x, start[1] + perp_y, start[2])
            glVertex3f(end[0] + perp_x, end[1] + perp_y, end[2])
            glVertex3f(end[0] - perp_x, end[1] - perp_y, end[2])

            #Side faces for better 3D effect
            glVertex3f(start[0] - perp_x, start[1] - perp_y, start[2])
            glVertex3f(start[0] - perp_x, start[1] - perp_y, start[2] + canyon.depth)
            glVertex3f(end[0] - perp_x, end[1] - perp_y, end[2] + canyon.depth)
            glVertex3f(end[0] - perp_x, end[1] - perp_y, end[2])

            glVertex3f(start[0] + perp_x, start[1] + perp_y, start[2])
            glVertex3f(start[0] + perp_x, start[1] + perp_y, start[2] + canyon.depth)
            glVertex3f(end[0] + perp_x, end[1] + perp_y, end[2] + canyon.depth)
            glVertex3f(end[0] + perp_x, end[1] + perp_y, end[2])

            glEnd()

            glEnable(GL_LIGHTING)

        glDisable(GL_BLEND)
        glPopMatrix()

    def render_threat(self, threat: Threat3D):
        if not self.show_threats:
            return

        glPushMatrix()

        # Position
        x, y, z = threat.position
        glTranslatef(x, y, z)

        # Set color
        current_time = time.time()
        pulse_factor = 1.0 + 0.2 * math.sin(current_time * 5)

        # Color based on threat level
        if threat.threat_level == 'CRITICAL':
            color = (1.0, 0.0, 0.0)  # Red
        elif threat.threat_level == 'HIGH':
            color = (1.0, 0.5, 0.0)  # Orange
        elif threat.threat_level == 'MEDIUM':
            color = (1.0, 1.0, 0.0)  # Yellow
        else:
            color = (0.5, 0.5, 0.5)  # Gray

        glColor3f(*color)
        glMaterialfv(GL_FRONT, GL_AMBIENT_AND_DIFFUSE, (*color, 1.0))
        quadric = gluNewQuadric()
        gluSphere(quadric, threat.size * pulse_factor, 12, 12)
        gluDeleteQuadric(quadric)

        glPopMatrix()

    def render_asset(self, asset: Asset3D):
        if not self.show_assets:
            return

        glPushMatrix()

        # Position
        x, y, z = asset.position
        glTranslatef(x, y, z)
        glColor3f(*asset.color)
        glMaterialfv(GL_FRONT, GL_AMBIENT_AND_DIFFUSE, (*asset.color, 1.0))

        # Draw pyramid
        size = asset.size

        glBegin(GL_TRIANGLES)
        # Base
        glVertex3f(-size, 0, -size)
        glVertex3f(size, 0, -size)
        glVertex3f(size, 0, size)

        glVertex3f(-size, 0, -size)
        glVertex3f(size, 0, size)
        glVertex3f(-size, 0, size)

        # Sides
        glVertex3f(-size, 0, -size)
        glVertex3f(0, size * 1.5, 0)
        glVertex3f(size, 0, -size)

        glVertex3f(size, 0, -size)
        glVertex3f(0, size * 1.5, 0)
        glVertex3f(size, 0, size)

        glVertex3f(size, 0, size)
        glVertex3f(0, size * 1.5, 0)
        glVertex3f(-size, 0, size)

        glVertex3f(-size, 0, size)
        glVertex3f(0, size * 1.5, 0)
        glVertex3f(-size, 0, -size)
        glEnd()

        glPopMatrix()

    def render_good_drones(self):
        """Render good drones as simple green spheres"""
        if not self.good_drone_controller or not self.show_good_drones:
            return

        if not self.good_drone_controller.drones:
            return

        # Save current OpenGL state
        glPushAttrib(GL_ALL_ATTRIB_BITS)
        glMatrixMode(GL_MODELVIEW)
        glPushMatrix()

        #lighting
        glEnable(GL_LIGHTING)
        glEnable(GL_LIGHT0)
        glEnable(GL_COLOR_MATERIAL)
        glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)

        # Set material properties for drones
        glMaterialfv(GL_FRONT, GL_AMBIENT_AND_DIFFUSE, (0.0, 0.8, 0.0, 1.0))
        glMaterialfv(GL_FRONT, GL_SPECULAR, (0.5, 0.5, 0.5, 1.0))
        glMaterialf(GL_FRONT, GL_SHININESS, 50.0)

        # Render each drone
        for drone in self.good_drone_controller.drones:
            glPushMatrix()

            # Position
            x, y, z = drone.position
            glTranslatef(x, y, z)

            # Set color
            glColor3f(0.0, 0.8, 0.0)

            # Draw as a sphere
            quadric = gluNewQuadric()
            gluSphere(quadric, drone.size, 16, 16)
            gluDeleteQuadric(quadric)

            glPopMatrix()

        # Restore OpenGL state
        glPopMatrix()
        glMatrixMode(GL_MODELVIEW)
        glPopAttrib()

    def render_radar_pulses(self):
        """Render simple radar pulses as expanding spheres from good drones"""
        if not self.good_drone_controller or not self.show_good_drones or not self.show_radar_pulses:
            return

        glPushAttrib(GL_ALL_ATTRIB_BITS)
        glMatrixMode(GL_MODELVIEW)
        glPushMatrix()

        # Set up for wireframe
        glDisable(GL_LIGHTING)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glDepthMask(GL_FALSE)
        glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)

        # Render pulses for each drone
        for drone in self.good_drone_controller.drones:
            if not hasattr(drone, 'radar_system'):
                continue

            radar = drone.radar_system
            x, y, z = drone.position

            # Draw pulse
            for i, pulse_radius in enumerate(radar.radar_pulse_radii):
                progress = pulse_radius / 300.0  # 300m max range
                r = 0.0
                g = 1.0 - progress
                b = progress
                alpha = 0.7 - (progress * 0.6)  # Fade out

                glColor4f(r, g, b, alpha)

                # Draw sphere at drone position
                glPushMatrix()
                glTranslatef(x, y, z)

                quadric = gluNewQuadric()
                gluQuadricDrawStyle(quadric, GLU_LINE)
                gluSphere(quadric, pulse_radius, 12, 6)  # Simple wireframe sphere
                gluDeleteQuadric(quadric)

                glPopMatrix()

        # Restore OpenGL state
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
        glDepthMask(GL_TRUE)
        glDisable(GL_BLEND)
        glEnable(GL_LIGHTING)
        glPopMatrix()
        glPopAttrib()

    def update_enemies(self, delta_time: float):
        if self.bad_drone_controller:
            # Update bad drones
            self.bad_drone_controller.update_enemies(delta_time)

            # Update good drone controller with new enemy positions
            if self.good_drone_controller:
                self.good_drone_controller.enemies = self.bad_drone_controller.enemies
        elif self.good_drone_controller and self.good_drone_controller.enemies:
            # Fallback - update enemies with proper speed calculation
            for enemy in self.good_drone_controller.enemies:
                x, y, z = enemy.position
                vx, vy, vz = enemy.velocity

                new_x = x + vx * delta_time
                new_y = y + vy * delta_time
                new_z = z + vz * delta_time

                # Simple boundary check
                if new_x < -600 or new_x > 600:
                    vx = -vx * 0.8
                if new_y < -600 or new_y > 600:
                    vy = -vy * 0.8
                if new_z < 30 or new_z > 400:
                    vz = -vz * 0.8

                enemy.position = (new_x, new_y, new_z)
                enemy.velocity = (vx, vy, vz)

    def render_bad_drones(self):
        """Render enemy drones as red spheres"""
        if not self.bad_drone_controller or not self.show_threats:
            return

        if not self.bad_drone_controller.enemies:
            return

        # Save current OpenGL state
        glPushAttrib(GL_ALL_ATTRIB_BITS)
        glMatrixMode(GL_MODELVIEW)
        glPushMatrix()

        # Enable lighting
        glEnable(GL_LIGHTING)
        glEnable(GL_LIGHT0)
        glEnable(GL_COLOR_MATERIAL)
        glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)

        # Set material properties for RED drones
        glMaterialfv(GL_FRONT, GL_SPECULAR, (0.5, 0.2, 0.2, 1.0))  # Reddish specular
        glMaterialf(GL_FRONT, GL_SHININESS, 30.0)

        # Render each enemy drone
        for enemy in self.bad_drone_controller.enemies:
            glPushMatrix()

            # Position
            x, y, z = enemy.position
            glTranslatef(x, y, z)

            current_time = time.time()
            pulse_factor = 0.9 + 0.1 * math.sin(current_time * 4)  # Gentle pulse
            r, g, b = enemy.color
            glColor3f(r * pulse_factor, g * pulse_factor, b * pulse_factor)
            glMaterialfv(GL_FRONT, GL_AMBIENT_AND_DIFFUSE,
                         (r * pulse_factor, g * pulse_factor, b * pulse_factor, 1.0))

            quadric = gluNewQuadric()
            gluSphere(quadric, enemy.size, 16, 16)
            gluDeleteQuadric(quadric)

            glPopMatrix()

        # Restore OpenGL state
        glPopMatrix()
        glMatrixMode(GL_MODELVIEW)
        glPopAttrib()

    def render_hud(self, model: Lahore3DModel):
        glMatrixMode(GL_PROJECTION)
        glPushMatrix()
        glLoadIdentity()
        glOrtho(0, self.screen_width, self.screen_height, 0, -1, 1)
        glMatrixMode(GL_MODELVIEW)
        glPushMatrix()
        glLoadIdentity()
        glDisable(GL_LIGHTING)

        #  semi transparent background for text
        s = pygame.Surface((250, 180), pygame.SRCALPHA)
        s.fill((0, 0, 0, 180))
        self.screen.blit(s, (5, 5))

        s = pygame.Surface((205, 430), pygame.SRCALPHA)
        s.fill((0, 0, 0, 180))
        self.screen.blit(s, (self.screen_width - 210, 5))

        if self.good_drone_controller and self.show_good_drones:
            s = pygame.Surface((250, 100), pygame.SRCALPHA)
            s.fill((0, 20, 0, 180))
            self.screen.blit(s, (5, self.screen_height - 110))
        y_offset = 20

        #stats
        stats_text = [
            "LAHORE 3D DEFENSE SIMULATION",
            f"Buildings: {model.stats.get('total_buildings', 0)} [B]",
            f"Canyons: {model.stats.get('total_canyons', 0)} [C]",
            f"Threats: {model.stats.get('total_threats', 0)} [T]",
            f"Assets: {model.stats.get('total_assets', 0)} [A]",
            f"FPS: {self.fps:.1f}",
            f"Camera: {self.camera_distance:.0f}m"
        ]

        for i, text in enumerate(stats_text):
            color = (255, 255, 255) if i == 0 else (220, 220, 220)
            text_surface = self.small_font.render(text, True, color)
            self.screen.blit(text_surface, (15, y_offset))
            y_offset += 25 if i == 0 else 22

        # Good drone status
        if self.good_drone_controller and self.show_good_drones:
            active_drones = len(self.good_drone_controller.drones)
            enemy_count = 0
            if self.bad_drone_controller:
                enemy_count = len(self.bad_drone_controller.enemies)
            elif self.good_drone_controller.enemies:
                enemy_count = len(self.good_drone_controller.enemies)

            drone_text = [
                "GOOD DRONES [D]",
                f"Drones: {active_drones}",
                f"Enemies: {enemy_count}",
                f"Radar: {'ON' if self.show_radar_pulses else 'OFF'} [P]",
                f"Press 'M' for report",
                f"Press 'I' to init"
            ]

            y_offset_drone = self.screen_height - 100
            for i, text in enumerate(drone_text):
                color = (100, 255, 100) if i == 0 else (180, 255, 180)
                text_surface = self.small_font.render(text, True, color)
                self.screen.blit(text_surface, (15, y_offset_drone))
                y_offset_drone += 22

        controls_text = [
            "CONTROLS:",
            "Mouse Drag: Rotate",
            "Scroll: Zoom",
            "B: Buildings",
            "C: Canyons",
            "T: Threats",
            "A: Assets",
            "D: Drones",
            "P: Radar Pulses",
            "I: Init Drones",
            "M: ASDA/LSTM Report",
            "W: Wireframe",
            "G: Ground",
            "X: Axes",
            "R: Reset Camera",
            "ESC: Exit"
        ]

        y_offset = 20
        for i, text in enumerate(controls_text):
            color = (200, 200, 255) if i == 0 else (180, 220, 255)
            text_surface = self.small_font.render(text, True, color)
            self.screen.blit(text_surface, (self.screen_width - 200, y_offset))
            y_offset += 22

        status_colors = {
            True: (0, 255, 0),
            False: (255, 100, 100)
        }

        status_text = [
            f"Buildings: {'ON' if self.show_buildings else 'OFF'}",
            f"Canyons: {'ON' if self.show_canyons else 'OFF'}",
            f"Threats: {'ON' if self.show_threats else 'OFF'}",
            f"Assets: {'ON' if self.show_assets else 'OFF'}",
            f"Drones: {'ON' if self.show_good_drones else 'OFF'}",
            f"Radar: {'ON' if self.show_radar_pulses else 'OFF'}",
            f"Wireframe: {'ON' if self.show_wireframe else 'OFF'}",
            f"Ground: {'ON' if self.show_ground else 'OFF'}",
            f"Axes: {'ON' if self.show_axes else 'OFF'}"
        ]

        status_values = [
            self.show_buildings,
            self.show_canyons,
            self.show_threats,
            self.show_assets,
            self.show_good_drones,
            self.show_radar_pulses,
            self.show_wireframe,
            self.show_ground,
            self.show_axes
        ]

        y_offset = 370
        for i, (text, value) in enumerate(zip(status_text, status_values)):
            color = status_colors[value]
            text_surface = self.small_font.render(text, True, color)
            self.screen.blit(text_surface, (self.screen_width - 200, y_offset))
            y_offset += 22

        cam_text = f"Camera: X={self.camera_angle_x:.0f}°, Y={self.camera_angle_y:.0f}°"
        text_surface = self.small_font.render(cam_text, True, (255, 200, 200))
        self.screen.blit(text_surface, (15, self.screen_height - 30))
        glEnable(GL_LIGHTING)
        glMatrixMode(GL_PROJECTION)
        glPopMatrix()
        glMatrixMode(GL_MODELVIEW)
        glPopMatrix()

    def calculate_fps(self):
        self.frame_count += 1
        current_time = time.time()
        if current_time - self.last_fps_time > 1.0:
            self.fps = self.frame_count / (current_time - self.last_fps_time)
            self.frame_count = 0
            self.last_fps_time = current_time

    def render(self, model: Lahore3DModel):
        # Clear screen
        glClearColor(0.08, 0.10, 0.15, 1.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        # Setup projection
        self.setup_projection()

        #Update camera
        self.update_camera()

        #update drones
        if self.good_drone_controller and self.show_good_drones:
            self.update_enemies(delta_time=0.016)
            # Then update good drones
            self.good_drone_controller.update_drones(delta_time=0.016)

        if not self.bad_drone_controller:
            self.update_threats(model, 0.016)

        # Render scene
        self.render_ground(model)
        self.render_city_boundary(model)
        self.render_axes()

        #Rendering
        for building in model.buildings:
            self.render_building(building)

        for asset in model.assets:
            self.render_asset(asset)

        for canyon in model.canyons:
            self.render_canyon(canyon, model)  # Pass model as parameter

        self.render_radar_pulses()

        if self.good_drone_controller and self.show_good_drones:
            self.render_good_drones()

        #render bad drones
        if self.bad_drone_controller and self.bad_drone_controller.enemies and self.show_threats:
            self.render_bad_drones()
        else:
            for threat in model.threats:
                self.render_threat(threat)

        self.render_hud(model)

        # Update
        pygame.display.flip()
        self.calculate_fps()

    def update_threats(self, model: Lahore3DModel, delta_time: float):
        current_time = time.time()
        if current_time - self.last_threat_update > self.threat_update_interval:
            #  city bounds
            city_min_x = model.min_x if model.min_x < model.max_x else -600
            city_max_x = model.max_x if model.max_x > model.min_x else 600
            city_min_y = model.min_y if model.min_y < model.max_y else -600
            city_max_y = model.max_y if model.max_y > model.min_y else 600

            buffer = 50 #buffer zone for saftey
            city_min_x -= buffer
            city_max_x += buffer
            city_min_y -= buffer
            city_max_y += buffer

            for threat in model.threats:
                # Update position
                x, y, z = threat.position
                vx, vy, vz = threat.velocity
                new_x = x + vx * 20
                new_y = y + vy * 20
                new_z = max(50, min(z + vz * 20, 300))

                bounce_factor = 0.8

                if new_y < city_min_y or new_y > city_max_y:
                    vy = -vy * bounce_factor
                    new_y = max(city_min_y, min(new_y, city_max_y))

                if new_x < city_min_x or new_x > city_max_x:
                    vx = -vx * bounce_factor
                    new_x = max(city_min_x, min(new_x, city_max_x))


                if new_z < 30 or new_z > 400:
                    vz = -vz * bounce_factor
                    new_z = max(30, min(new_z, 400))

                # Update threat properties
                threat.position = (new_x, new_y, new_z)
                threat.velocity = (vx, vy, vz)

                # Occasionally change direction
                if np.random.random() < 0.02:
                    threat.velocity = (
                        np.random.uniform(-0.3, 0.3),
                        np.random.uniform(-0.3, 0.3),
                        np.random.uniform(-0.1, 0.1)
                    )

            self.last_threat_update = current_time

    def run(self, model: Lahore3DModel):
        clock = pygame.time.Clock()
        running = True

        print("Controls:")
        print("  Mouse Drag: Rotate camera")
        print("  Scroll: Zoom in/out")
        print("  B/C/T/A: Toggle Buildings/Canyons/Threats/Assets")
        print("  D: Toggle Good Drones")
        print("  P: Toggle Radar Pulses")
        print("  I: Initialize Drones")
        print("  M: Show ASDA/LSTM Report")
        print("  ESC: Exit")

        while running:
            running = self.handle_events()
            self.render(model)
            clock.tick(120)

        pygame.quit()
