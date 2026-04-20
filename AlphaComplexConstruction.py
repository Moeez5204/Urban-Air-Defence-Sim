import numpy as np
import gudhi as gd
from shapely.geometry import Point
import matplotlib.pyplot as plt
import random
import json
from shapely.geometry import shape
import utm


def load_building_data_from_export(filename='building_data_3.1.1.json'):

    print(f"Loading building data from {filename}")

    try:
        with open(filename, 'r') as f:
            data = json.load(f)

        building_footprints = []
        for item in data:
            building = {
                'height': item['height'],
                'stories': item['stories'],
                'center': tuple(item['center']),
                'zone': item['zone'],
                'area': item['area'],
                'footprint': shape(item['footprint'])
            }
            building_footprints.append(building)

        print(f"loaded {len(building_footprints)} buildings from {filename}")
        return building_footprints

    except FileNotFoundError:
        print(f"error: {filename} not found. Run CityModelReconstruct.py first.")
        return None
    except Exception as e:
        print(f"failed")
        return None


def sample_building_points(building_footprints, points_per_building=8):

    # Convert continuous building surfaces to point cloud

    point_cloud = []
    building_labels = []

    print("Sampling building points for alpha complex")

    for i, building in enumerate(building_footprints):
        if i % 100 == 0:
            print(f"Processing building {i}/{len(building_footprints)}")

        poly = building['footprint']
        height = building['height']
        zone = building['zone']

        # Sample roof points
        roof_points = sample_polygon_surface(poly, height, max(4, points_per_building))
        point_cloud.extend(roof_points)
        building_labels.extend([f"{zone}_roof_{i}"] * len(roof_points))

        # Sample ground points
        ground_points = sample_polygon_surface(poly, 0, max(2, points_per_building // 3))
        point_cloud.extend(ground_points)
        building_labels.extend([f"{zone}_ground_{i}"] * len(ground_points))

        #Sample mid-height points
        if height > 16:
            mid_points = sample_polygon_surface(poly, height / 2, 2)
            point_cloud.extend(mid_points)
            building_labels.extend([f"{zone}_mid_{i}"] * len(mid_points))

    print(f"Generated {len(point_cloud)} total points from {len(building_footprints)} buildings")
    return np.array(point_cloud), building_labels


def sample_polygon_surface(polygon, z_height, num_points):
  # monte carlo smapling
    points = []
    bounds = polygon.bounds
    max_attempts = num_points * 50

    attempts = 0
    while len(points) < num_points and attempts < max_attempts:
        x = random.uniform(bounds[0], bounds[2])
        y = random.uniform(bounds[1], bounds[3])
        if polygon.contains(Point(x, y)):
            points.append([x, y, z_height])
        attempts += 1

    while len(points) < num_points:
        if len(points) == 0:
            centroid = polygon.centroid
            points.append([centroid.x, centroid.y, z_height])
        else:
            exterior = list(polygon.exterior.coords)
            point = exterior[len(points) % len(exterior)]
            points.append([point[0], point[1], z_height])

    return points


def project_to_utm(point_cloud):
    print("point cloud to meters")
    projected = []
    for lon, lat, h in point_cloud:
        easting, northing, zone_num, zone_letter = utm.from_latlon(lat, lon)
        projected.append([easting, northing, h])
    return np.array(projected)


def build_alpha_complex(point_cloud, max_alpha_square=2500):

    try:
        alpha_complex = gd.AlphaComplex(points=point_cloud)
        simplex_tree = alpha_complex.create_simplex_tree(max_alpha_square=max_alpha_square)
        persistence = simplex_tree.persistence()

        print(f"Alpha Complex constructed with {simplex_tree.num_vertices()} vertices")
        print(f"Total simplices: {simplex_tree.num_simplices()}")

        return simplex_tree, persistence

    except:
        return None


def analyze_topological_features(simplex_tree, persistence):

    #classify topological features from persistance
    features = {
        'components': [], 'loops': [], 'voids': [],'canyons': [], 'obstacles': [] }

    if persistence is None:
        return features

    for dim, (birth, death) in persistence:
        persistence_value = death - birth
        if dim == 0:
            feature_data = {
                'birth': birth,
                'death': death,
                'persistence': persistence_value,
                'type': 'component'
            }
            features['components'].append(feature_data)
            if persistence_value > 100:  # 10m scale
                features['obstacles'].append(feature_data)

        elif dim == 1:
            feature_data = {
                'birth': birth,
                'death': death,
                'persistence': persistence_value,
                'type': 'canyon_loop'
            }
            features['loops'].append(feature_data)
            if persistence_value > 400:  # 20m scale
                features['canyons'].append(feature_data)

        elif dim == 2:
            features['voids'].append({
                'birth': birth,
                'death': death,
                'persistence': persistence_value,
                'type': 'urban_void'
            })

    for key in features:
        if features[key]:
            features[key].sort(key=lambda x: x['persistence'], reverse=True)

    print(f"Components: {len(features['components'])}, Loops: {len(features['loops'])}, Voids: {len(features['voids'])}")
    print(f"Canyons: {len(features['canyons'])}, Obstacles: {len(features['obstacles'])}")

    return features


def visualize_alpha_complex(point_cloud, simplex_tree, features):
    #visualize point cloud

    fig = plt.figure(figsize=(15, 5))

    ax1 = fig.add_subplot(131, projection='3d')
    ax1.scatter(point_cloud[:, 0], point_cloud[:, 1], point_cloud[:, 2],
                c='blue', alpha=0.6, s=1)
    ax1.set_title('3D Point Cloud\n(UTM Projected)')
    ax1.set_xlabel('Easting (m)')
    ax1.set_ylabel('Northing (m)')
    ax1.set_zlabel('Height (m)')

    ax2 = fig.add_subplot(132)
    if simplex_tree is not None:
        persistence = simplex_tree.persistence()
        for dim in range(3):
            births = [birth for (d, (birth, death)) in persistence if d == dim]
            deaths = [death if death != float('inf') else birth + 1000 for (d, (birth, death)) in persistence if d == dim]
            if births:
                color = ['blue', 'red', 'green'][dim]
                ax2.scatter(births, deaths, c=color, label=f'Dim {dim}', alpha=0.7)
    ax2.plot([0, 10000], [0, 10000], 'k--', alpha=0.5)
    ax2.set_xlabel('Birth (m²)')
    ax2.set_ylabel('Death (m²)')
    ax2.set_title('Persistence Diagram')
    ax2.legend()

    ax3 = fig.add_subplot(133)
    summary_text = "Topological Features:\n\n"
    summary_text += f"Components (H0): {len(features['components'])}\n"
    summary_text += f"Loops (H1): {len(features['loops'])}\n"
    summary_text += f"Voids (H2): {len(features['voids'])}\n\n"
    summary_text += f"Urban Canyons: {len(features['canyons'])}\n"
    summary_text += f"Major Obstacles: {len(features['obstacles'])}\n"
    ax3.text(0.1, 0.9, summary_text, transform=ax3.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace')
    ax3.set_title('Feature Summary')

    plt.tight_layout()
    plt.show()
    return fig


def export_topological_data(features, point_cloud, filename='topological_features_3.1.2.json'):
    export_data = {
        'features': features,
        'point_cloud_sample': point_cloud[:1000].tolist() if len(point_cloud) > 1000 else point_cloud.tolist(),
        'summary': {
            'total_points': len(point_cloud),
            'canyons_count': len(features['canyons']),
            'obstacles_count': len(features['obstacles']),
            'total_features': len(features['components']) + len(features['loops']) + len(features['voids'])
        }
    }

    with open(filename, 'w') as f:
        json.dump(export_data, f, indent=2)

    print(f"Topological features exported to {filename}")


def run_phase_3_1_2():
    building_footprints = load_building_data_from_export()
    if building_footprints is None:
        return None

    point_cloud, labels = sample_building_points(building_footprints)
    point_cloud = project_to_utm(point_cloud)

    simplex_tree, persistence = build_alpha_complex(point_cloud, max_alpha_square=2500)  # 50m scale

    features = analyze_topological_features(simplex_tree, persistence)
    visualize_alpha_complex(point_cloud, simplex_tree, features)
    export_topological_data(features, point_cloud)

    return features, point_cloud, simplex_tree


if __name__ == "__main__":
    run_phase_3_1_2()