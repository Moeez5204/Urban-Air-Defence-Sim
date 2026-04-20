import numpy as np
import json
import matplotlib.pyplot as plt


def load_strategic_data(filename='strategic_analysis_3.1.3.json'):


    try:
        with open(filename, 'r') as f:
            data = json.load(f)

        print(f"Loaded {len(data['classified_features']['primary_canyons'])} primary canyons")
        print(f"Loaded {len(data['classified_features']['major_obstacles'])} major obstacles")
        return data

    except FileNotFoundError:
        print(f"Error: {filename} not found. Please run PersistentHomologyAnalysis.py first.")
        return None


def extract_canyon_centerline(canyon_feature, point_cloud, building_density_map):

    canyon_points = find_points_in_canyon_region(canyon_feature, point_cloud)

    if len(canyon_points) < 10:#backup
        return estimate_centerline_from_persistence(canyon_feature)

    # find canyon orientation using PCA
    points_2d = canyon_points[:, :2]
    cov_matrix = np.cov(points_2d.T)
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    major_axis = eigenvectors[:, np.argmax(eigenvalues)]

    projections = points_2d.dot(major_axis)
    min_proj, max_proj = np.min(projections), np.max(projections)

    # centerline
    centerline_length = max_proj - min_proj
    num_points = max(5, int(centerline_length / 20))  # Points every 20 m

    centerline_points = []
    for i in range(num_points):
        t = i / (num_points - 1) if num_points > 1 else 0.5
        proj_val = min_proj + t * (max_proj - min_proj)
        center_point_2d = eigenvectors[:, 0] * proj_val + np.mean(points_2d, axis=0)

        # Estimate height (average of nearby points)
        center_point_3d = np.array([center_point_2d[0], center_point_2d[1], np.mean(canyon_points[:, 2])])
        centerline_points.append(center_point_3d)

    return np.array(centerline_points)


def find_points_in_canyon_region(canyon_feature, point_cloud, radius_multiplier=2.0):

    birth_radius = np.sqrt(canyon_feature['birth']) #use birth death radii to find the points
    death_radius = np.sqrt(canyon_feature['death']) if canyon_feature['death'] != float('inf') else birth_radius * 3
    search_radius = (birth_radius + death_radius) / 2 * radius_multiplier


    if len(point_cloud) > 0:
        cloud_center = np.mean(point_cloud, axis=0)
        feature_location = cloud_center + np.random.uniform(-50, 50, 3)

        distances = np.linalg.norm(point_cloud[:, :3] - feature_location[:3], axis=1)
        within_radius = distances <= search_radius
        canyon_points = point_cloud[within_radius]

        if len(canyon_points) >= 10:
            return canyon_points

    return np.array([])


def estimate_centerline_from_persistence(canyon_feature):

    persistence = canyon_feature['persistence']
    length_estimate = np.sqrt(persistence) * 2

    num_points = max(3, int(length_estimate / 25))

    centerline_points = []
    for i in range(num_points):
        t = i / (num_points - 1) if num_points > 1 else 0.5
        point = np.array([
            t * length_estimate,  0,  15 + np.sin(t * np.pi) * 5  ])
        centerline_points.append(point)

    return np.array(centerline_points)


def calculate_threat_level(feature, feature_type, urban_context):

    base_persistence = feature['persistence']

    if feature_type == 'canyon':
        # more strategic value with higher persistance
        persistence_factor = min(1.0, base_persistence / 1000)
        concealment = calculate_concealment_value(feature, urban_context)
        accessibility = 0.8  # Canyons are highly accessible for aircraft

        threat_score = (persistence_factor * 0.4 +
                        concealment * 0.4 +
                        accessibility * 0.2)

    elif feature_type == 'obstacle':
        #obstaclescreate radar shadows which can be more challenging
        persistence_factor = min(1.0, base_persistence / 500)
        height_factor = min(1.0, feature.get('estimated_height', 30) / 100)

        threat_score = (persistence_factor * 0.3 +
                        height_factor * 0.7)

    elif feature_type == 'void':
        # voids are open areas
        persistence_factor = min(1.0, base_persistence / 800)
        threat_score = persistence_factor * 0.6


    if threat_score >= 0.7:
        return 'high', threat_score
    elif threat_score >= 0.4:
        return 'medium', threat_score
    else:
        return 'low', threat_score


def calculate_concealment_value(feature, urban_context):

    persistence = feature['persistence']

    if feature.get('type', '').startswith('canyon'):
        concealment = min(1.0, persistence / 800)
        urban_density = urban_context.get('local_density', 0.5)
        concealment = (concealment * 0.7 + urban_density * 0.3)

    elif feature.get('type', '').startswith('obstacle'):
        concealment = min(1.0, persistence / 400)

    else:
        concealment = min(1.0, persistence / 600)

    return concealment


def analyze_urban_context(point_cloud, feature_locations):

    context = {}

    for i, location in enumerate(feature_locations):
        context[i] = {
            'local_density': np.random.uniform(0.3, 0.9),
            'avg_building_height': np.random.uniform(10, 50),
            'height_variance': np.random.uniform(5, 20)
        }

    return context


def create_strategic_feature_database(classified_features, point_cloud):
    print("Creating strategic feature database...")

    urban_context = analyze_urban_context(point_cloud, [])

    strategic_database = {
        'canyons': [],
        'obstacles': [],
        'voids': [],
        'engagement_zones': []
    }

    # Process primary canyons
    for i, canyon in enumerate(classified_features['primary_canyons']):
        centerline = extract_canyon_centerline(canyon, point_cloud, urban_context)
        threat_level, threat_score = calculate_threat_level(canyon, 'canyon', urban_context)
        concealment = calculate_concealment_value(canyon, urban_context)

        strategic_feature = {
            'id': f'canyon_primary_{i}',
            'persistence': canyon['persistence'],
            'threat_level': threat_level,
            'threat_score': threat_score,
            'centerline': centerline.tolist(),
            'concealment_value': concealment,
            'type': 'urban_canyon',
            'birth': canyon['birth'],
            'death': canyon['death'],
            'strategic_value': 'high',
            'engagement_priority': 1 if threat_level == 'high' else 2
        }
        strategic_database['canyons'].append(strategic_feature)

    for i, obstacle in enumerate(classified_features['major_obstacles']): #obstacles
        threat_level, threat_score = calculate_threat_level(obstacle, 'obstacle', urban_context)
        concealment = calculate_concealment_value(obstacle, urban_context)

        strategic_feature = {
            'id': f'obstacle_major_{i}',
            'persistence': obstacle['persistence'],
            'threat_level': threat_level,
            'threat_score': threat_score,
            'centerline': [],  # Obstacles don't have centerlines
            'concealment_value': concealment,
            'type': 'urban_obstacle',
            'birth': obstacle['birth'],
            'death': obstacle['death'],
            'strategic_value': 'high',
            'engagement_priority': 1,
            'estimated_height': obstacle['persistence'] / 10
        }
        strategic_database['obstacles'].append(strategic_feature)

    for i, void in enumerate(classified_features['strategic_voids']): # voids
        threat_level, threat_score = calculate_threat_level(void, 'void', urban_context)

        strategic_feature = {
            'id': f'void_strategic_{i}',
            'persistence': void['persistence'],
            'threat_level': threat_level,
            'threat_score': threat_score,
            'centerline': [],
            'concealment_value': 0.1,
            'type': 'urban_void',
            'birth': void['birth'],
            'death': void['death'],
            'strategic_value': 'medium',
            'engagement_priority': 3,
            'is_engagement_zone': True
        }
        strategic_database['voids'].append(strategic_feature)
        strategic_database['engagement_zones'].append(strategic_feature)

    print(f"Strategic database created: {len(strategic_database['canyons'])} canyons, "
          f"{len(strategic_database['obstacles'])} obstacles, "
          f"{len(strategic_database['voids'])} voids")

    return strategic_database


def visualize_strategic_features(strategic_database, point_cloud):

    print("Generating strategic feature visualization...")

    fig = plt.figure(figsize=(15, 10))

    ax1 = fig.add_subplot(221, projection='3d')

    ax1.scatter(point_cloud[:, 0], point_cloud[:, 1], point_cloud[:, 2],
                c='lightgray', alpha=0.3, s=1, label='Urban Model')

    for canyon in strategic_database['canyons']:
        if len(canyon['centerline']) > 0:
            centerline = np.array(canyon['centerline'])
            color = 'red' if canyon['threat_level'] == 'high' else 'orange'
            ax1.plot(centerline[:, 0], centerline[:, 1], centerline[:, 2],
                     color=color, linewidth=2, marker='o', markersize=3,
                     label=f"Canyon ({canyon['threat_level']})")

    for obstacle in strategic_database['obstacles']:
        ax1.scatter(obstacle['birth'], obstacle['death'], obstacle['persistence'] / 10,
                    c='blue', s=50, marker='^', label='Major Obstacle')

    ax1.set_xlabel('Easting (m)')
    ax1.set_ylabel('Northing (m)')
    ax1.set_zlabel('Height (m)')
    ax1.set_title('Strategic Feature Map')

    ax2 = fig.add_subplot(222)
    threat_levels = [f['threat_level'] for f in strategic_database['canyons'] +
                     strategic_database['obstacles'] + strategic_database['voids']]

    threat_counts = {
        'high': threat_levels.count('high'),
        'medium': threat_levels.count('medium'),
        'low': threat_levels.count('low')
    }

    colors = ['red', 'orange', 'green']
    bars = ax2.bar(threat_counts.keys(), threat_counts.values(), color=colors, alpha=0.7)

    for bar, count in zip(bars, threat_counts.values()):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                 f'{count}', ha='center', va='bottom', fontweight='bold')

    ax2.set_ylabel('Number of Features')
    ax2.set_title('Threat Level Distribution')
    ax2.grid(True, alpha=0.3)

    ax3 = fig.add_subplot(223)
    feature_types = ['Canyons', 'Obstacles', 'Voids']
    feature_counts = [
        len(strategic_database['canyons']),
        len(strategic_database['obstacles']),
        len(strategic_database['voids'])
    ]

    bars = ax3.bar(feature_types, feature_counts, color=['red', 'blue', 'green'], alpha=0.7)

    for bar, count in zip(bars, feature_counts):
        ax3.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                 f'{count}', ha='center', va='bottom', fontweight='bold')

    ax3.set_ylabel('Count')
    ax3.set_title('Strategic Feature Types')
    ax3.grid(True, alpha=0.3)

    # Strategic summary
    ax4 = fig.add_subplot(224)
    ax4.axis('off')

    high_threat = threat_counts['high']
    total_features = sum(feature_counts)
    engagement_zones = len(strategic_database['engagement_zones'])

    summary_text = (
        "STRATEGIC ASSESSMENT\n\n"
        f"High Threat Features: {high_threat}\n"
        f"Total Features: {total_features}\n"
        f"Engagement Zones: {engagement_zones}\n"
        f"Primary Canyons: {len(strategic_database['canyons'])}\n"
        f"Major Obstacles: {len(strategic_database['obstacles'])}\n\n"
        "READINESS: OPERATIONAL"
    )

    ax4.text(0.1, 0.9, summary_text, transform=ax4.transAxes, fontsize=12,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))

    ax4.set_title('Military Assessment')

    plt.tight_layout()
    plt.show()

    return fig


def export_strategic_database(strategic_database, filename='strategic_features_3.1.4.json'):


    export_data = {
        'strategic_features': strategic_database,
        'metadata': {
            'total_canyons': len(strategic_database['canyons']),
            'total_obstacles': len(strategic_database['obstacles']),
            'total_voids': len(strategic_database['voids']),
            'high_threat_features': len([f for f in strategic_database['canyons'] +
                                         strategic_database['obstacles'] +
                                         strategic_database['voids']
                                         if f['threat_level'] == 'high']),
            'export_timestamp': np.datetime64('now').astype(str)
        }
    }

    with open(filename, 'w') as f:
        json.dump(export_data, f, indent=2)

    print(f"Strategic features exported to {filename}")
    return export_data


def run_phase_3_1_4():

    print("Strategic Feature Extraction")

    strategic_data = load_strategic_data()
    if strategic_data is None:
        return None

    point_cloud = np.array(strategic_data.get('point_cloud_sample', []))
    if len(point_cloud) == 0:
        print("Warning: No point cloud data available, using synthetic points")
        point_cloud = np.random.randn(1000, 3) * 100  # Fallback

    strategic_database = create_strategic_feature_database(
        strategic_data['classified_features'],
        point_cloud
    )

    visualize_strategic_features(strategic_database, point_cloud)
    export_result = export_strategic_database(strategic_database)


    print(f"High threat features: {export_result['metadata']['high_threat_features']}")
    print(f"Primary canyons: {export_result['metadata']['total_canyons']}")
    print(f"Major obstacles: {export_result['metadata']['total_obstacles']}")
    print("Ready for Phase 3.2: Topology-Aware Predictive Tracking")

    return strategic_database


if __name__ == "__main__":
    run_phase_3_1_4()