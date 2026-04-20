import numpy as np
import matplotlib.pyplot as plt
import json


def load_topological_data(filename='topological_features_3.1.2.json'):



    try:
        with open(filename, 'r') as f:
            data = json.load(f)

        print(f"loaded topological features")
        print(f"Summary: {data['summary']}")
        return data

    except FileNotFoundError:
        print(f"Error: {filename} not found. Please run AlphaComplexConstruction.py first.")
        return None


def analyze_persistence_barcodes(features):

    h0_persistence = [f['persistence'] for f in features['components']]
    h1_persistence = [f['persistence'] for f in features['loops']]
    h2_persistence = [f['persistence'] for f in features['voids']]

    stats = {
        'h0': {
            'mean': np.mean(h0_persistence) if h0_persistence else 0,
            'std': np.std(h0_persistence) if h0_persistence else 0,
            'max': max(h0_persistence) if h0_persistence else 0,
            'count': len(h0_persistence)
        },
        'h1': {
            'mean': np.mean(h1_persistence) if h1_persistence else 0,
            'std': np.std(h1_persistence) if h1_persistence else 0,
            'max': max(h1_persistence) if h1_persistence else 0,
            'count': len(h1_persistence)
        },
        'h2': {
            'mean': np.mean(h2_persistence) if h2_persistence else 0,
            'std': np.std(h2_persistence) if h2_persistence else 0,
            'max': max(h2_persistence) if h2_persistence else 0,
            'count': len(h2_persistence)
        }
    }

    print("Persistence statistics:")
    print(f"  H0 (Components): {stats['h0']['count']} features, max persistence: {stats['h0']['max']:.2f} m²")
    print(f"  H1 (Loops): {stats['h1']['count']} features, max persistence: {stats['h1']['max']:.2f} m²")
    print(f"  H2 (Voids): {stats['h2']['count']} features, max persistence: {stats['h2']['max']:.2f} m²")

    return stats, h0_persistence, h1_persistence, h2_persistence


def classify_urban_features(features, stats): #classify the features based off the persistance thresholds

    h1_threshold = stats['h1']['mean'] + 1.5 * stats['h1']['std']
    h0_threshold = stats['h0']['mean'] + 1.5 * stats['h0']['std']

    print(f"Adaptive thresholds:")
    print(f"  Canyon threshold: {h1_threshold:.2f} m²")
    print(f"  Obstacle threshold: {h0_threshold:.2f} m²")

    classified_features = {
        'primary_canyons': [], 'secondary_canyons': [],
        'major_obstacles': [], 'minor_obstacles': [], 'strategic_voids': []
    }

    for feature in features['loops']:
        if feature['persistence'] > h1_threshold:
            classified_features['primary_canyons'].append({**feature, 'strategic_value': 'high', 'type': 'primary_canyon'})
        else:
            classified_features['secondary_canyons'].append({**feature, 'strategic_value': 'medium', 'type': 'secondary_canyon'})

    for feature in features['components']:
        if feature['persistence'] > h0_threshold:
            classified_features['major_obstacles'].append({**feature, 'strategic_value': 'high', 'type': 'major_obstacle'})
        else:
            classified_features['minor_obstacles'].append({**feature, 'strategic_value': 'low', 'type': 'minor_obstacle'})

    for feature in features['voids']:
        classified_features['strategic_voids'].append({**feature, 'strategic_value': 'high', 'type': 'strategic_void'})

    print(f"Classification Results:")
    print(f"  Primary Canyons: {len(classified_features['primary_canyons'])}")
    print(f"  Secondary Canyons: {len(classified_features['secondary_canyons'])}")
    print(f"  Major Obstacles: {len(classified_features['major_obstacles'])}")
    print(f"  Minor Obstacles: {len(classified_features['minor_obstacles'])}")
    print(f"  Strategic Voids: {len(classified_features['strategic_voids'])}")

    return classified_features


def visualize_persistence_barcodes(h0_persistence, h1_persistence, h2_persistence, classified_features):


    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

    top_n_h0,top_n_h1 = 50,100
    top_n_h2 = 50

    h0_top = sorted(h0_persistence, reverse=True)[:top_n_h0]
    h1_top = sorted(h1_persistence, reverse=True)[:top_n_h1]
    h2_top = sorted(h2_persistence, reverse=True)[:top_n_h2]
    dimensions = [h0_top, h1_top, h2_top]
    colors = ['blue', 'red', 'green']
    labels = ['H0 Components', 'H1 Loops (Canyons)', 'H2 Voids']

    for dim, (persistence, color, label) in enumerate(zip(dimensions, colors, labels)):
        for i, persist_val in enumerate(persistence):
            ax1.barh(f'{label}_{i}', persist_val, color=color, alpha=0.7, height=0.8)

    ax1.set_xlabel('Persistence (m²)')
    ax1.set_title('Persistence Barcodes\n(Top Features Only — Clean View)')
    ax1.grid(True, alpha=0.3)

    categories = ['Primary\nCanyons', 'Secondary\nCanyons', 'Major\nObstacles', 'Minor\nObstacles', 'Strategic\nVoids']
    counts = [
        len(classified_features['primary_canyons']),
        len(classified_features['secondary_canyons']),
        len(classified_features['major_obstacles']),
        len(classified_features['minor_obstacles']),
        len(classified_features['strategic_voids'])
    ]

    colors_bar = ['darkred', 'red', 'darkblue', 'lightblue', 'green']
    bars = ax2.bar(categories, counts, color=colors_bar, alpha=0.8)

    for bar, count in zip(bars, counts):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2., height + 0.5,
                 f'{count}', ha='center', va='bottom', fontweight='bold')

    ax2.set_ylabel('Number of Features')
    ax2.set_title('Strategic Urban Feature Classification')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()
    return fig


def calculate_strategic_metrics(classified_features):

    metrics = {}

    if classified_features['primary_canyons']:
        canyon_p = [c['persistence'] for c in classified_features['primary_canyons']]
        metrics['canyon_stability'] = np.mean(canyon_p)
        metrics['dominant_canyon'] = max(canyon_p)
    else:
         metrics['dominant_canyon'] = 0

    metrics['obstacle_density'] = len(classified_features['major_obstacles'])
    metrics['urban_complexity'] = sum(len(f) for f in classified_features.values())
    metrics['defense_suitability'] = (metrics['canyon_stability'] + metrics['obstacle_density'] * 100) / 100

    print(" strategic Metrics:")
    print(f"  Canyon Stability: {metrics['canyon_stability']:.2f} m²")
    print(f"  Obstacle Density: {metrics['obstacle_density']}")
    print(f"  Defense Suitability: {metrics['defense_suitability']:.2f}")

    return metrics


def export_strategic_analysis(classified_features, metrics, filename='strategic_analysis_3.1.3.json'):
    export_data = {
        'classified_features': classified_features,
        'strategic_metrics': metrics,
        'summary': {
            'primary_canyons': len(classified_features['primary_canyons']),
            'major_obstacles': len(classified_features['major_obstacles']),
            'defense_suitability': metrics['defense_suitability']
        }
    }
    with open(filename, 'w') as f:
        json.dump(export_data, f, indent=2)
    print("exported ")
    return export_data


def run_phase_3_1_3():

    print("Phase 3.1.3: persistent homology analysis")

    topological_data = load_topological_data()
    if topological_data is None:
        return None

    features = topological_data['features']
    stats, h0_persistence, h1_persistence, h2_persistence = analyze_persistence_barcodes(features)
    classified_features = classify_urban_features(features, stats)
    visualize_persistence_barcodes(h0_persistence, h1_persistence, h2_persistence, classified_features)
    metrics = calculate_strategic_metrics(classified_features)
    strategic_analysis = export_strategic_analysis(classified_features, metrics)

    print(f"Primary Canyons: {len(classified_features['primary_canyons'])}")
    print(f"Major Obstacles: {len(classified_features['major_obstacles'])}")
    print(f"Defense Suitability: {metrics['defense_suitability']:.2f}")

    return strategic_analysis


if __name__ == "__main__":
    run_phase_3_1_3()