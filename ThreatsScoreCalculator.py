import numpy as np
import json
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
from scipy.spatial import KDTree


@dataclass
class ThreatParameters:
    distance_weight: float = 0.30
    approach_angle_weight: float = 0.15
    speed_weight: float = 0.20
    urban_context_weight: float = 0.35
    max_threat_score: float = 10.0
    critical_distance: float = 500.0
    critical_speed: float = 100.0


@dataclass
class TargetState:
    position: np.ndarray
    velocity: np.ndarray
    timestamp: float
    target_id: str
    asset_type: str
    sector: Optional[str] = None


@dataclass
class DefendedAsset:
    position: np.ndarray
    asset_id: str
    asset_name: str
    priority: float
    critical_radius: float
    asset_type: str
    description: str = ""
    sector: str = ""


class UrbanThreatContext:
    #Analyse urban topological

    def __init__(self, strategic_features_file='lahore_strategic_features.json'):
        self.strategic_features = self.load_lahore_strategic_features(strategic_features_file)
        self.feature_tree = None
        self.build_feature_tree()

    def load_lahore_strategic_features(self, filename):
        try:
            with open(filename, 'r') as f:
                data = json.load(f)
            return data['strategic_features']
        except FileNotFoundError:
            return self.create_default_lahore_features()

    def create_default_lahore_features(self):
        return {
            'canyons': [
                {
                    'name': 'The_Mall_Road_Corridor',
                    'centerline': [[74.33, 31.55], [74.34, 31.56], [74.35, 31.57]],
                    'concealment_value': 0.8,
                    'sector': 'Central_Lahore'
                },
                {
                    'name': 'Gulberg_Main_Boulevard',
                    'centerline': [[74.34, 31.51], [74.35, 31.52], [74.36, 31.53]],
                    'concealment_value': 0.7,
                    'sector': 'Gulberg'
                }
            ],
            'obstacles': [
                {
                    'name': 'Minar_e_Pakistan',
                    'position': [74.3093, 31.5883],
                    'concealment_value': 0.6,
                    'sector': 'Walled_City'
                }
            ]
        }

    def build_feature_tree(self):
        feature_points,feature_data= [], []

        for canyon in self.strategic_features['canyons']:
            if 'centerline' in canyon:
                for point in canyon['centerline']:
                    feature_points.append(point[:2])
                    feature_data.append({
                        'type': 'canyon',
                        'name': canyon.get('name', 'unknown'),
                        'concealment': canyon.get('concealment_value', 0),
                        'sector': canyon.get('sector', 'unknown')
                    })

        for obstacle in self.strategic_features['obstacles']:
            if 'position' in obstacle:
                feature_points.append(obstacle['position'][:2])
                feature_data.append({
                    'type': 'obstacle',
                    'name': obstacle.get('name', 'unknown'),
                    'concealment': obstacle.get('concealment_value', 0),
                    'sector': obstacle.get('sector', 'unknown')
                })

        if feature_points:
            self.feature_tree = KDTree(feature_points)
            self.feature_data = feature_data

    def get_urban_context(self, position: np.ndarray, max_distance: float = 0.01) -> Dict:
        #Get urban topological context
        if self.feature_tree is None or len(self.feature_data) == 0:
            return self.get_default_context(position)

        position_2d = position[:2]
        distances, indices = self.feature_tree.query(position_2d, k=min(3, len(self.feature_data)))

        if distances[0] < max_distance:
            closest_feature = self.feature_data[indices[0]]

            context = {
                'type': closest_feature['type'],
                'name': closest_feature['name'],
                'concealment': closest_feature['concealment'],
                'sector': closest_feature.get('sector', 'unknown'),
                'distance_to_feature': distances[0]
            }

            if closest_feature['type'] == 'canyon':
                context['threat_advantage'] = 0.7 + (closest_feature['concealment'] * 0.3)
            elif closest_feature['type'] == 'obstacle':
                context['threat_advantage'] = 0.6 + (closest_feature['concealment'] * 0.4)
            else:
                context['threat_advantage'] = 0.3

            return context

        return self.get_default_context(position)

    def get_default_context(self, position: np.ndarray) -> Dict: #get the urban context of the area
        return {
            'type': 'open_area',
            'name': 'open_area',
            'concealment': 0.1,
            'threat_advantage': 0.1,
            'sector': self.get_lahore_sector(position[:2]),
            'distance_to_feature': float('inf')
        }

    def get_lahore_sector(self, position: np.ndarray) -> str:
        longitude, latitude = position

        if 74.30 <= longitude <= 74.32 and 31.58 <= latitude <= 31.60:
            return "Walled_City"
        elif 74.36 <= longitude <= 74.38 and 31.53 <= latitude <= 31.55:
            return "Cantonment"
        elif 74.33 <= longitude <= 74.36 and 31.50 <= latitude <= 31.53:
            return "Gulberg"
        elif 74.32 <= longitude <= 74.36 and 31.55 <= latitude <= 31.58:
            return "Central_Lahore"
        else:
            return "Other_Sector"


class ThreatScoreCalculator:

    def __init__(self, defended_assets: List[DefendedAsset], params: ThreatParameters = None):
        self.params = params or ThreatParameters()
        self.defended_assets = defended_assets
        self.urban_context = UrbanThreatContext()
        self.threat_history = []

    def calculate_threat_score(self, target: TargetState, asset_id: str = None) -> Dict:
        if asset_id:
            asset = next((a for a in self.defended_assets if a.asset_id == asset_id), None)
            if asset is None:
                raise ValueError(f"Asset {asset_id} not found")
            return self.calculate_single_threat(target, asset)
        else:
            max_threat = {'overall_score': 0}
            for asset in self.defended_assets:
                threat = self.calculate_single_threat(target, asset)
                if threat['overall_score'] > max_threat['overall_score']:
                    max_threat = threat
            return max_threat

    def calculate_single_threat(self, target: TargetState, asset: DefendedAsset) -> Dict:

        distance = self.haversine_distance(target.position, asset.position)
        distance_score = self.calculate_distance_score(distance, asset.critical_radius)

        approach_angle_score = self.calculate_approach_angle_score(target, asset)

        speed = np.linalg.norm(target.velocity[:2]) * 111000
        speed_score = self.calculate_speed_score(speed)

        urban_context = self.urban_context.get_urban_context(target.position)
        urban_context_score = urban_context['threat_advantage']

        overall_score = (
                                self.params.distance_weight * distance_score +
                                self.params.approach_angle_weight * approach_angle_score +
                                self.params.speed_weight * speed_score +
                                self.params.urban_context_weight * urban_context_score
                        ) * self.params.max_threat_score

        threat_data = {
            'timestamp': target.timestamp,
            'target_id': target.target_id,
            'target_type': target.asset_type,
            'asset_id': asset.asset_id,
            'asset_name': asset.asset_name,
            'asset_type': asset.asset_type,
            'asset_sector': asset.sector,
            'overall_score': overall_score,
            'distance_to_asset': distance,
            'urban_context': urban_context,
            'target_position': target.position.tolist(),
            'target_sector': target.sector or urban_context['sector']
        }

        self.threat_history.append(threat_data)
        return threat_data

    def haversine_distance(self, pos1: np.ndarray, pos2: np.ndarray) -> float:
        lon1, lat1, _ = np.radians(pos1)
        lon2, lat2, _ = np.radians(pos2)
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
        return 2 * 6371000 * np.arcsin(np.sqrt(a))

    def calculate_distance_score(self, distance: float, critical_radius: float) -> float:
        if distance <= 0:
            return 1.0
        normalized_distance = distance / self.params.critical_distance
        score = 1.0 / (1.0 + np.exp((normalized_distance - 0.5) * 4))
        if distance < critical_radius:
            score = min(1.0, score * 1.5)
        return score

    def calculate_approach_angle_score(self, target: TargetState, asset: DefendedAsset) -> float:
        to_asset = asset.position - target.position
        to_asset_normalized = to_asset / max(np.linalg.norm(to_asset), 1e-6)
        velocity_normalized = target.velocity / max(np.linalg.norm(target.velocity), 1e-6)
        cosine_angle = np.dot(velocity_normalized, to_asset_normalized)
        return max(0, cosine_angle)

    def calculate_speed_score(self, speed: float) -> float:
        normalized_speed = min(1.0, speed / self.params.critical_speed)
        return 1.0 - np.exp(-normalized_speed * 3)

    def assess_multiple_threats(self, targets: List[TargetState]) -> List[Dict]:
        threats = [self.calculate_threat_score(target) for target in targets]
        threats.sort(key=lambda x: x['overall_score'], reverse=True)
        return threats

    def get_threat_level(self, score: float) -> str:
        if score >= self.params.max_threat_score * 0.8:
            return "CRITICAL"
        elif score >= self.params.max_threat_score * 0.6:
            return "HIGH"
        elif score >= self.params.max_threat_score * 0.4:
            return "MEDIUM"
        elif score >= self.params.max_threat_score * 0.2:
            return "LOW"
        else:
            return "MINIMAL"

    def visualize_threat_assessment(self, targets: List[TargetState]):

        fig = plt.figure(figsize=(15, 5))
        threats = self.assess_multiple_threats(targets)
        ax1 = plt.subplot(1, 3, 1)

        threat_names = [t['target_id'] for t in threats]
        threat_scores = [t['overall_score'] for t in threats]
        threat_colors = []

        for score in threat_scores:
            if score >= 8:
                color = 'red'
            elif score >= 6:
                color = 'orange'
            elif score >= 4:
                color = 'yellow'
            elif score >= 2:
                color = 'lightgreen'
            else:
                color = 'green'
            threat_colors.append(color)

        bars = ax1.barh(threat_names, threat_scores, color=threat_colors, alpha=0.8)
        ax1.set_xlabel('Threat Score (0-10)')
        ax1.set_title('Top Threats Ranking')
        ax1.set_xlim(0, 10)

        for i, (bar, score) in enumerate(zip(bars, threat_scores)):
            ax1.text(score + 0.1, bar.get_y() + bar.get_height() / 2,
                     f'{score:.1f}', va='center', fontweight='bold')

        # radaar chart
        ax2 = plt.subplot(1, 3, 2, polar=True)

        if threats:
            top_threat = threats[0]
            components = ['Distance', 'Angle', 'Speed', 'Urban']

            # Get component scores
            urban_ctx = top_threat['urban_context']
            urban_score = urban_ctx['threat_advantage']
            distance_score = max(0.3, min(0.9, 1 - (top_threat['distance_to_asset'] / 2000)))

            values = [distance_score, 0.7, 0.5, urban_score]
            values += values[:1]

            angles = np.linspace(0, 2 * np.pi, len(components), endpoint=False).tolist()
            angles += angles[:1]

            ax2.plot(angles, values, 'o-', linewidth=2, color='red')
            ax2.fill(angles, values, alpha=0.25, color='red')
            ax2.set_xticks(angles[:-1])
            ax2.set_xticklabels(components)
            ax2.set_ylim(0, 1)
            ax2.set_title(f'Threat Profile: {top_threat["target_id"]}', pad=20)
            ax2.grid(True)

        # heat map
        ax3 = plt.subplot(1, 3, 3)

        sector_threats = {}
        for threat in threats:
            sector = threat['asset_sector']
            if sector not in sector_threats:
                sector_threats[sector] = []
            sector_threats[sector].append(threat['overall_score'])

        sector_avg = {sector: np.mean(scores) for sector, scores in sector_threats.items()}

        # mpa visulisastion
        sectors = list(sector_avg.keys())
        sector_scores = list(sector_avg.values())

        colors = []
        for score in sector_scores:
            if score >= 8:
                color = 'red'
            elif score >= 6:
                color = 'orange'
            elif score >= 4:
                color = 'yellow'
            elif score >= 2:
                color = 'lightgreen'
            else:
                color = 'green'
            colors.append(color)

        bars = ax3.bar(sectors, sector_scores, color=colors, alpha=0.8)
        ax3.set_ylabel('Average Threat Score')
        ax3.set_title('Threat by Lahore Sector')
        ax3.set_ylim(0, 10)
        plt.xticks(rotation=45, ha='right')

        for bar, score in zip(bars, sector_scores):
            ax3.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.2,
                     f'{score:.1f}', ha='center', va='bottom', fontsize=9)

        plt.suptitle('Lahore Urban Defense - Threat Assessment', fontsize=14, fontweight='bold', y=1.05)
        plt.tight_layout()
        plt.show()

    def export_for_3d(self, filename='lahore_3d_data.json'):
        assets_3d = []
        for asset in self.defended_assets:
            assets_3d.append({
                'id': asset.asset_id,
                'name': asset.asset_name,
                'type': asset.asset_type,
                'position': asset.position.tolist(),
                'priority': asset.priority,
                'sector': asset.sector
            })

        threats_3d = []
        recent_threats = self.threat_history[-50:] if self.threat_history else []
        for threat in recent_threats:
            threats_3d.append({
                'target_id': threat['target_id'],
                'asset_id': threat['asset_id'],
                'threat_score': threat['overall_score'],
                'threat_level': self.get_threat_level(threat['overall_score']),
                'target_position': threat['target_position'],
                'urban_context': threat['urban_context']
            })

        export_data = {
            'defended_assets': assets_3d,
            'threat_data': threats_3d,
            'summary': {
                'total_threats': len(self.threat_history),
                'critical_count': len([h for h in self.threat_history
                                       if self.get_threat_level(h['overall_score']) == 'CRITICAL'])
            }
        }

        with open(filename, 'w') as f:
            json.dump(export_data, f, indent=2)

        print(f"Exported to {filename}")
        return export_data


def create_lahore_defended_assets():
    return [
        DefendedAsset(np.array([74.3587, 31.5657, 0]), "gov_house", "Governor House", 0.95, 300, "government",
                      sector="Central_Lahore"),
        DefendedAsset(np.array([74.3093, 31.5883, 0]), "lahore_fort", "Lahore Fort", 0.90, 250, "cultural",
                      sector="Walled_City"),
        DefendedAsset(np.array([74.3212, 31.5932, 0]), "badshahi_mosque", "Badshahi Mosque", 0.88, 200, "religious",
                      sector="Walled_City"),
        DefendedAsset(np.array([74.3639, 31.5185, 0]), "airport", "Allama Iqbal Airport", 0.85, 500, "transportation",
                      sector="Other_Sector"),
        DefendedAsset(np.array([74.2878, 31.4550, 0]), "power_plant", "Lahore Power Plant", 0.92, 400, "infrastructure",
                      sector="Other_Sector"),
        DefendedAsset(np.array([74.3484, 31.4887, 0]), "liberty_market", "Liberty Market", 0.70, 150, "commercial",
                      sector="Gulberg"),
        DefendedAsset(np.array([74.2823, 31.4712, 0]), "university", "University of Lahore", 0.75, 300, "education",
                      sector="Other_Sector"),
        DefendedAsset(np.array([74.3421, 31.5023, 0]), "mayo_hospital", "Mayo Hospital", 0.85, 250, "healthcare",
                      sector="Central_Lahore"),
        DefendedAsset(np.array([74.3312, 31.4321, 0]), "water_plant", "Lahore Water Plant", 0.88, 350, "utilities",
                      sector="Other_Sector"),
        DefendedAsset(np.array([74.3521, 31.5123, 0]), "broadcast_tower", "PEMRA Tower", 0.80, 100, "communications",
                      sector="Gulberg")
    ]


def create_lahore_threat_scenarios():
    return [
        TargetState(np.array([74.3100, 31.5850, 100]), np.array([0.0005, 0.0010, 0]), 0.0, "Walled_City_Drone", "drone",
                    "Walled_City"),
        TargetState(np.array([74.3500, 31.5100, 200]), np.array([0.0010, 0.0005, 0]), 0.1, "Airport_Approach_UAV",
                    "uav", "Other_Sector"),
        TargetState(np.array([74.3300, 31.5400, 150]), np.array([0.0008, 0.0000, 0]), 0.2, "Mall_Road_Threat", "drone",
                    "Central_Lahore"),
        TargetState(np.array([74.3700, 31.5400, 80]), np.array([-0.0003, 0.0008, 0]), 0.3, "Cantt_Surveillance",
                    "drone", "Cantonment"),
        TargetState(np.array([74.3450, 31.4900, 120]), np.array([0.0002, 0.0003, 0]), 0.4, "Gulberg_Threat", "drone",
                    "Gulberg")
    ]


def main():

    lahore_assets = create_lahore_defended_assets()
    calculator = ThreatScoreCalculator(lahore_assets)
    threats = create_lahore_threat_scenarios()
    all_threats = calculator.assess_multiple_threats(threats)

    print("Top threats:")
    for i, threat in enumerate(all_threats[:3], 1):
        level = calculator.get_threat_level(threat['overall_score'])
        print(f"{i}. {threat['target_id']} → {threat['asset_name']}")
        print(f"   Score: {threat['overall_score']:.1f}/10 ({level})")
        print(f"   Sector: {threat['asset_sector']}")
        print(f"   Urban Context: {threat['urban_context']['name']}")

    calculator.visualize_threat_assessment(threats)
    export_data = calculator.export_for_3d()
    print(f"• Assets Protected: {len(lahore_assets)}")
    print(f"• Threats Assessed: {len(threats)}")
    print(f"• Total Records: {len(calculator.threat_history)}")
    print(f"• 3D Data Exported: {len(export_data['threat_data'])} threats")


if __name__ == "__main__":
    main()