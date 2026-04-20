import osmnx as ox
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, box
import random
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import json
from shapely.geometry import shape


ox.settings.log_console = True
ox.settings.use_cache = True
ox.settings.timeout = 300


def download_lahore_data():
    # get the real data for builings in Lahore paksitan

    north, south, east, west = 31.6000, 31.4800, 74.4500, 74.2800 #lahore boundaries

    try:
        buildings_gdf = ox.geometries_from_bbox(north, south, east, west, tags={'building': True})
        print(f"Downloaded {len(buildings_gdf)} buildings")
        return buildings_gdf, (north, south, east, west)

    except Exception as e:
        print(f"Error downloading data: {e}")
        return None, (north, south, east, west)


def process_lahore_buildings(buildings_gdf, bbox):


    north, south, east, west = bbox
    building_footprints = []

    # get major Lahore zones for height assignment
    def get_lahore_zone(centroid):
        x, y = centroid.x, centroid.y

        zones = {
            'Old City': ((74.30, 31.55), (74.35, 31.59)),
            'Gulberg': ((74.33, 31.50), (74.36, 31.53)),
            'Defence': ((74.36, 31.48), (74.39, 31.51)),
            'Cantt': ((74.34, 31.53), (74.37, 31.56)),
            'DHA': ((74.38, 31.46), (74.42, 31.50)),
            'Model Town': ((74.32, 31.49), (74.35, 31.52)),
        }

        for zone_name, ((x1, y1), (x2, y2)) in zones.items():
            if x1 <= x <= x2 and y1 <= y <= y2:
                return zone_name
        return 'Other'

    def assign_height_from_zone(zone, area):
        zone_heights = {
            'Old City': (1, 3, 3.0),  # low-rise
            'Gulberg': (5, 15, 3.5),  # high-rise
            'Defence': (4, 12, 3.4),  # mixed
            'Cantt': (2, 8, 3.2),  # military
            'DHA': (2, 6, 3.2),  # hifi socities (my family)
            'Model Town': (2, 5, 3.2),  # residential
            'Other': (1, 4, 3.0)
        }

        min_stories, max_stories, meter_per_story = zone_heights.get(zone, (1, 4, 3.0))
        area_factor = min(1.0, area / 1000.0)
        story_bonus = int(area_factor * (max_stories - min_stories))
        stories = random.randint(min_stories, min_stories + story_bonus)
        height = stories * meter_per_story

        return height, stories

    processed_count = 0
    for idx, building in buildings_gdf.iterrows():

            if building.geometry.type == 'Polygon' and building.geometry.area > 10:
                centroid = building.geometry.centroid
                zone = get_lahore_zone(centroid)
                height, stories = assign_height_from_zone(zone, building.geometry.area)

                building_data = {
                    'footprint': building.geometry,
                    'height': height,
                    'stories': stories,
                    'center': (centroid.x, centroid.y),
                    'zone': zone,
                    'area': building.geometry.area
                }
                building_footprints.append(building_data)
                processed_count += 1



    print(f"processed {len(building_footprints)} buildings")
    return building_footprints


def export_building_data(building_footprints, filename='building_data_3.1.1.json'):
    print(f"Exporting building data to {filename}...")

    export_data = []
    for building in building_footprints:
        building_export = {
            'height': building['height'],
            'stories': building['stories'],
            'center': building['center'],
            'zone': building['zone'],
            'area': building['area'],
            'footprint': building['footprint'].__geo_interface__  # convert shapely to GeoJSON
        }
        export_data.append(building_export)

    with open(filename, 'w') as f:
        json.dump(export_data, f, indent=2)

    print(f"Exported {len(export_data)} buildings to {filename}")
    return filename


def load_building_data(filename='building_data_3.1.1.json'):
    #load building data
    print(f"load building data from {filename}...")

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
            'footprint': shape(item['footprint'])  # Convert back to shapely
        }
        building_footprints.append(building)

    print(f"Loaded {len(building_footprints)} buildings from {filename}")
    return building_footprints


def run_phase_3_1_1():


    #download real lahore data
    buildings_gdf, bbox = download_lahore_data()

    if buildings_gdf is not None:
        # Process real data
        building_footprints = process_lahore_buildings(buildings_gdf, bbox)

    #analyse
    analyze_building_data(building_footprints)

    # Export the data for 3.1.2
    export_filename = export_building_data(building_footprints)

    print(f"Phase 3.1.1 Complete!")
    print(f"Data exported to {export_filename} for Phase 3.1.2")

    return building_footprints, bbox


# Visualization and analysis functions (keep your existing ones)
def analyze_building_data(building_footprints):
    heights = [b['height'] for b in building_footprints]
    stories = [b['stories'] for b in building_footprints]
    zones = [b['zone'] for b in building_footprints]
    areas = [b['area'] for b in building_footprints]

    print(f"Total buildings: {len(building_footprints)}")
    print(f"height:")
    print(f"  Min: {min(heights):.1f}m, Max: {max(heights):.1f}m, Avg: {np.mean(heights):.1f}m")
    print(f"Area:")
    print(f"  Min: {min(areas):.1f}m², Max: {max(areas):.1f}m², Avg: {np.mean(areas):.1f}m²")

    # Zone distribution
    print(f"distribution:")
    zone_counts = {}
    for zone in set(zones):
        zone_buildings = [b for b in building_footprints if b['zone'] == zone]
        zone_heights = [b['height'] for b in zone_buildings]
        zone_counts[zone] = len(zone_buildings)
        print(f"  {zone}: {len(zone_buildings)} buildings, Avg height: {np.mean(zone_heights):.1f}m")

    #categories
    low_rise = len([h for h in heights if h <= 12.8])
    mid_rise = len([h for h in heights if 12.8 < h <= 25.6])
    high_rise = len([h for h in heights if h > 25.6])

    print(f"Building categories:")
    print(f"Low-rise (≤4 stories): {low_rise} buildings ({low_rise / len(heights) * 100:.1f}%)")
    print(f"Mid-rise (4-8 stories): {mid_rise} buildings ({mid_rise / len(heights) * 100:.1f}%)")
    print(f"High-rise (>8 stories): {high_rise} buildings ({high_rise / len(heights) * 100:.1f}%)")


if __name__ == "__main__":
    run_phase_3_1_1()