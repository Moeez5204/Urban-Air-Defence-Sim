<div align="center">

# 🛡️ Urban Air Defense Simulation with Topological AI & Predictive Engagement

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyGame](https://img.shields.io/badge/PyGame-3D%20Engine-orange.svg)](https://www.pygame.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Status](https://img.shields.io/badge/Status-Active-success.svg)]()

*A cutting-edge simulation framework turning complex 3D urban terrain into a predictive asset using computational topology, persistent homology, and advanced machine learning.*

[Key Features](#-key-features) • [Architecture](#%EF%B8%8F-system-architecture--pipeline) • [Installation](#-installation) • [Usage](#-usage) • [Performance](#-results--performance) • [Generalizability](#-cross-domain-generalizability)

</div>

---

## 🧭 Overview

Modern air defense systems face the **"urban air defense trilemma"**:
1. **Sensor & Signal Degradation:** Urban clutter, multipath interference, and radar shadow zones caused by dense buildings.
2. **Temporal Compression:** High-speed aerial threats operating within seconds of decision windows.
3. **Strategic Complexity:** Non-linear movement profiles dictated by complex city canyons.

**Urban-Air-Defence-Sim** addresses these challenges by modeling a complete urban air defense simulation set in **Lahore, Pakistan**. By leveraging **computational topology**, **persistent homology**, and **Interactive Multiple Model (IMM) LSTMs**, the system maps structural urban bottlenecks, predicts trajectory shifts through complex street canyons, and dynamically allocates defensive resources.

---

## ⚙️ System Architecture & Pipeline

The framework is structured into three core algorithmic phases running sequentially:

### Phase 1: Topological City Reconstruction & Feature Extraction
* **3D Synthetic Synthesis:** Extracts 2D building footprints via OSMNX and constructs a 3D synthetic model of Lahore using heuristic height zoning (low-rise residential, mid-rise commercial, high-rise CBD).
* **Persistent Homology & Alpha Complexes:** Samples dense point clouds and computes persistence pairs across multi-scale filtrations ($\epsilon$-birth and $\epsilon$-death).
* **Topological Feature Classification:** Filters out background noise to isolate primary/secondary canyons ($H_1$ loops), major/minor obstacles ($H_0$ components), and strategic voids ($H_2$ cavities).

### Phase 2: Topology-Aware Predictive Tracking
* **IMM Filtering:** Integrates an augmented Kalman filter and an Interactive Multiple Model filter running three parallel motion models: *Canyon Follower*, *Open Area Flyer*, and *Obstacle Dodger*.
* **Behavioral Parameterization:** Expands the tracking state vector to incorporate a dynamic **Canyon Affinity Parameter ($C_A$)**.
* **Multi-Scale Feature Fusion (MSFF) LSTM:** Combines kinematic tracking data with topological context features to achieve robust trajectory forecasting.

### Phase 3: Threat Assessment & Adaptive Sector Defense Allocation (ASDA)
* **Weighted Threat Scoring:** Evaluates threats using a composite function combining distance (30%), approach angle (15%), speed (20%), and topological urban context (35%) queried via a KD-tree.
* **Dynamic Sector Allocation:** Balances a 60/40 baseline sector priority split (e.g., Central Lahore: 0.9, Walled City: 0.8, Cantonment: 0.85) with real-time threat scores.
* **$\epsilon$-Greedy Resource Management:** Allocates a 10% exploratory resource budget ($\epsilon = 0.1$) to lower-priority sectors to prevent blind spots and feedback loops.

---
<img width="357" height="239" alt="Screenshot 2026-08-31 at 2 52 28 PM" src="https://github.com/user-attachments/assets/cd896bc4-2f55-46c8-9625-dcc6b6e242cc" />

<img width="609" height="471" alt="Screenshot 2026-08-31 at 2 52 15 PM" src="https://github.com/user-attachments/assets/facb9599-43ee-41a7-964a-2034d4f38024" />



## ✨ Key Features

* 🏙️ **Topological City Modeling:** Converts raw GIS data into mathematically rigorous alpha complexes.
* 🧠 **Predictive Trajectory AI:** Leverages Topological Transition Forecasting (TTF) for environment-aware uncertainty modeling.
* 🎯 **Dynamic Defense Allocation:** Intelligent resource distribution balancing high-value assets with exploratory coverage.
* 🎮 **Interactive 3D Visualizer:** Built on PyGame 3D with pulsing concentric radar spheres, live HUD telemetry reports, and post-mission Matplotlib trajectory analytics.
* 🌐 **Cross-Domain Generalizability:** Adaptable to search-and-rescue, radiation-dense nuclear zones (e.g., Chernobyl), and deep-sea obstacle mapping.


<img width="547" height="239" alt="Screenshot 2026-08-31 at 2 56 42 PM" src="https://github.com/user-attachments/assets/741aa74d-2c93-4206-9729-82f835b72a1a" />

---

## 🛠️ Tech Stack

* **Language:** Python 3.10+
* **Core Libraries:** NumPy, SciPy, Pandas, Scikit-learn
* **Topology & GIS:** Gudhi / Ripser (Persistent Homology), OSMNX, GeoPandas
* **Machine Learning:** PyTorch / TensorFlow (LSTM Networks)
* **Visualization:** PyGame, Matplotlib, OpenGL bindings

---

## 📦 Installation

Ensure you have **Python 3.10 or higher** installed on your system. Follow these steps to set up your local development environment:

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/Urban-Air-Defence-Sim.git
   cd Urban-Air-Defence-Sim
   
