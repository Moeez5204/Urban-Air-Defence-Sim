import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import json
from dataclasses import dataclass, field
from typing import List, Dict, Tuple
import scipy.linalg


@dataclass
class IMMModel:
    name: str
    probability: float
    process_noise: float
    color: str
    # Model-specific parameters
    max_acceleration: float = 0.0
    prefers_canyons: bool = False
    avoids_obstacles: bool = False


@dataclass
class TrackingSnapshot:
    """One moment in time for the IMM filter"""
    timestamp: float
    true_position: List[float]
    estimated_position: List[float]
    model_probabilities: List[float]
    best_model: str
    position_uncertainty: List[float]
    urban_context: str


class IMMUrbanTracker:

    def __init__(self, urban_map_data=None):
        self.urban_map = urban_map_data or {}

        # different motion models for urban environments
        self.motion_models = [
            IMMModel(
                name="Canyon_Follower",
                probability=0.3,
                process_noise=1.0,
                color='red',
                max_acceleration=2.0,
                prefers_canyons=True,
                avoids_obstacles=True
            ),
            IMMModel(
                name="Open_Area_Flyer",
                probability=0.4,
                process_noise=2.0,
                color='blue',
                max_acceleration=3.0,
                prefers_canyons=False,
                avoids_obstacles=False
            ),
            IMMModel(
                name="Obstacle_Dodger",
                probability=0.3,
                process_noise=3.0,
                color='green',
                max_acceleration=5.0,
                prefers_canyons=False,
                avoids_obstacles=True
            )
        ]

        # Tracking state
        self.current_state = np.zeros(4)  # [x, y, vx, vy]
        self.state_covariance = np.eye(4) * 10

        #history
        self.tracking_history: List[TrackingSnapshot] = []
        self.time_step = 0.1

        print(f"Models: {[model.name for model in self.motion_models]}")

    def load_previous_tracking_data(self, filename='urban_tracking_data.json'):

        try:
            with open(filename, 'r') as f:
                data = json.load(f)

            #extract posision history
            positions = []
            urban_contexts = []
            velocities = []

            for i, step in enumerate(data['complete_tracking_history']):
                positions.append(step['estimated_position'][:2])  # Use only x,y
                urban_contexts.append(step['urban_terrain_type'])

                # Estimate velocity from position changes
                if i > 0:
                    prev_pos = data['complete_tracking_history'][i - 1]['estimated_position'][:2]
                    curr_pos = step['estimated_position'][:2]
                    velocity = [(curr_pos[0] - prev_pos[0]) / self.time_step,
                                (curr_pos[1] - prev_pos[1]) / self.time_step]
                else:
                    velocity = [0, 0]
                velocities.append(velocity)

            self.previous_positions = np.array(positions)
            self.previous_velocities = np.array(velocities)
            self.previous_contexts = urban_contexts
            self.canyon_preference = data['summary_info']['final_canyon_preference']

            print(f"Loaded {len(positions)} previous tracking steps")
            print(f"Target canyon preference: {self.canyon_preference:.3f}")

            return True
        except FileNotFoundError:
            return " file not found"

    def analyze_urban_context(self, position: np.ndarray, timestamp: float) -> str:

        if not hasattr(self, 'previous_contexts'): #determine what area it is in
            return "unknown"

        #get context from previous data using a timestamp
        time_idx = int(timestamp / self.time_step)
        if time_idx < len(self.previous_contexts):
            return self.previous_contexts[time_idx]

        return "unknown"

    def predict_next_state(self, model: IMMModel, urban_context: str) -> Tuple[np.ndarray, np.ndarray]:
        """Predict next state using a specific motion model"""

        # transistion matrix for constant velocity model
        F = np.array([
            [1, 0, self.time_step, 0],
            [0, 1, 0, self.time_step],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])

        predicted_state = F @ self.current_state

        if model.prefers_canyons and urban_context == "canyon":
            process_noise = model.process_noise * 0.5
        elif model.avoids_obstacles and urban_context == "obstacle":
            process_noise = model.process_noise * 1.5
        else:
            process_noise = model.process_noise

        #noise matrix
        Q = np.eye(4) * process_noise

        #Predict the covariance
        predicted_covariance = F @ self.state_covariance @ F.T + Q

        return predicted_state, predicted_covariance

    def update_model_probabilities(self, measurement: np.ndarray, urban_context: str):
        #update the model based off new data

        likelihoods = []

        for i, model in enumerate(self.motion_models):
            # Predict
            predicted_state, predicted_cov = self.predict_next_state(model, urban_context)
            predicted_pos = predicted_state[:2]  # Extract position only

            #Calculate
            error = measurement - predicted_pos
            distance_error = np.linalg.norm(error)

            #model  likelihood
            if (model.prefers_canyons and urban_context == "canyon") or \
                    (model.avoids_obstacles and urban_context == "obstacle") or \
                    (not model.prefers_canyons and urban_context == "open"):
                context_bonus = 1.2
            else:
                context_bonus = 0.8

            likelihood = np.exp(-distance_error / 10) * context_bonus
            likelihoods.append(likelihood)

        # Normalize
        total_likelihood = sum(likelihoods)
        if total_likelihood > 0:
            likelihoods = [l / total_likelihood for l in likelihoods]

        # update model probabilities
        for i, model in enumerate(self.motion_models):
            model.probability = 0.7 * model.probability + 0.3 * likelihoods[i]

        # Renormalize
        total_prob = sum(model.probability for model in self.motion_models)
        for model in self.motion_models:
            model.probability /= total_prob

    def update_with_measurement(self, measurement: np.ndarray, timestamp: float):

        urban_context = self.analyze_urban_context(measurement, timestamp)

        # Update model probabilities from new measurmnets
        self.update_model_probabilities(measurement, urban_context)

        # get weighted prediction
        mixed_state = np.zeros(4)
        mixed_covariance = np.zeros((4, 4))

        for model in self.motion_models:
            pred_state, pred_cov = self.predict_next_state(model, urban_context)
            mixed_state += model.probability * pred_state
            mixed_covariance += model.probability * pred_cov

        innovation = measurement - mixed_state[:2]
        kalman_gain = 0.8  # Simplified gain

        self.current_state[:2] = mixed_state[:2] + kalman_gain * innovation # update posistion

        #update velocity
        if len(self.tracking_history) > 0:

            prev_position = np.array(self.tracking_history[-1].estimated_position)
            self.current_state[2:] = (measurement - prev_position) / self.time_step
        else:
            time_idx = int(timestamp / self.time_step)
            if hasattr(self, 'previous_velocities') and time_idx < len(self.previous_velocities):
                self.current_state[2:] = self.previous_velocities[time_idx]

        self.state_covariance = mixed_covariance * 0.9
        best_model_idx = np.argmax([model.probability for model in self.motion_models])
        best_model = self.motion_models[best_model_idx].name

        snapshot = TrackingSnapshot(
            timestamp=timestamp,
            true_position=measurement.tolist(),
            estimated_position=self.current_state[:2].tolist(),
            model_probabilities=[model.probability for model in self.motion_models],
            best_model=best_model,
            position_uncertainty=[np.sqrt(self.state_covariance[0, 0]),
                                  np.sqrt(self.state_covariance[1, 1])],
            urban_context=urban_context
        )

        self.tracking_history.append(snapshot)

        return snapshot

    def run_imm_tracking(self):
        if not hasattr(self, 'previous_positions'):
            self.load_previous_tracking_data()

        #initialize
        self.current_state[:2] = self.previous_positions[0]
        if hasattr(self, 'previous_velocities'):
            self.current_state[2:] = self.previous_velocities[0]

        #Create base snapshot
        initial_snapshot = TrackingSnapshot(
            timestamp=0.0,
            true_position=self.previous_positions[0].tolist(),
            estimated_position=self.current_state[:2].tolist(),
            model_probabilities=[model.probability for model in self.motion_models],
            best_model=self.motion_models[0].name,
            position_uncertainty=[np.sqrt(self.state_covariance[0, 0]),
                                  np.sqrt(self.state_covariance[1, 1])],
            urban_context=self.analyze_urban_context(self.previous_positions[0], 0.0)
        )
        self.tracking_history.append(initial_snapshot)

        print("Tracking target through urban environment...")
        print("Time  Position  Best Model  Urban Context")
        print(
            f"{0:4d}  ({initial_snapshot.estimated_position[0]:6.1f}, {initial_snapshot.estimated_position[1]:6.1f})  "
            f"{initial_snapshot.best_model:15}  {initial_snapshot.urban_context}")

        # Process remaining measurements
        for t in range(1, min(10, len(self.previous_positions))):  # Limit to 10 steps for demo
            measurement = self.previous_positions[t]
            snapshot = self.update_with_measurement(measurement, t * self.time_step)

            print(f"{t:4d}  ({snapshot.estimated_position[0]:6.1f}, {snapshot.estimated_position[1]:6.1f})  "
                  f"{snapshot.best_model:15}  {snapshot.urban_context}")

    def visualize_imm_performance(self):
        if len(self.tracking_history) < 2:
            print("Not enough tracking data for visualization")
            return None

        fig = plt.figure(figsize=(16, 12))

        #Trajectory and Model Usage
        ax1 = plt.subplot(2, 3, 1)

        # plot trajectory
        positions = np.array([snapshot.estimated_position for snapshot in self.tracking_history])
        models = [snapshot.best_model for snapshot in self.tracking_history]

        model_colors = {'Canyon_Follower': 'red', 'Open_Area_Flyer': 'blue', 'Obstacle_Dodger': 'green'}

        for model_name, color in model_colors.items():
            model_indices = [i for i, m in enumerate(models) if m == model_name]
            if model_indices:
                ax1.scatter(positions[model_indices, 0], positions[model_indices, 1],
                            c=color, label=model_name, s=50, alpha=0.7)

        if hasattr(self, 'previous_positions'):
            true_positions = self.previous_positions[:len(self.tracking_history)]
            ax1.plot(true_positions[:, 0], true_positions[:, 1], 'k--', alpha=0.5, label='True Path')

        ax1.set_xlabel('X Position (m)')
        ax1.set_ylabel('Y Position (m)')
        ax1.set_title('Target Trajectory\n(Color = Best Motion Model)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        #Model Probability Evolution
        ax2 = plt.subplot(2, 3, 2)

        times = [snapshot.timestamp for snapshot in self.tracking_history]
        canyon_probs = [snapshot.model_probabilities[0] for snapshot in self.tracking_history]
        open_probs = [snapshot.model_probabilities[1] for snapshot in self.tracking_history]
        obstacle_probs = [snapshot.model_probabilities[2] for snapshot in self.tracking_history]

        ax2.plot(times, canyon_probs, 'r-', linewidth=2, label='Canyon Follower')
        ax2.plot(times, open_probs, 'b-', linewidth=2, label='Open Area Flyer')
        ax2.plot(times, obstacle_probs, 'g-', linewidth=2, label='Obstacle Dodger')

        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Model Probability')
        ax2.set_title('Model Probability Evolution')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, 1)

        #urban Context Analysis
        ax3 = plt.subplot(2, 3, 3)

        contexts = [snapshot.urban_context for snapshot in self.tracking_history]
        context_counts = {context: contexts.count(context) for context in set(contexts)}

        colors = {'canyon': 'red', 'open': 'blue', 'obstacle': 'green', 'mixed': 'orange', 'unknown': 'gray'}
        bar_colors = [colors.get(context, 'gray') for context in context_counts.keys()]

        bars = ax3.bar(context_counts.keys(), context_counts.values(), color=bar_colors, alpha=0.7)

        for bar, count in zip(bars, context_counts.values()):
            ax3.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                     f'{count}', ha='center', va='bottom', fontweight='bold')

        ax3.set_ylabel('Time Steps')
        ax3.set_title('Urban Context Distribution')
        ax3.grid(True, alpha=0.3)

        #Position Uncertainty
        ax4 = plt.subplot(2, 3, 4)

        x_uncertainty = [snapshot.position_uncertainty[0] for snapshot in self.tracking_history]
        y_uncertainty = [snapshot.position_uncertainty[1] for snapshot in self.tracking_history]

        ax4.plot(times, x_uncertainty, 'b-', linewidth=2, label='X Uncertainty', alpha=0.7)
        ax4.plot(times, y_uncertainty, 'r-', linewidth=2, label='Y Uncertainty', alpha=0.7)
        ax4.plot(times, np.sqrt(np.array(x_uncertainty) ** 2 + np.array(y_uncertainty) ** 2),
                 'k-', linewidth=2, label='Total Uncertainty')

        ax4.set_xlabel('Time (s)')
        ax4.set_ylabel('Uncertainty (m)')
        ax4.set_title('Position Uncertainty Over Time')
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        #Model Performance by Urban Context
        ax5 = plt.subplot(2, 3, 5)

        #calculate best model in context
        context_model_performance = {}
        for context in set(contexts):
            context_snapshots = [s for s in self.tracking_history if s.urban_context == context]
            if context_snapshots:
                model_wins = {}
                for model_name in model_colors.keys():
                    wins = sum(1 for s in context_snapshots if s.best_model == model_name)
                    model_wins[model_name] = wins / len(context_snapshots)
                context_model_performance[context] = model_wins

        if context_model_performance:
            contexts_list = list(context_model_performance.keys())
            bottom = np.zeros(len(contexts_list))

            for model_name in model_colors.keys():
                values = [context_model_performance[context].get(model_name, 0) for context in contexts_list]
                ax5.bar(contexts_list, values, bottom=bottom, label=model_name,
                        color=model_colors[model_name], alpha=0.7)
                bottom += values

            ax5.set_ylabel('Proportion of Time Steps')
            ax5.set_title('Best Model by Urban Context')
            ax5.legend()
            ax5.grid(True, alpha=0.3)

        ax6 = plt.subplot(2, 3, 6)
        ax6.axis('off')

        total_steps = len(self.tracking_history)
        final_probs = [f"{model.name}: {model.probability:.3f}" for model in self.motion_models]
        avg_uncertainty = np.mean([np.sqrt(u[0] ** 2 + u[1] ** 2) for u in
                                   [snapshot.position_uncertainty for snapshot in self.tracking_history]])

        context_breakdown = "\n".join([f"{context}: {count} steps ({count / total_steps * 100:.1f}%)"
                                       for context, count in context_counts.items()])

        summary_text = (
            "IMM FILTER SUMMARY\n\n"
            f"Total Tracking Steps: {total_steps}\n"
            f"Final Model Probabilities:\n"
            f"  {final_probs[0]}\n"
            f"  {final_probs[1]}\n"
            f"  {final_probs[2]}\n"
            f"Average Position Uncertainty: {avg_uncertainty:.2f} m\n\n"
            f"Urban Context Breakdown:\n{context_breakdown}\n\n"
            f"Target Canyon Preference: {self.canyon_preference:.3f}"
        )

        ax6.text(0.1, 0.95, summary_text, transform=ax6.transAxes, fontsize=10,
                 verticalalignment='top', fontfamily='monospace',
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))

        plt.suptitle('Interactive Multiple Model (IMM) Filter: Urban Target Tracking Performance',
                     fontsize=16, fontweight='bold', y=0.95)

        plt.tight_layout()
        plt.show()

        return fig

    def export_imm_results(self, filename='imm_tracking_results_3.2.2.json'):

        export_data = {
            'tracking_summary': {
                'total_steps': len(self.tracking_history),
                'final_model_probabilities': {model.name: model.probability for model in self.motion_models},
                'target_canyon_preference': self.canyon_preference,
                'average_uncertainty': np.mean([np.sqrt(u[0] ** 2 + u[1] ** 2) for u in
                                                [snapshot.position_uncertainty for snapshot in self.tracking_history]])
            },
            'model_descriptions': {
                model.name: {
                    'prefers_canyons': model.prefers_canyons,
                    'avoids_obstacles': model.avoids_obstacles,
                    'max_acceleration': model.max_acceleration,
                    'process_noise': model.process_noise
                } for model in self.motion_models
            },
            'performance_by_context': {},
            'sample_tracking_data': [
                {
                    'timestamp': snapshot.timestamp,
                    'estimated_position': snapshot.estimated_position,
                    'best_model': snapshot.best_model,
                    'urban_context': snapshot.urban_context,
                    'model_probabilities': snapshot.model_probabilities,
                    'position_uncertainty': snapshot.position_uncertainty
                } for snapshot in self.tracking_history
            ]
        }

        #context-based performance
        contexts = set(snapshot.urban_context for snapshot in self.tracking_history)
        for context in contexts:
            context_snapshots = [s for s in self.tracking_history if s.urban_context == context]
            model_performance = {}
            for model in self.motion_models:
                wins = sum(1 for s in context_snapshots if s.best_model == model.name)
                model_performance[model.name] = wins / len(context_snapshots) if context_snapshots else 0
            export_data['performance_by_context'][context] = model_performance

        with open(filename, 'w') as f:
            json.dump(export_data, f, indent=2)

        print(f"IMM tracking results exported to {filename}")
        return export_data


def run_phase_3_2_2():

    print("Interactive Multiple Model Filter")

    imm_tracker = IMMUrbanTracker()
    imm_tracker.load_previous_tracking_data()
    imm_tracker.run_imm_tracking()
    imm_tracker.visualize_imm_performance()
    results = imm_tracker.export_imm_results()

    print(f"• Uses {len(imm_tracker.motion_models)} different motion models")
    for model in imm_tracker.motion_models:
        print(f"  - {model.name}: {model.probability:.3f}")
    print(f"• Average position uncertainty: {results['tracking_summary']['average_uncertainty']:.2f} m")

    return imm_tracker


if __name__ == "__main__":
    run_phase_3_2_2()