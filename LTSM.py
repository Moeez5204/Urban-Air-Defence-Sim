# LSTM_TopologyPredictor.py
import torch
import torch.nn as nn
import numpy as np
import json
import matplotlib.pyplot as plt
from dataclasses import dataclass
import torch.optim as optim
import warnings
import os



@dataclass
class TopologicalFeatures:
   #topological features
    feature_id: str
    normalized_persistence: float
    distance_to_centerline: float
    feature_type: str
    threat_level: float
    concealment_value: float


class SimpleTopologyContextMapper:
    def __init__(self):
        self.strategic_database = None

    def load_strategic_data(self, filename='strategic_features_3.1.4.json'):
        try:
            with open(filename, 'r') as f:
                data = json.load(f)
            self.strategic_database = data['strategic_features']
            print(f"Loaded {len(self.strategic_database['canyons'])} canyons, "
                  f"{len(self.strategic_database['obstacles'])} obstacles")
            return self.strategic_database
        except FileNotFoundError:
            print(f"Warning: {filename} not found")
            return None

    def get_current_features(self, target_position, timestamp):
        if not self.strategic_database:
            return self._get_default_features()

        target_2d = np.array(target_position[:2])
        best_features = None
        min_distance = float('inf')

        # Check canyons
        for canyon in self.strategic_database['canyons']:
            if canyon.get('centerline'):
                centerline = np.array(canyon['centerline'])[:, :2]
                distances = np.linalg.norm(centerline - target_2d, axis=1)
                dist = np.min(distances)

                if dist < min_distance and dist < 75.0:
                    min_distance = dist
                    best_features = TopologicalFeatures(
                        feature_id=canyon['id'],
                        normalized_persistence=min(1.0, canyon['persistence'] / 1000),
                        distance_to_centerline=dist,
                        feature_type='canyon',
                        threat_level=0.8 if canyon['threat_level'] == 'high' else 0.5,
                        concealment_value=canyon['concealment_value']
                    )

        #check obstacles if no canyon found
        if not best_features:
            for obstacle in self.strategic_database['obstacles']:
                obstacle_center = np.array([obstacle.get('birth', 0), obstacle.get('death', 0)])
                dist = np.linalg.norm(target_2d - obstacle_center)
                if dist < 100.0:
                    best_features = TopologicalFeatures(
                        feature_id=obstacle['id'],
                        normalized_persistence=min(1.0, obstacle['persistence'] / 500),
                        distance_to_centerline=dist,
                        feature_type='obstacle',
                        threat_level=0.9 if obstacle['threat_level'] == 'high' else 0.6,
                        concealment_value=obstacle['concealment_value']
                    )
                    break

        return best_features or self._get_default_features()

    def _get_default_features(self):
        return TopologicalFeatures(
            feature_id='open_area',
            normalized_persistence=0.1,
            distance_to_centerline=100.0,
            feature_type='void',
            threat_level=0.3,
            concealment_value=0.1
        )


class BetterTopologyAwareLSTM(nn.Module):

    def __init__(self, input_size=11, hidden_size=48, num_layers=2, output_size=3):
        super(BetterTopologyAwareLSTM, self).__init__()

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)

        # Better fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, output_size)
        )

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_out = lstm_out[:, -1, :]  # Use last time step
        prediction = self.fc(last_out)
        return prediction


class ImprovedLSTMPredictor:


    def __init__(self):
        self.context_mapper = SimpleTopologyContextMapper()
        self.model = None
        self.sequence_length = 5
        self.prediction_horizon = 1
        # Store normalization parameters for later use
        self.pos_mean = None
        self.pos_std = None
        self.feat_mean = None
        self.feat_std = None

    def load_all_data(self):
        print("Loading existing data from JSON files...")

        strategic_data = self.context_mapper.load_strategic_data()

        # Kalman filter results
        try:
            with open('urban_tracking_data.json', 'r') as f:
                kalman_data = json.load(f)
            print(f"✓ Loaded {len(kalman_data['complete_tracking_history'])} tracking steps")

            # Print some data stats for debugging
            positions = [step['estimated_position'] for step in kalman_data['complete_tracking_history']]
            positions = np.array(positions)
            print(f"Position range: X({positions[:, 0].min():.1f} to {positions[:, 0].max():.1f}), "
                  f"Y({positions[:, 1].min():.1f} to {positions[:, 1].max():.1f}), "
                  f"Z({positions[:, 2].min():.1f} to {positions[:, 2].max():.1f})")

        except FileNotFoundError:
            print("urban_tracking_data.json not found")
            return None, None

        return strategic_data, kalman_data

    def create_better_training_data(self, kalman_data, num_sequences=50):

        sequences = []
        targets = []
        tracking_history = kalman_data['complete_tracking_history']

        if len(tracking_history) < self.sequence_length + 1:
            print("Not enough real data, creating better synthetic sequences...")
            return self._create_better_synthetic_data(num_sequences)

        #collect data for normalisation
        all_positions = []
        all_velocities = []
        all_features = []

        for step in tracking_history:
            position = step['estimated_position']
            velocity = step['estimated_velocity']
            canyon_affinity = step['canyon_preference']

            topo_features = self.context_mapper.get_current_features(position, 0)

            features = position + velocity + [
                canyon_affinity,
                topo_features.normalized_persistence,
                topo_features.distance_to_centerline / 100.0,
                topo_features.threat_level,
                topo_features.concealment_value
            ]

            all_positions.append(position)
            all_velocities.append(velocity)
            all_features.append(features)

        all_positions = np.array(all_positions)
        all_velocities = np.array(all_velocities)
        all_features = np.array(all_features)

        #normalization parameters
        self.pos_std = all_positions.std(axis=0)
        self.pos_mean = all_positions.mean(axis=0)
        vel_mean, vel_std = all_velocities.mean(axis=0), all_velocities.std(axis=0)
        self.feat_mean = all_features.mean(axis=0)
        self.feat_std = all_features.std(axis=0)

        # avoid divide by 0
        self.pos_std = np.where(self.pos_std == 0, 1.0, self.pos_std)
        vel_std = np.where(vel_std == 0, 1.0, vel_std)
        self.feat_std = np.where(self.feat_std == 0, 1.0, self.feat_std)

        # Create sequences with normalization
        for i in range(len(tracking_history) - self.sequence_length):
            sequence = []

            for j in range(self.sequence_length):  #Normalize features

                features = all_features[i + j]
                normalized_features = (features - self.feat_mean) / self.feat_std
                sequence.append(normalized_features)

            # Normalize target
            target_position = all_positions[i + self.sequence_length]
            normalized_target = (target_position - self.pos_mean) / self.pos_std

            sequences.append(sequence)
            targets.append(normalized_target)

            if len(sequences) >= num_sequences:
                break

        # Add fake data if needed
        if len(sequences) < num_sequences:
            print(f"Adding {num_sequences - len(sequences)} synthetic sequences...")
            synth_sequences, synth_targets = self._create_better_synthetic_data(
                num_sequences - len(sequences), self.pos_mean, self.pos_std, self.feat_mean, self.feat_std)
            sequences.extend(synth_sequences)
            targets.extend(synth_targets)

        sequences = np.array(sequences)
        targets = np.array(targets)

        print(f"Created {len(sequences)} training sequences")
        print(f"Input shape: {sequences.shape}, Target shape: {targets.shape}")

        return sequences, targets, (self.pos_mean, self.pos_std, self.feat_mean, self.feat_std)


    def _create_better_synthetic_data(self, num_sequences, pos_mean=None, pos_std=None, feat_mean=None, feat_std=None):
        #Create realistic synthetic data with proper urban speeds
        targets = []

        sequences = []

        # use  normalization or create reasonable defaults
        if pos_mean is None:
            pos_mean = np.array([150, 25, 50])
            pos_std = np.array([100, 50, 20])
        if feat_mean is None:
            feat_mean = np.zeros(11)
            feat_std = np.ones(11)

        for seq_idx in range(num_sequences):
            sequence = []

            #realistic scenario with proper urban speeds
            scenario = seq_idx % 4
            if scenario == 0:  # Canyon following
                start_pos = np.array([50, 10, 45])
                velocity = np.array([12, 1, 0])
                topo_features = [0.8, 0.7, 0.1, 0.8, 0.9]
            elif scenario == 1:  #Open area
                start_pos = np.array([200, 50, 60])
                velocity = np.array([18, 4, 0.5])
                topo_features = [0.2, 0.1, 0.8, 0.3, 0.1]
            elif scenario == 2:  #obstacle avoidance
                start_pos = np.array([100, 30, 50])
                velocity = np.array([10, -2, -0.5])
                topo_features = [0.5, 0.5, 0.3, 0.7, 0.6]
            else:  # Urban transit
                start_pos = np.array([120, 40, 55])
                velocity = np.array([15, 3, 0.2])
                topo_features = [0.6, 0.4, 0.5, 0.6, 0.4]

            for step in range(self.sequence_length + 1):
                #random variations to velocity
                current_vel = velocity + np.random.normal(0, 0.3, 3)
                current_pos = start_pos + current_vel * step * 1.0  # 1 second steps

                if step < self.sequence_length:
                    features = current_pos.tolist() + current_vel.tolist() + topo_features
                    normalized_features = (np.array(features) - feat_mean) / feat_std
                    sequence.append(normalized_features)

                if step == self.sequence_length:
                    target_pos = start_pos + current_vel * (self.sequence_length + 1) * 1.0
                    normalized_target = (target_pos - pos_mean) / pos_std
                    target = normalized_target

            sequences.append(sequence)
            targets.append(target)

        return sequences, targets

    def train_better_model(self, sequences, targets, normalization_params, epochs=100):

        # split data
        split_idx = int(0.8 * len(sequences))
        train_seq, val_seq = sequences[:split_idx], sequences[split_idx:]
        train_tar, val_tar = targets[:split_idx], targets[split_idx:]

        train_seq_tensor = torch.FloatTensor(train_seq)
        train_tar_tensor = torch.FloatTensor(train_tar)
        val_seq_tensor = torch.FloatTensor(val_seq)
        val_tar_tensor = torch.FloatTensor(val_tar)

        #Initialise
        self.model = BetterTopologyAwareLSTM(
            input_size=11,
            output_size=3
        )

        optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        criterion = nn.MSELoss()
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)

        train_losses = []
        val_losses = []
        best_val_loss = float('inf')
        patience = 0

        for epoch in range(epochs):
            self.model.train()
            optimizer.zero_grad()

            train_pred = self.model(train_seq_tensor)
            train_loss = criterion(train_pred, train_tar_tensor)
            train_loss.backward()
            optimizer.step()

            self.model.eval()
            with torch.no_grad():
                val_pred = self.model(val_seq_tensor)
                val_loss = criterion(val_pred, val_tar_tensor)

            train_losses.append(train_loss.item())
            val_losses.append(val_loss.item())

            scheduler.step(val_loss)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience = 0
                torch.save(self.model.state_dict(), 'best_lstm_model.pth')
            else:
                patience += 1

            if epoch % 20 == 0:
                print(f'Epoch {epoch:3d}: Train Loss = {train_loss.item():.6f}, '
                      f'Val Loss = {val_loss.item():.6f}')

            if patience >= 15:
                print(f"Early stopping at epoch {epoch}")
                break

        #load best model
        self.model.load_state_dict(torch.load('best_lstm_model.pth'))

        print("LTSM on now")

        self._save_normalization_params(normalization_params)

        # save model
        self._save_model_info()

        # plot the training data
        plt.figure(figsize=(11, 5))
        plt.plot(train_losses, label='Training Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.title('LSTM Training Progress')
        plt.xlabel('Epoch')
        plt.ylabel('MSE Loss')
        plt.legend()
        plt.yscale('log')
        plt.show()

        print(f"Best validation loss: {best_val_loss:.6f}")
        return self.model, normalization_params

    def _save_normalization_params(self, normalization_params):
        pos_mean, pos_std, feat_mean, feat_std = normalization_params

        norm_data = {
            'pos_mean': pos_mean.tolist(),
            'pos_std': pos_std.tolist(),
            'feat_mean': feat_mean.tolist(),
            'feat_std': feat_std.tolist(),
            'sequence_length': self.sequence_length,
            'prediction_horizon': self.prediction_horizon,
            'timestamp': str(np.datetime64('now'))
        }

        with open('best_lstm_model_norm.json', 'w') as f:
            json.dump(norm_data, f, indent=2)

        print("Normalization parameters saved to 'best_lstm_model_norm.json'")

        #save file
        np.savez('lstm_normalization.npz',
                 pos_mean=pos_mean,
                 pos_std=pos_std,
                 feat_mean=feat_mean,
                 feat_std=feat_std)
        print(" Normalization parameters saved to 'lstm_normalization.npz'")

    def _save_model_info(self):
        model_info = {
            'model_type': 'BetterTopologyAwareLSTM',
            'input_size': 11,
            'hidden_size': 48,
            'num_layers': 2,
            'output_size': 3,
            'sequence_length': self.sequence_length,
            'prediction_horizon': self.prediction_horizon,
            'model_file': 'best_lstm_model.pth',
            'norm_file': 'best_lstm_model_norm.json'
        }

        with open('lstm_model_info.json', 'w') as f:
            json.dump(model_info, f, indent=2)

        print("Model info saved to 'lstm_model_info.json'")

    def load_saved_model(self):
        try:
            # Load model
            self.model = BetterTopologyAwareLSTM(input_size=11, output_size=3)
            self.model.load_state_dict(torch.load('best_lstm_model.pth', map_location='cpu'))
            self.model.eval()

            # Load normalization parameters
            if os.path.exists('best_lstm_model_norm.json'):
                with open('best_lstm_model_norm.json', 'r') as f:
                    norm_data = json.load(f)
                self.pos_mean = np.array(norm_data['pos_mean'])
                self.pos_std = np.array(norm_data['pos_std'])
                self.feat_mean = np.array(norm_data['feat_mean'])
                self.feat_std = np.array(norm_data['feat_std'])
                self.sequence_length = norm_data.get('sequence_length', 5)
                print("✓ Loaded trained LSTM model with normalization parameters")
                return True
            else:
                print("⚠ Model file found but normalization parameters missing")
                return False
        except Exception as e:
            print(f"⚠ Error loading saved model: {e}")
            return False

    def predict_next(self, current_position, current_velocity, canyon_affinity=0.5,
                     topo_features=None, use_history=False, history_positions=None):
        #predict next posistion
        if self.model is None:
            print("Model not loaded. Call load_saved_model() first.")
            return None

        if self.pos_mean is None or self.feat_mean is None:
            print("Normalization parameters not loaded.")
            return None

        try:
            # feature vector
            if topo_features is None:
                topo_features = [0.1, 1.0, 0.3, 0.1]

            features = list(current_position) + list(current_velocity) + [canyon_affinity] + topo_features
            features = np.array(features)
            normalized_features = (features - self.feat_mean) / self.feat_std
            sequence = [normalized_features] * self.sequence_length
            input_tensor = torch.FloatTensor([sequence])

            # predict
            with torch.no_grad():
                normalized_pred = self.model(input_tensor)

                #denormalize
                predicted_pos = normalized_pred.numpy().flatten() * self.pos_std + self.pos_mean

            return tuple(predicted_pos)

        except Exception as e:
            print(f"Prediction error: {e}")
            return None

    def demonstrate_better_predictions(self, kalman_data, normalization_params):
        print("Demonstrating Improved LSTM Predictions")

        tracking_history = kalman_data['complete_tracking_history']
        pos_mean, pos_std, feat_mean, feat_std = normalization_params

        if len(tracking_history) < self.sequence_length + 1:
            print("Not enough data")
            return
        real_sequence = []
        for i in range(self.sequence_length):
            step = tracking_history[i]
            position = step['estimated_position']
            velocity = step['estimated_velocity']
            canyon_affinity = step['canyon_preference']

            topo_features = self.context_mapper.get_current_features(position, i * 0.1)

            features = position + velocity + [
                canyon_affinity,
                topo_features.normalized_persistence,
                topo_features.distance_to_centerline / 100.0,
                topo_features.threat_level,
                topo_features.concealment_value
            ]

            normalized_features = (np.array(features) - feat_mean) / feat_std  #normalize

            real_sequence.append(normalized_features)

        # actual next posiston
        actual_next = tracking_history[self.sequence_length]['estimated_position']

        #make prediction
        self.model.eval()
        with torch.no_grad():
            test_input = torch.FloatTensor([real_sequence])
            normalized_prediction = self.model(test_input)

            #Denormalize prediction
            predicted_next = normalized_prediction.numpy().flatten() * pos_std + pos_mean

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))

        #Position comparison
        labels = ['X', 'Y', 'Z']
        x_pos = np.arange(3)
        width = 0.35

        ax1.bar(x_pos - width / 2, predicted_next, width, label='Predicted', alpha=0.7, color='blue')
        ax1.bar(x_pos + width / 2, actual_next, width, label='Actual', alpha=0.7, color='red')
        ax1.set_xlabel('Coordinate')
        ax1.set_ylabel('Position (m)')
        ax1.set_title('Predicted vs Actual Position')
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(labels)
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        #Error analysis
        errors = np.abs(np.array(predicted_next) - np.array(actual_next))
        total_error = np.linalg.norm(predicted_next - actual_next)

        ax2.bar(labels, errors, color='orange', alpha=0.7)
        ax2.set_ylabel('Absolute Error (m)')
        ax2.set_title(f'Prediction Errors\nTotal Error: {total_error:.2f}m')
        ax2.grid(True, alpha=0.3)

        #Trajectory plot
        recent_positions = [step['estimated_position'] for step in tracking_history[:self.sequence_length + 1]]
        recent_positions = np.array(recent_positions)

        ax3.plot(recent_positions[:, 0], recent_positions[:, 1], 'go-', label='Past Path', linewidth=2)
        ax3.plot(actual_next[0], actual_next[1], 'ro', markersize=8, label='Actual Next')
        ax3.plot(predicted_next[0], predicted_next[1], 'bo', markersize=8, label='Predicted Next')
        ax3.set_xlabel('X Position (m)')
        ax3.set_ylabel('Y Position (m)')
        ax3.set_title('2D Trajectory View')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        #Text summary
        ax4.axis('off')
        summary_text = f"""
Prediction Results:
-------------------
Predicted: [{predicted_next[0]:.1f}, {predicted_next[1]:.1f}, {predicted_next[2]:.1f}]
Actual:    [{actual_next[0]:.1f}, {actual_next[1]:.1f}, {actual_next[2]:.1f}]
Total Error: {total_error:.2f} m

Improvement: {(69.25 - total_error) / 69.25 * 100:.1f}% better than before
"""
        ax4.text(0.1, 0.9, summary_text, transform=ax4.transAxes, fontsize=10,
                 verticalalignment='top', fontfamily='monospace',
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))

        plt.tight_layout()
        plt.show()

        print(f"Predicted: {predicted_next}")
        print(f"Actual:    {actual_next}")
        print(f"Error: {total_error:.2f}m ({(69.25 - total_error) / 69.25 * 100:.1f}% improvement)")

        return predicted_next, actual_next

    def analyze_urban_behavior(self, normalization_params):
        print("\nAnalyzing Urban Behavior Patterns...")

        if self.model is None:
            print("Model not trained yet")
            return

        pos_mean, pos_std, feat_mean, feat_std = normalization_params

        # More diverse test scenarios
        scenarios = {
            'Canyon_Following': [50, 25, 45, 20, 5, 0, 0.8, 0.7, 0.1, 0.8, 0.9],
            'Open_Area_Flight': [150, 100, 50, 25, 15, 0, 0.2, 0.1, 0.8, 0.3, 0.1],
            'Obstacle_Avoidance': [80, 60, 48, 10, -5, 2, 0.5, 0.5, 0.3, 0.7, 0.6],
            'Urban_Transit': [120, 40, 55, 18, 8, 1, 0.6, 0.4, 0.5, 0.6, 0.4]
        }

        self.model.eval()
        print("Behavior Analysis Results:")

        for scenario_name, features in scenarios.items():
            #Normalize input
            normalized_features = (np.array(features) - feat_mean) / feat_std
            test_sequence = [normalized_features] * self.sequence_length
            test_input = torch.FloatTensor([test_sequence])

            with torch.no_grad():
                normalized_prediction = self.model(test_input)
                #deenormalize prediction
                predicted_pos = normalized_prediction.numpy().flatten() * pos_std + pos_mean

            movement_vector = predicted_pos - np.array(features[:3])
            speed = np.linalg.norm(movement_vector) / 0.1  # m/s

            print(f"{scenario_name}:")
            print(f"  Start: [{features[0]:.1f}, {features[1]:.1f}, {features[2]:.1f}]")
            print(f"  Predicted: [{predicted_pos[0]:.1f}, {predicted_pos[1]:.1f}, {predicted_pos[2]:.1f}]")
            print(f"  Movement: [{movement_vector[0]:.1f}, {movement_vector[1]:.1f}, {movement_vector[2]:.1f}]")
            print(f"  Speed: {speed:.1f} m/s")


def run_phase_3_2_3():

    print("PHASE 3.2.3: IMPROVED LSTM Network Extension")

    # Initialize
    lstm_predictor = ImprovedLSTMPredictor()
    strategic_data, kalman_data = lstm_predictor.load_all_data()

    if kalman_data is None:
        print(" can not go on without data")
        return

    # make training data
    sequences, targets, norm_params = lstm_predictor.create_better_training_data(kalman_data, num_sequences=50)

    # train
    trained_model, norm_params = lstm_predictor.train_better_model(sequences, targets, norm_params, epochs=100)

    predicted, actual = lstm_predictor.demonstrate_better_predictions(kalman_data, norm_params)

    # Analyse
    lstm_predictor.analyze_urban_behavior(norm_params)

    return lstm_predictor


if __name__ == "__main__":
    run_phase_3_2_3()