import gymnasium as gym
from gymnasium import spaces
import numpy as np
import cv2
import os
from typing import Dict, Tuple, Any, List
import torch
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import DummyVecEnv
import glob
from collections import deque
import random


class ImprovedGazeEnv(gym.Env):
    """
    Environment with reward shaping and exploration
    """

    def __init__(self, video_folder: str, frame_stack: int = 4, max_videos: int = 5):
        super(ImprovedGazeEnv, self).__init__()

        self.video_files = sorted(glob.glob(os.path.join(video_folder, "*.mp4")))[:max_videos]
        print(f"Loaded {len(self.video_files)} videos for training")

        self.frame_stack = frame_stack
        self.target_size = (84, 84)

        # Action space - continuous gaze coordinates
        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(2,), dtype=np.float32)

        # Observation space
        self.observation_space = spaces.Dict({
            'frames': spaces.Box(low=0, high=255, shape=(frame_stack, *self.target_size, 1), dtype=np.uint8),
            'gaze_history': spaces.Box(low=0.0, high=1.0, shape=(frame_stack, 2), dtype=np.float32)
        })

        self.current_video_idx = 0
        self._load_current_video()

    def _get_salient_points(self, frame: np.ndarray) -> List[Tuple[float, float]]:
        """
        Get multiple points from the frame using different strategies
        """
        salient_points = []

        # Strategy 1: Edge detection
        edges = cv2.Canny(frame.astype(np.uint8), 50, 150)
        if np.sum(edges) > 100:
            edge_points = np.column_stack(np.where(edges > 0))
            for point in edge_points[:10]:  # Sample up to 10 edge points
                y_norm = point[0] / edges.shape[0]
                x_norm = point[1] / edges.shape[1]
                salient_points.append((x_norm, y_norm))

        # Strategy 2: Bright areas
        bright_threshold = np.percentile(frame, 75)
        bright_mask = frame > bright_threshold
        if np.sum(bright_mask) > 50:
            bright_points = np.column_stack(np.where(bright_mask))
            for point in bright_points[:5]:  # Sample up to 5 bright points
                y_norm = point[0] / bright_mask.shape[0]
                x_norm = point[1] / bright_mask.shape[1]
                salient_points.append((x_norm, y_norm))

        # Strategy 3: Motion detection (if we have previous frame)
        if hasattr(self, 'prev_frame') and self.prev_frame is not None:
            diff = cv2.absdiff(frame, self.prev_frame)
            motion_threshold = np.percentile(diff, 80)
            motion_mask = diff > motion_threshold
            if np.sum(motion_mask) > 20:
                motion_points = np.column_stack(np.where(motion_mask))
                for point in motion_points[:3]:  # Sample up to 3 motion points
                    y_norm = point[0] / motion_mask.shape[0]
                    x_norm = point[1] / motion_mask.shape[1]
                    salient_points.append((x_norm, y_norm))

        self.prev_frame = frame.copy()
        return salient_points

    def _traditional_gaze_controller(self, frame: np.ndarray) -> np.ndarray:
        """
        traditional controller that produces more varied gaze patterns
        """
        salient_points = self._get_salient_points(frame)

        if salient_points:
            # Sometimes follow a point, sometimes move smoothly
            if random.random() < 0.7:  # 70% chance to follow a salient point
                chosen_point = random.choice(salient_points)
                return np.array(chosen_point, dtype=np.float32)
            else:  # 30% chance for smooth movement
                if hasattr(self, 'last_gaze') and self.last_gaze is not None:
                    # Move smoothly from last position
                    movement = np.random.normal(0, 0.05, 2)
                    new_gaze = self.last_gaze + movement
                    new_gaze = np.clip(new_gaze, 0.1, 0.9)
                    return new_gaze.astype(np.float32)

        # Fallback: random position but avoid edges
        return np.array([random.uniform(0.2, 0.8), random.uniform(0.2, 0.8)], dtype=np.float32)

    def _calculate_reward(self, predicted_gaze: np.ndarray, expert_gaze: np.ndarray) -> float:
        """
        reward function that encourages meaningful gaze behavior
        """
        distance = np.linalg.norm(predicted_gaze - expert_gaze)

        # Base reward - inverse distance
        reward = (1.0 - distance) * 10.0

        # Bonus for very accurate predictions
        if distance < 0.05:
            reward += 15.0
        elif distance < 0.1:
            reward += 8.0
        elif distance < 0.15:
            reward += 3.0

        # Penalty for predicting near edges (encourage looking at content)
        if (predicted_gaze[0] < 0.1 or predicted_gaze[0] > 0.9 or
                predicted_gaze[1] < 0.1 or predicted_gaze[1] > 0.9):
            reward -= 5.0

        # Encourage some movement (variety in predictions)
        if len(self.gaze_history) > 1:
            last_pred = self.gaze_history[-1]
            movement = np.linalg.norm(predicted_gaze - last_pred)
            if movement > 0.02:  # Good movement
                reward += 2.0
            elif movement < 0.005:  # Too little movement
                reward -= 3.0

        # Bonus for predicting near salient regions
        salient_points = self._get_salient_points(self.frame_buffer[-1])
        if salient_points:
            min_salient_dist = min(np.linalg.norm(predicted_gaze - np.array(point))
                                   for point in salient_points)
            if min_salient_dist < 0.1:
                reward += 4.0

        return float(reward)

    def _load_current_video(self):
        """Load current video"""
        if self.current_video_idx >= len(self.video_files):
            self.current_video_idx = 0

        video_path = self.video_files[self.current_video_idx]
        self.cap = cv2.VideoCapture(video_path)

        if not self.cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")

        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

        ret, frame = self.cap.read()
        if ret:
            self.height, self.width = frame.shape[:2]
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        else:
            self.width, self.height = 1920, 1080

        print(f"Processing: {os.path.basename(video_path)} - {self.total_frames} frames")

        self.frame_buffer = deque(maxlen=self.frame_stack)
        self.gaze_history = deque(maxlen=self.frame_stack)
        self.current_frame_idx = 0
        self.prev_frame = None
        self.last_gaze = None

    def _preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """frame to grayscale and resize"""
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return cv2.resize(gray_frame, self.target_size)

    def reset(self, seed: int = None, options: dict = None) -> Tuple[Dict, dict]:
        """Reset environment"""
        super().reset(seed=seed)
        self._load_current_video()

        self.frame_buffer.clear()
        self.gaze_history.clear()
        self.prev_frame = None

        # Load initial frames with varied starting positions
        for i in range(self.frame_stack):
            ret, frame = self.cap.read()
            if ret:
                processed_frame = self._preprocess_frame(frame)
                self.frame_buffer.append(processed_frame)
                # Start with different positions to encourage variety
                start_pos = np.array([0.3 + 0.4 * (i / self.frame_stack),
                                      0.3 + 0.4 * ((i + 1) / self.frame_stack)], dtype=np.float32)
                self.gaze_history.append(start_pos)
            else:
                black_frame = np.zeros(self.target_size, dtype=np.uint8)
                self.frame_buffer.append(black_frame)
                self.gaze_history.append(np.array([0.5, 0.5], dtype=np.float32))

        self.current_frame_idx = 0
        self.last_gaze = self.gaze_history[-1].copy()

        return self._get_observation(), {}

    def _get_observation(self) -> Dict:
        """Get current observation"""
        frames = np.array(self.frame_buffer)
        frames = np.expand_dims(frames, axis=-1)

        return {
            'frames': frames,
            'gaze_history': np.array(self.gaze_history, dtype=np.float32)
        }

    def step(self, action: np.ndarray) -> Tuple[Dict, float, bool, bool, dict]:
        """Step environment"""
        ret, frame = self.cap.read()

        if not ret or self.current_frame_idx >= self.total_frames - 1:
            self.current_video_idx = (self.current_video_idx + 1) % len(self.video_files)
            terminated = True
            truncated = False
            return self._get_observation(), 0.0, terminated, truncated, {}

        processed_frame = self._preprocess_frame(frame)
        expert_action = self._traditional_gaze_controller(processed_frame)

        # Calculate reward
        reward = self._calculate_reward(action, expert_action)

        # Update buffers
        self.frame_buffer.append(processed_frame)
        self.gaze_history.append(action)
        self.current_frame_idx += 1
        self.last_gaze = action.copy()

        terminated = self.current_frame_idx >= self.total_frames - 1
        truncated = False

        info = {
            'true_gaze': expert_action,
            'predicted_gaze': action,
            'frame_idx': self.current_frame_idx,
            'video': os.path.basename(self.video_files[self.current_video_idx]),
            'distance': np.linalg.norm(action - expert_action)
        }

        return self._get_observation(), reward, terminated, truncated, info

    def close(self):
        """Clean up"""
        if hasattr(self, 'cap') and self.cap:
            self.cap.release()


class ImprovedCNN(BaseFeaturesExtractor):
    """Improved neural network with better architecture"""

    def __init__(self, observation_space: gym.spaces.Dict, features_dim: int = 512):
        super(ImprovedCNN, self).__init__(observation_space, features_dim)

        # Enhanced CNN for frames
        n_input_channels = observation_space['frames'].shape[0]

        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((6, 6)),
            nn.Flatten(),
        )

        # Compute CNN output size
        with torch.no_grad():
            sample = torch.zeros(1, n_input_channels, 84, 84)
            n_flatten = self.cnn(sample).shape[1]

        # Enhanced gaze history processing
        self.gaze_net = nn.Sequential(
            nn.Linear(observation_space['gaze_history'].shape[0] * 2, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
        )

        # Combined features with more capacity
        self.combined_net = nn.Sequential(
            nn.Linear(n_flatten + 64, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        # Process frames
        frames = observations['frames']
        batch_size = frames.shape[0]
        frames = frames.squeeze(-1)  # Remove channel dimension
        frames = frames.float() / 255.0
        frame_features = self.cnn(frames)

        # Process gaze history
        gaze_history = observations['gaze_history']
        gaze_flat = gaze_history.reshape(batch_size, -1)
        gaze_features = self.gaze_net(gaze_flat)

        # Combine
        combined = torch.cat([frame_features, gaze_features], dim=1)
        return self.combined_net(combined)


def train_improved_model(video_folder: str, total_timesteps: int = 50000):
    """Train improved model with better settings"""
    print("Training Improved Model...")

    # Create environment
    env = DummyVecEnv([lambda: ImprovedGazeEnv(video_folder, max_videos=5)])

    # Enhanced policy configuration
    policy_kwargs = dict(
        features_extractor_class=ImprovedCNN,
        features_extractor_kwargs=dict(features_dim=512),
        net_arch=[512, 256, 128]  # Deeper network
    )

    # Improved training settings
    model = PPO(
        "MultiInputPolicy",
        env,
        policy_kwargs=policy_kwargs,
        learning_rate=1e-4,  # Lower learning rate for stability
        n_steps=2048,  # More steps per update
        batch_size=128,
        n_epochs=10,
        gamma=0.99,
        ent_coef=0.1,  # Higher entropy for more exploration
        clip_range=0.2,
        verbose=1,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )

    print("Training improved model...")
    model.learn(total_timesteps=total_timesteps)
    model.save("improved_gaze_model")
    print("Improved model saved!")

    env.close()
    return model


def test_improved_model(video_folder: str, model_path: str = "improved_gaze_model"):
    """Test the improved model"""
    print("Testing Improved Model...")

    try:
        model = PPO.load(model_path)
        print(f"Model loaded: {model_path}")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    video_files = sorted(glob.glob(os.path.join(video_folder, "*.mp4")))[:2]

    for video_path in video_files:
        print(f"\nTesting on: {os.path.basename(video_path)}")

        cap = cv2.VideoCapture(video_path)
        frame_buffer = deque(maxlen=4)
        gaze_history = deque(maxlen=4)

        # Initialize with varied positions
        for i in range(4):
            gaze_history.append(np.array([0.3 + 0.4 * (i / 4), 0.3 + 0.4 * ((i + 1) / 4)], dtype=np.float32))

        # Load initial frames
        for _ in range(4):
            ret, frame = cap.read()
            if ret:
                processed = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (84, 84))
                frame_buffer.append(processed)

        predictions = []
        for i in range(100):  # Test 100 frames
            # Create observation
            frames_array = np.array(frame_buffer)
            frames_array = np.expand_dims(frames_array, axis=-1)

            obs = {
                'frames': frames_array,
                'gaze_history': np.array(gaze_history, dtype=np.float32)
            }

            # Predict (try both deterministic and stochastic)
            if i < 50:
                action, _ = model.predict(obs, deterministic=True)
            else:
                action, _ = model.predict(obs, deterministic=False)

            predictions.append(action)

            # Update
            ret, frame = cap.read()
            if not ret:
                break

            processed = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (84, 84))
            frame_buffer.append(processed)
            gaze_history.append(action)

        cap.release()

        # Analyze predictions
        predictions = np.array(predictions)
        print(f"Prediction Statistics:")
        print(f"  Mean: ({predictions[:, 0].mean():.3f}, {predictions[:, 1].mean():.3f})")
        print(f"  Std:  ({predictions[:, 0].std():.3f}, {predictions[:, 1].std():.3f})")
        print(f"  Range X: [{predictions[:, 0].min():.3f}, {predictions[:, 0].max():.3f}]")
        print(f"  Range Y: [{predictions[:, 1].min():.3f}, {predictions[:, 1].max():.3f}]")

        # Movement analysis
        movements = np.linalg.norm(predictions[1:] - predictions[:-1], axis=1)
        print(f"  Movement stats - Mean: {movements.mean():.4f}, Std: {movements.std():.4f}")
        print(f"  Max movement: {movements.max():.4f}, Min movement: {movements.min():.4f}")

        # Assessment
        if predictions.std() < 0.01:
            print("  ❌ Model is predicting constant values")
        elif movements.mean() < 0.01:
            print("  ⚠️  Model has very little movement")
        else:
            print("  ✅ Model shows varied gaze patterns!")


if __name__ == "__main__":
    video_folder = r"C:\Users\sudha\PycharmProjects\RL Project 1\stimuli"

    print("Improved Gaze Tracking Training")
    print("=" * 50)

    # Train the improved model
    model = train_improved_model(video_folder, total_timesteps=50000)

    # Test it
    print("\n" + "=" * 50)
    test_improved_model(video_folder, "improved_gaze_model")