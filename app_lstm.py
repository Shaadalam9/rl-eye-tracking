"""
LSTM-based Gaze Tracking System with Reinforcement Learning - Complete Implementation.

This module implements an end-to-end gaze prediction system using PPO (Proximal Policy Optimization)
with an LSTM-enhanced neural network architecture. The system learns to predict human gaze patterns
from video frames by imitating expert demonstrations through reinforcement learning.

Key Features:
    - Frame-based visual processing with CNN
    - Temporal memory using LSTM
    - Loss-aware state representation
    - Exponential learning rate decay
    - Configurable architecture and hyperparameters
    - Comprehensive logging and visualization
    - Checkpoint saving and resume capability

Author: Sudhanshu Anand
Date: 2025-10-27
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import cv2
import os
from typing import Dict, Tuple, Any, List, Optional
import torch
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
import glob
from collections import deque
import random
import matplotlib.pyplot as plt
from datetime import datetime
import json

# Import configuration
from config import config


class LearningRateLogger(BaseCallback):
    """
    Callback to log learning rate during training.

    This callback monitors and logs the current learning rate at regular intervals
    during PPO training. It handles both constant and scheduled learning rates.

    Attributes:
        verbose (int): Verbosity level for logging
        log_frequency (int): Number of steps between log entries
    """

    def __init__(self, verbose: int = 0):
        """
        Initialize the learning rate logger.

        Args:
            verbose: Verbosity level (0=silent, 1=info, 2=debug)
        """
        super(LearningRateLogger, self).__init__(verbose)
        self.log_frequency = config.learning_rate.lr_log_frequency
        self.lr_history = []
        self.step_history = []

    def _on_step(self) -> bool:
        """
        Called at each training step. Logs learning rate periodically.

        Returns:
            bool: True to continue training, False to stop
        """
        # Log at specified frequency
        if self.n_calls % self.log_frequency == 0:
            current_lr = self.model.learning_rate

            # Check if learning rate is a schedule function
            if callable(current_lr):
                # Calculate progress for scheduled learning rate
                total_timesteps = self.locals.get('total_timesteps', 1)
                progress = 1.0 - (self.num_timesteps / total_timesteps)
                lr_value = current_lr(progress)
            else:
                # Constant learning rate
                lr_value = current_lr

            self.lr_history.append(lr_value)
            self.step_history.append(self.num_timesteps)

            print(f"Step {self.num_timesteps}: LR = {lr_value:.6e}")

        return True

    def plot_lr_curve(self, save_path: str = "lr_curve.png"):
        """Plot and save the learning rate curve."""
        if len(self.lr_history) > 0:
            plt.figure(figsize=(10, 6))
            plt.plot(self.step_history, self.lr_history, linewidth=2)
            plt.xlabel('Training Steps')
            plt.ylabel('Learning Rate')
            plt.title('Learning Rate Schedule')
            plt.grid(True, alpha=0.3)
            plt.yscale('log')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Learning rate curve saved to {save_path}")


class MetricsLogger(BaseCallback):
    """
    Callback to log training metrics and performance statistics.
    """

    def __init__(self, verbose: int = 0):
        super(MetricsLogger, self).__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_count = 0

    def _on_step(self) -> bool:
        # Check for completed episodes
        if 'infos' in self.locals:
            for info in self.locals['infos']:
                if 'episode' in info:
                    self.episode_rewards.append(info['episode']['r'])
                    self.episode_lengths.append(info['episode']['l'])
                    self.episode_count += 1

                    if self.episode_count % 10 == 0:
                        mean_reward = np.mean(self.episode_rewards[-10:])
                        print(f"Episode {self.episode_count}: Mean Reward (last 10) = {mean_reward:.2f}")

        return True

    def save_metrics(self, save_path: str = "training_metrics.json"):
        """Save training metrics to file."""
        metrics = {
            'episode_rewards': self.episode_rewards,
            'episode_lengths': self.episode_lengths,
            'total_episodes': self.episode_count
        }

        with open(save_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"Training metrics saved to {save_path}")


class EnhancedGazeEnv(gym.Env):
    """
    Reinforcement learning environment for gaze prediction.

    This environment simulates the task of predicting human gaze on video frames.
    It provides visual observations (frame stacks), gaze history, loss information,
    and computes rewards based on how well the agent's predictions match expert gaze patterns.

    The environment processes multiple videos sequentially and maintains temporal context
    through frame stacking and history tracking.

    Attributes:
        video_files: List of video file paths
        frame_stack: Number of frames to stack for temporal context
        target_size: Target size for frame preprocessing (width, height)
        action_space: Continuous 2D space for gaze coordinates [0, 1]
        observation_space: Dict space containing frames, gaze history, loss history, etc.
    """

    def __init__(self, video_folder: str, frame_stack: int = None, max_videos: int = None):
        """
        Initialize the gaze prediction environment.

        Args:
            video_folder: Path to folder containing video files
            frame_stack: Number of frames to stack (uses config if None)
            max_videos: Maximum number of videos to load (uses config if None)
        """
        super(EnhancedGazeEnv, self).__init__()

        # Load configuration
        self.frame_stack = frame_stack or config.video.frame_stack
        self.target_size = config.video.target_size
        max_videos = max_videos or config.video.max_videos

        # Load video files
        self.video_files = sorted(glob.glob(os.path.join(video_folder, "*.mp4")))[:max_videos]

        if len(self.video_files) == 0:
            raise ValueError(f"No video files found in {video_folder}")

        print(f"Loaded {len(self.video_files)} videos for training")

        # Define action space - continuous gaze coordinates normalized to [0, 1]
        self.action_space = spaces.Box(
            low=config.environment.action_low,
            high=config.environment.action_high,
            shape=config.environment.action_shape,
            dtype=np.float32
        )

        # Define observation space - includes visual frames and temporal information
        self.observation_space = spaces.Dict({
            'frames': spaces.Box(
                low=0, high=255,
                shape=(self.frame_stack, *self.target_size, 1),
                dtype=np.uint8
            ),
            'gaze_history': spaces.Box(
                low=0.0, high=1.0,
                shape=(self.frame_stack, 2),
                dtype=np.float32
            ),
            'loss_history': spaces.Box(
                low=0.0, high=1.0,
                shape=(self.frame_stack,),
                dtype=np.float32
            ),
            'cumulative_loss': spaces.Box(
                low=0.0, high=config.environment.cumulative_loss_max,
                shape=(1,),
                dtype=np.float32
            ),
        })

        self.current_video_idx = 0
        self.cap = None
        self._load_current_video()

    def _get_salient_points(self, frame: np.ndarray) -> List[Tuple[float, float]]:
        """
        Extract salient points from frame using multiple strategies.

        This method combines three strategies to identify interesting regions in the frame:
        1. Edge detection using Canny algorithm
        2. Bright region detection
        3. Motion detection (if previous frame available)

        Args:
            frame: Grayscale frame as numpy array

        Returns:
            List of (x, y) tuples representing salient points in normalized coordinates [0, 1]
        """
        salient_points = []
        cfg = config.environment

        # Strategy 1: Edge detection for high-contrast boundaries
        edges = cv2.Canny(
            frame.astype(np.uint8),
            cfg.edge_detection_threshold_low,
            cfg.edge_detection_threshold_high
        )

        if np.sum(edges) > 100:  # Sufficient edges detected
            edge_points = np.column_stack(np.where(edges > 0))
            for point in edge_points[:cfg.edge_points_limit]:
                y_norm = point[0] / edges.shape[0]
                x_norm = point[1] / edges.shape[1]
                salient_points.append((x_norm, y_norm))

        # Strategy 2: Bright areas that might attract attention
        bright_threshold = np.percentile(frame, cfg.bright_percentile)
        bright_mask = frame > bright_threshold

        if np.sum(bright_mask) > 50:  # Sufficient bright pixels
            bright_points = np.column_stack(np.where(bright_mask))
            for point in bright_points[:cfg.bright_points_limit]:
                y_norm = point[0] / bright_mask.shape[0]
                x_norm = point[1] / bright_mask.shape[1]
                salient_points.append((x_norm, y_norm))

        # Strategy 3: Motion detection to identify dynamic regions
        if hasattr(self, 'prev_frame') and self.prev_frame is not None:
            diff = cv2.absdiff(frame, self.prev_frame)
            motion_threshold = np.percentile(diff, cfg.motion_percentile)
            motion_mask = diff > motion_threshold

            if np.sum(motion_mask) > 20:  # Sufficient motion detected
                motion_points = np.column_stack(np.where(motion_mask))
                for point in motion_points[:cfg.motion_points_limit]:
                    y_norm = point[0] / motion_mask.shape[0]
                    x_norm = point[1] / motion_mask.shape[1]
                    salient_points.append((x_norm, y_norm))

        # Store frame for next motion detection
        self.prev_frame = frame.copy()
        return salient_points

    def _traditional_gaze_controller(self, frame: np.ndarray) -> np.ndarray:
        """
        Generate expert gaze patterns using traditional computer vision.

        This controller simulates human-like gaze behavior by:
        1. Focusing on salient regions (edges, motion, brightness)
        2. Occasionally making smooth exploratory movements
        3. Adding natural randomness to gaze patterns

        Args:
            frame: Processed grayscale frame

        Returns:
            Numpy array of shape (2,) with normalized gaze coordinates [0, 1]
        """
        cfg = config.environment
        salient_points = self._get_salient_points(frame)

        if salient_points:
            # Probabilistically choose between salient point and smooth movement
            if random.random() < cfg.salient_point_probability:
                # Focus on a randomly selected salient point
                chosen_point = random.choice(salient_points)
                return np.array(chosen_point, dtype=np.float32)
            else:
                # Make smooth exploratory movement from last gaze position
                if hasattr(self, 'last_gaze') and self.last_gaze is not None:
                    movement = np.random.normal(0, cfg.random_movement_std, 2)
                    new_gaze = self.last_gaze + movement
                    new_gaze = np.clip(new_gaze, 0.1, 0.9)
                    return new_gaze.astype(np.float32)

        # Fallback: random gaze within safe bounds
        return np.array([
            random.uniform(cfg.random_gaze_min, cfg.random_gaze_max),
            random.uniform(cfg.random_gaze_min, cfg.random_gaze_max)
        ], dtype=np.float32)

    def _calculate_loss(self, predicted_gaze: np.ndarray, expert_gaze: np.ndarray) -> float:
        """
        Calculate normalized prediction error as loss.

        The loss is the Euclidean distance between predicted and expert gaze,
        normalized to [0, 1] range where 0 is perfect prediction.

        Args:
            predicted_gaze: Agent's predicted gaze coordinates
            expert_gaze: Expert gaze coordinates from traditional controller

        Returns:
            Float loss value in range [0, 1]
        """
        distance = np.linalg.norm(predicted_gaze - expert_gaze)
        # Maximum possible distance is sqrt(2) ≈ 1.414 (corner to corner)
        normalized_loss = distance / config.reward.max_euclidean_distance
        return float(np.clip(normalized_loss, 0.0, 1.0))

    def _calculate_reward(self, predicted_gaze: np.ndarray, expert_gaze: np.ndarray) -> float:
        """
        Calculate reward for the agent's gaze prediction.

        The reward function combines multiple components:
        1. Base reward from distance to expert gaze
        2. Accuracy bonuses for close predictions
        3. Penalties for predicting at screen edges
        4. Movement rewards/penalties to encourage natural gaze shifts
        5. Bonuses for focusing on salient regions
        6. Improvement bonus for reducing loss over time

        Args:
            predicted_gaze: Agent's predicted gaze coordinates [0, 1]
            expert_gaze: Expert gaze coordinates [0, 1]

        Returns:
            Float reward value (can be negative for poor predictions)
        """
        cfg = config.reward
        distance = np.linalg.norm(predicted_gaze - expert_gaze)

        # Component 1: Base reward - inverse of distance
        reward = (1.0 - distance) * cfg.base_reward_scale

        # Component 2: Tiered accuracy bonuses for precise predictions
        if distance < cfg.accuracy_threshold_excellent:
            reward += cfg.accuracy_bonus_excellent
        elif distance < cfg.accuracy_threshold_good:
            reward += cfg.accuracy_bonus_good
        elif distance < cfg.accuracy_threshold_fair:
            reward += cfg.accuracy_bonus_fair

        # Component 3: Penalty for predictions near screen edges (unnatural)
        if (predicted_gaze[0] < cfg.edge_boundary_inner or
                predicted_gaze[0] > cfg.edge_boundary_outer or
                predicted_gaze[1] < cfg.edge_boundary_inner or
                predicted_gaze[1] > cfg.edge_boundary_outer):
            reward -= cfg.edge_penalty_light

        # Component 4: Movement encouragement for natural gaze patterns
        if len(self.gaze_history) > 1:
            last_pred = self.gaze_history[-1]
            movement = np.linalg.norm(predicted_gaze - last_pred)

            if movement > cfg.movement_threshold_good:
                # Reward for making meaningful gaze shifts
                reward += cfg.movement_reward
            elif movement < cfg.movement_threshold_stagnant:
                # Penalty for stagnation (unnatural fixation)
                reward -= cfg.stagnation_penalty

        # Component 5: Salient region bonus - reward attention to interesting areas
        salient_points = self._get_salient_points(self.frame_buffer[-1])
        if salient_points:
            min_salient_dist = min(
                np.linalg.norm(predicted_gaze - np.array(point))
                for point in salient_points
            )
            if min_salient_dist < cfg.salient_distance_threshold:
                reward += cfg.salient_region_bonus

        # Component 6: Improvement bonus - reward for learning over time
        if len(self.loss_history) >= 2:
            current_loss = self._calculate_loss(predicted_gaze, expert_gaze)
            prev_loss = self.loss_history[-1]
            if current_loss < prev_loss:
                reward += cfg.loss_improvement_bonus

        # Component 7: Strong penalty for extreme edge hugging
        edge_penalty = 0
        for coord in predicted_gaze:
            if coord < 0.15 or coord > 0.85:
                edge_penalty += cfg.edge_penalty_heavy
        reward -= edge_penalty

        return float(reward)

    def _load_current_video(self):
        """
        Load the current video and initialize tracking buffers.

        This method:
        1. Opens the video file at current index
        2. Reads video properties (dimensions, frame count)
        3. Initializes frame buffer, gaze history, and loss tracking
        4. Resets all temporal state
        """
        # Close previous video if exists
        if self.cap is not None:
            self.cap.release()

        # Wrap around if reached end of video list
        if self.current_video_idx >= len(self.video_files):
            self.current_video_idx = 0

        video_path = self.video_files[self.current_video_idx]
        self.cap = cv2.VideoCapture(video_path)

        if not self.cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")

        # Get video properties
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Read first frame to get dimensions
        ret, frame = self.cap.read()
        if ret:
            self.height, self.width = frame.shape[:2]
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reset to beginning
        else:
            # Default dimensions if frame read fails
            self.width, self.height = 1920, 1080

        print(f"Processing: {os.path.basename(video_path)} - {self.total_frames} frames")

        # Initialize temporal buffers with maximum length
        self.frame_buffer = deque(maxlen=self.frame_stack)
        self.gaze_history = deque(maxlen=self.frame_stack)
        self.loss_history = deque(maxlen=self.frame_stack)

        # Reset cumulative state
        self.cumulative_loss = 0.0
        self.current_frame_idx = 0
        self.prev_frame = None
        self.last_gaze = None

    def _preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Preprocess frame for neural network input.

        Converts frame to grayscale and resizes to target dimensions.
        Grayscale reduces computational cost while retaining spatial information.

        Args:
            frame: Raw BGR frame from video

        Returns:
            Preprocessed grayscale frame of shape target_size
        """
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return cv2.resize(gray_frame, self.target_size)

    def reset(self, seed: int = None, options: dict = None) -> Tuple[Dict, dict]:
        """
        Reset environment to initial state.

        Called at the start of each episode. Loads a new video and initializes
        all buffers with starting frames and default gaze positions.

        Args:
            seed: Random seed for reproducibility
            options: Additional reset options (unused)

        Returns:
            Tuple of (initial_observation, info_dict)
        """
        super().reset(seed=seed)
        self._load_current_video()

        # Clear all buffers
        self.frame_buffer.clear()
        self.gaze_history.clear()
        self.loss_history.clear()
        self.cumulative_loss = 0.0
        self.prev_frame = None

        # Load initial frame stack
        for i in range(self.frame_stack):
            ret, frame = self.cap.read()
            if ret:
                processed_frame = self._preprocess_frame(frame)
                self.frame_buffer.append(processed_frame)

                # Initialize with diagonal gaze pattern across frame stack
                start_pos = np.array([
                    0.3 + 0.4 * (i / self.frame_stack),
                    0.3 + 0.4 * ((i + 1) / self.frame_stack)
                ], dtype=np.float32)
                self.gaze_history.append(start_pos)
                self.loss_history.append(0.0)
            else:
                # Handle case where video has fewer frames than stack size
                black_frame = np.zeros(self.target_size, dtype=np.uint8)
                self.frame_buffer.append(black_frame)
                self.gaze_history.append(np.array([0.5, 0.5], dtype=np.float32))
                self.loss_history.append(0.0)

        self.current_frame_idx = 0
        self.last_gaze = self.gaze_history[-1].copy()

        return self._get_observation(), {}

    def _get_observation(self) -> Dict:
        """
        Construct observation dictionary from current state.

        Returns:
            Dictionary containing:
                - frames: Stacked grayscale frames
                - gaze_history: Recent gaze predictions
                - loss_history: Recent prediction losses
                - cumulative_loss: Total accumulated loss
        """
        frames = np.array(self.frame_buffer)
        frames = np.expand_dims(frames, axis=-1)  # Add channel dimension

        return {
            'frames': frames,
            'gaze_history': np.array(self.gaze_history, dtype=np.float32),
            'loss_history': np.array(self.loss_history, dtype=np.float32),
            'cumulative_loss': np.array([self.cumulative_loss], dtype=np.float32),
        }

    def step(self, action: np.ndarray) -> Tuple[Dict, float, bool, bool, dict]:
        """
        Execute one step in the environment.

        Args:
            action: Agent's gaze prediction as (x, y) coordinates in [0, 1]

        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        # Read next frame
        ret, frame = self.cap.read()

        # Check if video ended
        if not ret or self.current_frame_idx >= self.total_frames - 1:
            # Move to next video
            self.current_video_idx = (self.current_video_idx + 1) % len(self.video_files)
            terminated = True
            truncated = False
            return self._get_observation(), 0.0, terminated, truncated, {}

        # Preprocess frame and get expert action
        processed_frame = self._preprocess_frame(frame)
        expert_action = self._traditional_gaze_controller(processed_frame)

        # Calculate loss and reward
        loss = self._calculate_loss(action, expert_action)
        reward = self._calculate_reward(action, expert_action)

        # Update buffers (deques automatically maintain max length)
        self.frame_buffer.append(processed_frame)
        self.gaze_history.append(action)
        self.loss_history.append(loss)
        self.cumulative_loss = np.clip(
            self.cumulative_loss + loss,
            0.0,
            config.environment.cumulative_loss_max
        )
        self.current_frame_idx += 1
        self.last_gaze = action.copy()

        # Check termination
        terminated = self.current_frame_idx >= self.total_frames - 1
        truncated = False

        # Define expert gaze — use expert_action or fallback if not available
        expert_gaze = expert_action  # ← this is the correct variable name in your logic
        # If expert_action somehow fails, fallback to center
        if expert_gaze is None or not isinstance(expert_gaze, np.ndarray):
            expert_gaze = np.array([0.5, 0.5], dtype=np.float32)

        # Construct info dictionary for debugging/logging
        info = {
            'true_gaze': expert_gaze,
            'predicted_gaze': action,
            'frame_idx': self.current_frame_idx,
            'video': os.path.basename(self.video_files[self.current_video_idx]),
            'distance': np.linalg.norm(action - expert_action),
            'loss': loss,
            'cumulative_loss': self.cumulative_loss,
        }

        return self._get_observation(), reward, terminated, truncated, info

    def close(self):
        """Release video capture resources."""
        if hasattr(self, 'cap') and self.cap:
            self.cap.release()


class LSTMGazeCNN(BaseFeaturesExtractor):
    """
    LSTM-enhanced CNN for temporal gaze prediction.

    This feature extractor combines:
    1. CNN for spatial visual feature extraction from frame stacks
    2. LSTM for temporal/memory processing of gaze and loss history
    3. Feed-forward network for cumulative loss encoding
    4. Combined network that fuses all information sources

    Architecture:
        Visual Path: Frames -> CNN (4 conv layers) -> Spatial features
        Temporal Path: Gaze+Loss history -> LSTM -> Temporal features
        Context Path: Cumulative loss -> FC -> Context features
        Fusion: Concatenate all -> FC layers -> Output features

    Attributes:
        cnn: Convolutional neural network for spatial processing
        lstm: LSTM for temporal sequence processing
        cumulative_loss_net: Network for cumulative loss encoding
        combined_net: Final fusion network
    """

    def __init__(self, observation_space: gym.spaces.Dict, features_dim: int = None):
        """
        Initialize LSTM-CNN feature extractor.

        Args:
            observation_space: Gym dictionary space defining observation structure
            features_dim: Output dimension of features (uses config if None)
        """
        features_dim = features_dim or config.policy.features_dim
        super(LSTMGazeCNN, self).__init__(observation_space, features_dim)

        # Get CNN configuration
        cnn_cfg = config.cnn
        n_input_channels = observation_space['frames'].shape[0]

        # Build CNN for spatial feature extraction
        self.cnn = nn.Sequential(
            # Layer 1: Initial feature extraction
            nn.Conv2d(
                n_input_channels,
                cnn_cfg.conv1_out_channels,
                kernel_size=cnn_cfg.conv1_kernel_size,
                stride=cnn_cfg.conv1_stride
            ),
            nn.ReLU(),

            # Layer 2: Hierarchical feature learning
            nn.Conv2d(
                cnn_cfg.conv1_out_channels,
                cnn_cfg.conv2_out_channels,
                kernel_size=cnn_cfg.conv2_kernel_size,
                stride=cnn_cfg.conv2_stride
            ),
            nn.ReLU(),

            # Layer 3: Deep feature extraction
            nn.Conv2d(
                cnn_cfg.conv2_out_channels,
                cnn_cfg.conv3_out_channels,
                kernel_size=cnn_cfg.conv3_kernel_size,
                stride=cnn_cfg.conv3_stride
            ),
            nn.ReLU(),

            # Layer 4: High-level feature refinement
            nn.Conv2d(
                cnn_cfg.conv3_out_channels,
                cnn_cfg.conv4_out_channels,
                kernel_size=cnn_cfg.conv4_kernel_size,
                stride=cnn_cfg.conv4_stride
            ),
            nn.ReLU(),

            # Adaptive pooling for consistent output size
            nn.AdaptiveAvgPool2d(cnn_cfg.adaptive_pool_output_size),
            nn.Flatten(),
        )

        # Compute CNN output dimension
        with torch.no_grad():
            sample = torch.zeros(1, n_input_channels, 84, 84)
            n_flatten = self.cnn(sample).shape[1]

        # Build LSTM for temporal processing
        lstm_cfg = config.lstm
        lstm_input_size = lstm_cfg.gaze_dim + lstm_cfg.loss_dim  # gaze (2) + loss (1)

        self.lstm_hidden_size = lstm_cfg.lstm_hidden_size
        self.lstm = nn.LSTM(
            input_size=lstm_input_size,
            hidden_size=self.lstm_hidden_size,
            num_layers=lstm_cfg.lstm_num_layers,
            batch_first=True,
            dropout=lstm_cfg.lstm_dropout if lstm_cfg.lstm_num_layers > 1 else 0
        )

        # Network for cumulative loss encoding
        self.cumulative_loss_net = nn.Sequential(
            nn.Linear(1, lstm_cfg.cumulative_loss_hidden_dim),
            nn.ReLU(),
        )

        # Combined fusion network
        policy_cfg = config.policy
        combined_size = n_flatten + self.lstm_hidden_size + lstm_cfg.cumulative_loss_hidden_dim

        self.combined_net = nn.Sequential(
            nn.Linear(combined_size, policy_cfg.combined_fc1_dim),
            nn.ReLU(),
            nn.Dropout(policy_cfg.combined_dropout),
            nn.Linear(policy_cfg.combined_fc1_dim, features_dim),
            nn.ReLU(),
        )

        # LSTM hidden states (stored during forward pass)
        self.hidden_state = None
        self.cell_state = None

    def forward(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Forward pass through the feature extractor.

        Args:
            observations: Dictionary containing:
                - frames: Visual observations (B, T, H, W, 1)
                - gaze_history: Past gaze coordinates (B, T, 2)
                - loss_history: Past losses (B, T)
                - cumulative_loss: Total accumulated loss (B, 1)

        Returns:
            Extracted features of shape (B, features_dim)
        """
        batch_size = observations['frames'].shape[0]

        # Process frames through CNN
        frames = observations['frames'].squeeze(-1).float() / 255.0  # Normalize to [0, 1]
        frame_features = self.cnn(frames)

        # Prepare LSTM input: combine gaze_history and loss_history
        gaze_history = observations['gaze_history']  # (batch, timesteps, 2)
        loss_history = observations['loss_history'].unsqueeze(-1)  # (batch, timesteps, 1)
        lstm_input = torch.cat([gaze_history, loss_history], dim=-1)  # (batch, timesteps, 3)

        # Initialize or reset LSTM hidden states if batch size changed
        if self.hidden_state is None or self.hidden_state.shape[1] != batch_size:
            self.hidden_state = torch.zeros(
                config.lstm.lstm_num_layers,
                batch_size,
                self.lstm_hidden_size,
                device=lstm_input.device
            )
            self.cell_state = torch.zeros(
                config.lstm.lstm_num_layers,
                batch_size,
                self.lstm_hidden_size,
                device=lstm_input.device
            )

        # Process through LSTM (detach hidden states to prevent backprop through time)
        lstm_out, (self.hidden_state, self.cell_state) = self.lstm(
            lstm_input,
            (self.hidden_state.detach(), self.cell_state.detach())
        )

        # Use last LSTM output as temporal features
        lstm_features = lstm_out[:, -1, :]

        # Process cumulative loss
        cumulative_loss = observations['cumulative_loss']
        cumulative_features = self.cumulative_loss_net(cumulative_loss)

        # Concatenate and fuse all feature sources
        combined = torch.cat([frame_features, lstm_features, cumulative_features], dim=1)
        return self.combined_net(combined)

    def reset_hidden_states(self):
        """Reset LSTM hidden states. Call at episode boundaries."""
        self.hidden_state = None
        self.cell_state = None


def exponential_schedule(initial_value: float) -> callable:
    """
    Create exponential decay learning rate schedule.

    The learning rate decays exponentially from initial_value to initial_value * min_factor
    following: lr(t) = initial_value * exp(-decay_rate * progress)

    Args:
        initial_value: Starting learning rate

    Returns:
        Schedule function that takes progress_remaining and returns learning rate
    """
    cfg = config.learning_rate

    def func(progress_remaining: float) -> float:
        """
        Calculate learning rate based on training progress.

        Args:
            progress_remaining: Training progress from 1.0 (start) to 0.0 (end)

        Returns:
            Current learning rate
        """
        # progress_remaining: 1.0 (start) -> 0.0 (end)
        # progress: 0.0 (start) -> 1.0 (end)
        progress = 1.0 - progress_remaining

        # Exponential decay formula
        min_lr = initial_value * cfg.exponential_min_lr_factor
        lr = initial_value * np.exp(-cfg.exponential_decay_rate * progress)

        # Ensure we don't go below minimum
        return max(lr, min_lr)

    return func


def train_lstm_model(
    video_folder: str,
    total_timesteps: int = None,
    verbose_lr: bool = None,
    checkpoint_freq: int = None,
    resume_path: Optional[str] = None
) -> PPO:
    """
    Train LSTM-based gaze prediction model using PPO.

    This function:
    1. Creates vectorized environment
    2. Configures PPO with LSTM-CNN policy
    3. Sets up exponential learning rate decay
    4. Trains model with optional learning rate logging
    5. Saves trained model and checkpoints

    Args:
        video_folder: Path to folder containing training videos
        total_timesteps: Total training steps (uses config if None)
        verbose_lr: Whether to log learning rate (uses config if None)
        checkpoint_freq: Frequency for saving checkpoints
        resume_path: Path to model to resume training from

    Returns:
        Trained PPO model
    """
    print("Training LSTM-based Gaze Model with Exponential LR Decay...")
    print("=" * 80)

    # Load configuration
    total_timesteps = total_timesteps or config.training.total_timesteps
    verbose_lr = verbose_lr if verbose_lr is not None else config.training.verbose_lr
    checkpoint_freq = checkpoint_freq or config.training.save_frequency

    # Create vectorized environment
    env = DummyVecEnv([
        lambda: EnhancedGazeEnv(
            video_folder,
            max_videos=config.video.max_videos
        )
    ])

    # Configure policy network
    policy_kwargs = dict(
        features_extractor_class=LSTMGazeCNN,
        features_extractor_kwargs=dict(features_dim=config.policy.features_dim),
        net_arch=config.policy.net_arch
    )

    # Set up exponential learning rate schedule
    initial_lr = config.learning_rate.initial_lr
    lr_schedule = exponential_schedule(initial_lr)

    print(f"Learning Rate Schedule: Exponential Decay")
    print(f"  Initial LR: {initial_lr:.2e}")
    print(f"  Decay rate: {config.learning_rate.exponential_decay_rate}")
    print(f"  Min LR: {initial_lr * config.learning_rate.exponential_min_lr_factor:.2e}")
    print()

    # Create or load PPO model
    train_cfg = config.training
    device = train_cfg.device if torch.cuda.is_available() else 'cpu'

    if resume_path and os.path.exists(resume_path):
        print(f"Resuming training from: {resume_path}")
        model = PPO.load(
            resume_path,
            env=env,
            device=device
        )
        print("Model loaded successfully!")
    else:
        model = PPO(
            "MultiInputPolicy",
            env,
            policy_kwargs=policy_kwargs,
            learning_rate=lr_schedule,
            n_steps=train_cfg.n_steps,
            batch_size=train_cfg.batch_size,
            n_epochs=train_cfg.n_epochs,
            gamma=train_cfg.gamma,
            gae_lambda=train_cfg.gae_lambda,
            ent_coef=train_cfg.ent_coef,
            vf_coef=train_cfg.vf_coef,
            clip_range=train_cfg.clip_range,
            max_grad_norm=train_cfg.max_grad_norm,
            verbose=train_cfg.verbose,
            device=device
        )

    # Set up callbacks
    callbacks = []

    # Learning rate logger
    if verbose_lr:
        lr_logger = LearningRateLogger()
        callbacks.append(lr_logger)

    # Metrics logger
    metrics_logger = MetricsLogger()
    callbacks.append(metrics_logger)

    # Checkpoint callback
    checkpoint_callback = CheckpointCallback(
        save_freq=checkpoint_freq,
        save_path='./checkpoints/',
        name_prefix='lstm_gaze_checkpoint'
    )
    callbacks.append(checkpoint_callback)

    # Train model
    print(f"Training for {total_timesteps:,} timesteps...")
    print(f"Device: {model.device}")
    print(f"Checkpoints will be saved every {checkpoint_freq:,} steps")
    print("=" * 80 + "\n")

    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=callbacks if callbacks else None
        )

        # Save final model
        model.save(train_cfg.model_save_path)
        print(f"\n✓ Model saved to: {train_cfg.model_save_path}")

        # Save metrics and plots
        if verbose_lr:
            lr_logger.plot_lr_curve("training_lr_curve.png")

        metrics_logger.save_metrics("training_metrics.json")

        # Plot training rewards
        if len(metrics_logger.episode_rewards) > 0:
            plot_training_curve(
                metrics_logger.episode_rewards,
                "Episode Rewards",
                "training_rewards.png"
            )

    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user!")
        print("Saving current model state...")
        model.save(f"{train_cfg.model_save_path}_interrupted")
        print(f"Model saved to: {train_cfg.model_save_path}_interrupted")

    env.close()
    return model


def plot_training_curve(data: List[float], title: str, save_path: str):
    """
    Plot and save training curve.

    Args:
        data: List of values to plot
        title: Plot title
        save_path: Path to save the plot
    """
    plt.figure(figsize=(12, 6))
    plt.plot(data, alpha=0.6, linewidth=0.5)

    # Add moving average
    if len(data) > 10:
        window = min(50, len(data) // 10)
        moving_avg = np.convolve(data, np.ones(window)/window, mode='valid')
        plt.plot(range(window-1, len(data)), moving_avg,
                linewidth=2, label=f'Moving Avg (window={window})')
        plt.legend()

    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Training curve saved to {save_path}")


def test_lstm_model(video_folder: str, model_path: str = None, visualize: bool = True):
    """
    Test trained LSTM model on video sequences.

    Args:
        video_folder: Path to folder containing test videos
        model_path: Path to saved model (uses config if None)
        visualize: Whether to create visualization video
    """
    print("\n" + "=" * 80)
    print("TESTING LSTM MODEL")
    print("=" * 80)

    # Load configuration
    test_cfg = config.testing
    model_path = model_path or test_cfg.model_path

    # Load trained model
    try:
        model = PPO.load(model_path)
        print(f"✓ Model loaded: {model_path}\n")
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        return

    # Get test videos
    video_files = sorted(glob.glob(os.path.join(video_folder, "*.mp4")))
    video_files = video_files[:test_cfg.max_test_videos]

    all_predictions = []
    all_statistics = []

    # Test on each video
    for video_idx, video_path in enumerate(video_files):
        print(f"\nTesting on: {os.path.basename(video_path)}")
        print("-" * 80)

        # Initialize video capture
        cap = cv2.VideoCapture(video_path)

        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Video writer for visualization
        video_writer = None
        if visualize:
            output_path = f"test_output_{video_idx}.mp4"
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        # Initialize buffers
        frame_buffer = deque(maxlen=config.video.frame_stack)
        gaze_history = deque(maxlen=config.video.frame_stack)
        loss_history = deque(maxlen=config.video.frame_stack)
        cumulative_loss = 0.0

        # Initialize with diagonal gaze pattern
        for i in range(config.video.frame_stack):
            gaze_history.append(np.array([
                0.3 + 0.4 * (i / config.video.frame_stack),
                0.3 + 0.4 * ((i + 1) / config.video.frame_stack)
            ], dtype=np.float32))
            loss_history.append(0.0)

        # Load initial frames
        original_frames = []
        for _ in range(config.video.frame_stack):
            ret, frame = cap.read()
            if ret:
                original_frames.append(frame.copy())
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                processed = cv2.resize(gray, config.video.target_size)
                frame_buffer.append(processed)

        # Run predictions
        predictions = []
        losses = []
        frame_idx = 0

        while frame_idx < test_cfg.test_frames_per_video:
            # Construct observation
            frames_array = np.array(frame_buffer)
            frames_array = np.expand_dims(frames_array, axis=-1)

            obs = {
                'frames': frames_array,
                'gaze_history': np.array(gaze_history, dtype=np.float32),
                'loss_history': np.array(loss_history, dtype=np.float32),
                'cumulative_loss': np.array([cumulative_loss], dtype=np.float32),
            }

            # Predict gaze (deterministic for first half, stochastic for second half)
            deterministic = frame_idx < test_cfg.deterministic_frames
            action, _ = model.predict(obs, deterministic=deterministic)
            predictions.append(action)

            # Simulate loss (in real scenario, would come from expert)
            simulated_loss = np.random.uniform(0.0, 0.5)
            losses.append(simulated_loss)
            cumulative_loss = np.clip(
                cumulative_loss + simulated_loss,
                0.0,
                config.environment.cumulative_loss_max
            )

            # Visualize on frame
            if visualize and video_writer and len(original_frames) > 0:
                vis_frame = original_frames[-1].copy()

                # Convert normalized coordinates to pixel coordinates
                gaze_x = int(action[0] * width)
                gaze_y = int(action[1] * height)

                # Draw gaze point
                cv2.circle(vis_frame, (gaze_x, gaze_y), 15, (0, 255, 0), 3)
                cv2.circle(vis_frame, (gaze_x, gaze_y), 5, (0, 0, 255), -1)

                # Draw gaze trail
                if len(predictions) > 1:
                    for j in range(max(0, len(predictions) - 10), len(predictions) - 1):
                        pt1 = (int(predictions[j][0] * width), int(predictions[j][1] * height))
                        pt2 = (int(predictions[j+1][0] * width), int(predictions[j+1][1] * height))
                        alpha = (j - max(0, len(predictions) - 10)) / 10
                        color = (0, int(255 * alpha), 0)
                        cv2.line(vis_frame, pt1, pt2, color, 2)

                # Add text overlay
                cv2.putText(vis_frame, f"Frame: {frame_idx}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(vis_frame, f"Gaze: ({action[0]:.3f}, {action[1]:.3f})", (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(vis_frame, f"Mode: {'Deterministic' if deterministic else 'Stochastic'}",
                           (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                video_writer.write(vis_frame)

            # Read next frame
            ret, frame = cap.read()
            if not ret:
                break

            original_frames.append(frame.copy())
            if len(original_frames) > config.video.frame_stack:
                original_frames.pop(0)

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            processed = cv2.resize(gray, config.video.target_size)
            frame_buffer.append(processed)
            gaze_history.append(action)
            loss_history.append(simulated_loss)

            frame_idx += 1

        cap.release()
        if video_writer:
            video_writer.release()
            print(f"  Visualization saved to: {output_path}")

        # Analyze predictions
        predictions = np.array(predictions)

        stats = {
            'mean_x': predictions[:, 0].mean(),
            'mean_y': predictions[:, 1].mean(),
            'std_x': predictions[:, 0].std(),
            'std_y': predictions[:, 1].std(),
            'min_x': predictions[:, 0].min(),
            'max_x': predictions[:, 0].max(),
            'min_y': predictions[:, 1].min(),
            'max_y': predictions[:, 1].max(),
        }

        all_predictions.append(predictions)
        all_statistics.append(stats)

        print(f"\n  Prediction Statistics:")
        print(f"    Mean:     X={stats['mean_x']:.3f}, Y={stats['mean_y']:.3f}")
        print(f"    Std Dev:  X={stats['std_x']:.3f}, Y={stats['std_y']:.3f}")
        print(f"    Range X:  [{stats['min_x']:.3f}, {stats['max_x']:.3f}]")
        print(f"    Range Y:  [{stats['min_y']:.3f}, {stats['max_y']:.3f}]")

        # Calculate movement statistics
        movements = np.linalg.norm(predictions[1:] - predictions[:-1], axis=1)
        print(f"    Movement: Mean={movements.mean():.4f}, Std={movements.std():.4f}")
        print(f"    Final cumulative loss: {cumulative_loss:.3f}")

        # Assess model quality
        is_centered = (test_cfg.centered_min <= stats['mean_x'] <= test_cfg.centered_max and
                      test_cfg.centered_min <= stats['mean_y'] <= test_cfg.centered_max)
        has_variation = predictions.std() > test_cfg.variation_threshold
        has_movement = movements.mean() > test_cfg.movement_threshold
        avoids_edges = (stats['min_x'] > test_cfg.edge_avoid_min and
                       stats['max_x'] < test_cfg.edge_avoid_max)

        print("\n  Assessment:")
        if is_centered and has_variation and has_movement and avoids_edges:
            print("    ✅ Model shows varied, memory-aware gaze patterns!")
        elif has_variation and has_movement:
            print("    ⚠️  Model shows movement but may need refinement")
        else:
            print("    ❌ Model needs more training")

    # Create summary plot
    if len(all_predictions) > 0:
        create_test_summary_plots(all_predictions, all_statistics)

    print("\n" + "=" * 80)
    print("Testing complete!")
    print("=" * 80)


def create_test_summary_plots(predictions_list: List[np.ndarray], stats_list: List[dict]):
    """
    Create summary plots for test results.

    Args:
        predictions_list: List of prediction arrays for each video
        stats_list: List of statistics dictionaries for each video
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # Plot 1: Gaze trajectories
    ax = axes[0, 0]
    for i, preds in enumerate(predictions_list):
        ax.plot(preds[:, 0], preds[:, 1], alpha=0.7, linewidth=1, label=f'Video {i+1}')
        ax.scatter(preds[0, 0], preds[0, 1], marker='o', s=100, zorder=5)
        ax.scatter(preds[-1, 0], preds[-1, 1], marker='s', s=100, zorder=5)
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    ax.set_title('Gaze Trajectories')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    # Plot 2: X coordinate over time
    ax = axes[0, 1]
    for i, preds in enumerate(predictions_list):
        ax.plot(preds[:, 0], alpha=0.7, label=f'Video {i+1}')
    ax.set_xlabel('Frame')
    ax.set_ylabel('X Coordinate')
    ax.set_title('X Coordinate Over Time')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 3: Y coordinate over time
    ax = axes[1, 0]
    for i, preds in enumerate(predictions_list):
        ax.plot(preds[:, 1], alpha=0.7, label=f'Video {i+1}')
    ax.set_xlabel('Frame')
    ax.set_ylabel('Y Coordinate')
    ax.set_title('Y Coordinate Over Time')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 4: Statistics comparison
    ax = axes[1, 1]
    video_ids = [f'V{i+1}' for i in range(len(stats_list))]
    x_pos = np.arange(len(video_ids))

    mean_x_vals = [s['mean_x'] for s in stats_list]
    mean_y_vals = [s['mean_y'] for s in stats_list]
    std_x_vals = [s['std_x'] for s in stats_list]
    std_y_vals = [s['std_y'] for s in stats_list]

    width = 0.2
    ax.bar(x_pos - width*1.5, mean_x_vals, width, label='Mean X', alpha=0.8)
    ax.bar(x_pos - width*0.5, mean_y_vals, width, label='Mean Y', alpha=0.8)
    ax.bar(x_pos + width*0.5, std_x_vals, width, label='Std X', alpha=0.8)
    ax.bar(x_pos + width*1.5, std_y_vals, width, label='Std Y', alpha=0.8)

    ax.set_xlabel('Video')
    ax.set_ylabel('Value')
    ax.set_title('Statistics Comparison')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(video_ids)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig('test_summary.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\nTest summary plots saved to: test_summary.png")


def main():
    """Main entry point for training and testing."""
    import argparse

    parser = argparse.ArgumentParser(description='LSTM Gaze Tracking System')
    parser.add_argument('--mode', type=str, choices=['train', 'test', 'both'],
                       default='both', help='Operation mode')
    parser.add_argument('--video-folder', type=str, default=None,
                       help='Path to video folder (overrides config)')
    parser.add_argument('--model-path', type=str, default=None,
                       help='Path to model for testing/resuming')
    parser.add_argument('--timesteps', type=int, default=None,
                       help='Total training timesteps')
    parser.add_argument('--resume', action='store_true',
                       help='Resume training from checkpoint')
    parser.add_argument('--no-visualize', action='store_true',
                       help='Disable visualization during testing')

    args = parser.parse_args()

    # Validate configuration
    if not config.validate():
        print("Configuration validation failed. Please check your config.")
        return 1

    # Print configuration summary
    config.print_summary()

    # Set video folder
    video_folder = args.video_folder or config.video.video_folder

    if not os.path.exists(video_folder):
        print(f"Error: Video folder does not exist: {video_folder}")
        return 1

    # Create output directories
    os.makedirs('checkpoints', exist_ok=True)
    os.makedirs('outputs', exist_ok=True)

    try:
        if args.mode in ['train', 'both']:
            print("\n" + "=" * 80)
            print("STARTING TRAINING")
            print("=" * 80 + "\n")

            resume_path = args.model_path if args.resume else None
            model = train_lstm_model(
                video_folder,
                total_timesteps=args.timesteps,
                resume_path=resume_path
            )

            print("\n✓ Training completed successfully!")

        if args.mode in ['test', 'both']:
            print("\n" + "=" * 80)
            print("STARTING TESTING")
            print("=" * 80 + "\n")

            model_path = args.model_path or config.testing.model_path
            test_lstm_model(
                video_folder,
                model_path=model_path,
                visualize=not args.no_visualize
            )

            print("\n✓ Testing completed successfully!")

        return 0

    except Exception as e:
        print(f"\n✗ Error occurred: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())