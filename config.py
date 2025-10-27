"""
Configuration file for LSTM-based Gaze Tracking System.

This module contains all configurable parameters for the gaze tracking system,
including environment settings, model architecture, training hyperparameters,
and learning rate schedules.

Author: Sudhanshu Anand
Date: 2025-10-27
"""

import os
from dataclasses import dataclass, field
from typing import Tuple, List


@dataclass
class VideoConfig:
    """Configuration for video processing and dataset."""

    video_folder: str = r"C:\Users\sudha\PycharmProjects\RL Project 1\stimuli"  # Path to video files
    max_videos: int = 5  # Maximum number of videos to load for training
    target_size: Tuple[int, int] = (84, 84)  # Target frame size (width, height)
    frame_stack: int = 4  # Number of frames to stack for temporal information


@dataclass
class EnvironmentConfig:
    """Configuration for the reinforcement learning environment."""

    # Action space bounds (normalized coordinates)
    action_low: float = 0.0
    action_high: float = 1.0
    action_shape: Tuple[int] = (2,)  # (x, y) gaze coordinates

    # Observation space parameters
    cumulative_loss_max: float = 10.0  # Maximum cumulative loss value

    # Salient point detection parameters
    edge_detection_threshold_low: int = 50  # Canny edge detection lower threshold
    edge_detection_threshold_high: int = 150  # Canny edge detection upper threshold
    edge_points_limit: int = 10  # Maximum edge points to consider
    bright_percentile: float = 75.0  # Percentile threshold for bright areas
    bright_points_limit: int = 5  # Maximum bright points to consider
    motion_percentile: float = 80.0  # Percentile threshold for motion detection
    motion_points_limit: int = 3  # Maximum motion points to consider

    # Traditional controller parameters
    salient_point_probability: float = 0.7  # Probability of choosing salient point
    random_movement_std: float = 0.05  # Standard deviation for random movements
    random_gaze_min: float = 0.2  # Minimum random gaze coordinate
    random_gaze_max: float = 0.8  # Maximum random gaze coordinate


@dataclass
class RewardConfig:
    """Configuration for reward function parameters."""

    # Base reward parameters
    base_reward_scale: float = 10.0  # Scale factor for base distance reward
    max_euclidean_distance: float = 1.414  # sqrt(2) - maximum possible distance

    # Accuracy bonus thresholds and rewards
    accuracy_threshold_excellent: float = 0.05  # Threshold for excellent accuracy
    accuracy_bonus_excellent: float = 15.0  # Bonus for excellent accuracy
    accuracy_threshold_good: float = 0.1  # Threshold for good accuracy
    accuracy_bonus_good: float = 8.0  # Bonus for good accuracy
    accuracy_threshold_fair: float = 0.15  # Threshold for fair accuracy
    accuracy_bonus_fair: float = 3.0  # Bonus for fair accuracy

    # Edge prediction penalties
    edge_boundary_inner: float = 0.1  # Inner boundary for edge detection
    edge_boundary_outer: float = 0.9  # Outer boundary for edge detection
    edge_penalty_light: float = 5.0  # Light penalty for being near edges
    edge_penalty_heavy: float = 10.0  # Heavy penalty per edge coordinate

    # Movement rewards/penalties
    movement_threshold_good: float = 0.02  # Threshold for good movement
    movement_reward: float = 2.0  # Reward for good movement
    movement_threshold_stagnant: float = 0.005  # Threshold for stagnant movement
    stagnation_penalty: float = 3.0  # Penalty for lack of movement

    # Salient region bonus
    salient_distance_threshold: float = 0.1  # Distance threshold for salient regions
    salient_region_bonus: float = 4.0  # Bonus for looking at salient regions

    # Loss improvement bonus
    loss_improvement_bonus: float = 5.0  # Bonus for improving upon previous loss


@dataclass
class CNNConfig:
    """Configuration for CNN architecture."""

    # Convolutional layer parameters
    conv1_out_channels: int = 32
    conv1_kernel_size: int = 8
    conv1_stride: int = 4

    conv2_out_channels: int = 64
    conv2_kernel_size: int = 4
    conv2_stride: int = 2

    conv3_out_channels: int = 128
    conv3_kernel_size: int = 3
    conv3_stride: int = 1

    conv4_out_channels: int = 256
    conv4_kernel_size: int = 3
    conv4_stride: int = 1

    # Pooling parameters
    adaptive_pool_output_size: Tuple[int, int] = (6, 6)


@dataclass
class LSTMConfig:
    """Configuration for LSTM architecture."""

    # LSTM parameters
    lstm_hidden_size: int = 128  # Hidden state size
    lstm_num_layers: int = 2  # Number of LSTM layers
    lstm_dropout: float = 0.1  # Dropout between LSTM layers

    # Input/output dimensions
    gaze_dim: int = 2  # Dimensionality of gaze coordinates
    loss_dim: int = 1  # Dimensionality of loss values
    lstm_input_size: int = 3  # gaze (2) + loss (1)

    # Cumulative loss network
    cumulative_loss_hidden_dim: int = 32


@dataclass
class PolicyNetworkConfig:
    """Configuration for policy network architecture."""

    features_dim: int = 512  # Output dimension of feature extractor
    net_arch: List[int] = field(default_factory=lambda: [512, 256, 128])  # Policy network layers

    # Combined network parameters
    combined_fc1_dim: int = 512
    combined_dropout: float = 0.2


@dataclass
class LearningRateConfig:
    """Configuration for learning rate scheduling."""

    # Base learning rate
    initial_lr: float = 3e-4

    # Exponential decay parameters
    exponential_decay_rate: float = 3.0  # Controls decay speed (higher = faster decay)
    exponential_min_lr_factor: float = 0.05  # Minimum LR as fraction of initial (lr_min = initial * factor)

    # Linear decay parameters (if needed for comparison)
    linear_min_lr_factor: float = 0.1

    # Cosine annealing parameters
    cosine_min_lr_factor: float = 0.1

    # Warmup parameters
    warmup_fraction: float = 0.1  # Fraction of training for warmup
    warmup_min_lr_factor: float = 0.1

    # Logging
    lr_log_frequency: int = 1000  # Log learning rate every N steps


@dataclass
class TrainingConfig:
    """Configuration for PPO training hyperparameters."""

    # Total training steps
    total_timesteps: int = 200000

    # PPO hyperparameters
    n_steps: int = 2048  # Number of steps per environment per update
    batch_size: int = 128  # Minibatch size for updates
    n_epochs: int = 10  # Number of epochs when optimizing the surrogate loss
    gamma: float = 0.99  # Discount factor
    gae_lambda: float = 0.95  # GAE lambda parameter
    clip_range: float = 0.2  # Clipping parameter for PPO
    ent_coef: float = 0.15  # Entropy coefficient for exploration
    vf_coef: float = 0.5  # Value function coefficient
    max_grad_norm: float = 0.5  # Maximum norm for gradient clipping

    # Device configuration
    device: str = 'cuda'  # 'cuda' or 'cpu'

    # Model saving
    model_save_path: str = "lstm_gaze_model"
    save_frequency: int = 50000  # Save model every N steps

    # Verbosity
    verbose: int = 1
    verbose_lr: bool = True  # Print learning rate updates


@dataclass
class TestingConfig:
    """Configuration for model testing."""

    model_path: str = "lstm_gaze_model"  # Path to saved model
    max_test_videos: int = 2  # Number of videos to test on
    test_frames_per_video: int = 100  # Number of frames to test per video
    deterministic_frames: int = 50  # Use deterministic policy for first N frames

    # Assessment thresholds
    centered_min: float = 0.3  # Minimum coordinate for centered assessment
    centered_max: float = 0.7  # Maximum coordinate for centered assessment
    variation_threshold: float = 0.05  # Minimum std for variation assessment
    movement_threshold: float = 0.02  # Minimum mean movement for assessment
    edge_avoid_min: float = 0.1  # Minimum coordinate for edge avoidance
    edge_avoid_max: float = 0.9  # Maximum coordinate for edge avoidance


class Config:
    """
    Main configuration class that aggregates all sub-configurations.

    This class provides a centralized access point for all configuration
    parameters used throughout the gaze tracking system.
    """

    def __init__(self):
        """Initialize all configuration sub-components."""
        self.video = VideoConfig()
        self.environment = EnvironmentConfig()
        self.reward = RewardConfig()
        self.cnn = CNNConfig()
        self.lstm = LSTMConfig()
        self.policy = PolicyNetworkConfig()
        self.learning_rate = LearningRateConfig()
        self.training = TrainingConfig()
        self.testing = TestingConfig()

    def validate(self) -> bool:
        """
        Validate configuration parameters.

        Returns:
            bool: True if all parameters are valid, False otherwise.
        """
        # Validate video folder exists
        if not os.path.exists(self.video.video_folder):
            print(f"Warning: Video folder does not exist: {self.video.video_folder}")
            return False

        # Validate numeric ranges
        if self.environment.action_low >= self.environment.action_high:
            print("Error: action_low must be less than action_high")
            return False

        if self.training.batch_size > self.training.n_steps:
            print("Error: batch_size must be <= n_steps")
            return False

        if not (0.0 < self.training.gamma <= 1.0):
            print("Error: gamma must be in (0, 1]")
            return False

        # Validate device
        if self.training.device not in ['cuda', 'cpu']:
            print("Error: device must be 'cuda' or 'cpu'")
            return False

        print("✓ Configuration validated successfully")
        return True

    def print_summary(self):
        """Print a summary of key configuration parameters."""
        print("\n" + "=" * 80)
        print("CONFIGURATION SUMMARY")
        print("=" * 80)

        print("\n[VIDEO & ENVIRONMENT]")
        print(f"  Video folder: {self.video.video_folder}")
        print(f"  Max videos: {self.video.max_videos}")
        print(f"  Frame size: {self.video.target_size}")
        print(f"  Frame stack: {self.video.frame_stack}")

        print("\n[MODEL ARCHITECTURE]")
        print(f"  CNN layers: 4 conv layers -> {self.cnn.conv4_out_channels} channels")
        print(f"  LSTM: {self.lstm.lstm_num_layers} layers, {self.lstm.lstm_hidden_size} hidden units")
        print(f"  Policy network: {self.policy.net_arch}")
        print(f"  Features dim: {self.policy.features_dim}")

        print("\n[TRAINING]")
        print(f"  Total timesteps: {self.training.total_timesteps:,}")
        print(f"  Initial LR: {self.learning_rate.initial_lr:.2e}")
        print(f"  LR decay: exponential (rate={self.learning_rate.exponential_decay_rate})")
        print(f"  Min LR factor: {self.learning_rate.exponential_min_lr_factor}")
        print(f"  Batch size: {self.training.batch_size}")
        print(f"  PPO epochs: {self.training.n_epochs}")
        print(f"  Device: {self.training.device}")

        print("\n[REWARD FUNCTION]")
        print(f"  Base scale: {self.reward.base_reward_scale}")
        print(f"  Accuracy bonuses: {self.reward.accuracy_bonus_excellent}/" +
              f"{self.reward.accuracy_bonus_good}/{self.reward.accuracy_bonus_fair}")
        print(f"  Edge penalty: {self.reward.edge_penalty_heavy}")

        print("=" * 80 + "\n")


# Global configuration instance
config = Config()

if __name__ == "__main__":
    # Example usage and validation
    config.validate()
    config.print_summary()