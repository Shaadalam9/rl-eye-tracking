"""Gymnasium environment for gaze prediction using WebGazer data.

This environment loads videos and their associated gaze coordinates from
the WebGazer JSON files and trains an RL agent to predict human gaze.

Author: Sudhanshu Anand
Date: 2025-10-30
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import cv2
import os
from typing import Dict, Tuple, List, Optional
from collections import deque
from custom_logger import CustomLogger           # structured logging
from logmod import logs
import common
from data_loader import WebGazerDataLoader, load_config
import yaml

logs(show_level=common.get_configs("logger_level"), show_color=True)
logger = CustomLogger(__name__)


class WebGazerGazeEnv(gym.Env):
    """Gymnasium environment for gaze prediction.

    The agent observes video frames and past gaze history, and must predict
    the next gaze coordinates. The environment rewards the agent based on
    how close the prediction is to the actual human gaze.

    Attributes:
        video_files: List of video file paths.
        video_gaze_map: Dictionary mapping video names to gaze coordinates.
        config: Configuration dictionary.

    Observation space:
        - frames: Stack of recent video frames
        - gaze_history: Recent gaze coordinates
        - frame_index: Current frame number (normalized)

    Action space:
        - Continuous (x, y) coordinates in [0, 1] range
    """

    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(self,
                 video_files: List[str],
                 video_gaze_map: Dict[str, np.ndarray],
                 config: Dict,
                 render_mode: Optional[str] = None):
        """Initializes the gaze prediction environment.

        Args:
            video_files: List of video file paths to use.
            video_gaze_map: Dictionary mapping video filenames to gaze arrays.
            config: Configuration dictionary from YAML.
            render_mode: Optional render mode ('human' or 'rgb_array').
        """
        super().__init__()

        self.video_files = video_files
        self.video_gaze_map = video_gaze_map
        self.config = config
        self.render_mode = render_mode

        # Video processing parameters
        self.frame_stack = config['video']['frame_stack']
        self.target_width = config['video']['target_width']
        self.target_height = config['video']['target_height']
        self.grayscale = config['video']['grayscale']

        # Define action space: continuous (x, y) in [0, 1]
        self.action_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(2,),
            dtype=np.float32
        )

        # Define observation space
        # FIXED: For grayscale, use (frame_stack, H, W) without channel dimension
        # This matches PyTorch's expected format for Conv2d with batch processing
        if self.grayscale:
            frames_shape = (self.frame_stack, self.target_height, self.target_width)
        else:
            # For RGB, use (frame_stack, 3, H, W) - channels-first
            frames_shape = (self.frame_stack, 3, self.target_height, self.target_width)

        self.observation_space = spaces.Dict({
            'frames': spaces.Box(
                low=0,
                high=255,
                shape=frames_shape,
                dtype=np.uint8
            ),
            'gaze_history': spaces.Box(
                low=0.0,
                high=1.0,
                shape=(self.frame_stack, 2),
                dtype=np.float32
            ),
            'frame_index_normalized': spaces.Box(
                low=0.0,
                high=1.0,
                shape=(1,),
                dtype=np.float32
            )
        })

        # Internal state
        self.current_video_idx = 0
        self.current_frame_idx = 0
        self.cap = None
        self.total_frames = 0

        # Frame and gaze buffers
        self.frame_buffer = deque(maxlen=self.frame_stack)
        self.gaze_history = deque(maxlen=self.frame_stack)

        # Current video's gaze data (will store only x, y)
        self.current_gazes = None

        # Statistics
        self.episode_rewards = []
        self.episode_distances = []

        logger.info("WebGazerGazeEnv initialized")
        logger.info(f"Videos: {len(self.video_files)}")
        logger.info(f"Frame stack: {self.frame_stack}")
        logger.info(f"Frame size: {self.target_width}x{self.target_height}")
        logger.info(f"Grayscale: {self.grayscale}")
        logger.info(f"Frames shape: {frames_shape}")

    def _load_video(self, video_idx: int):
        """Loads a video and its associated gaze data.

        Args:
            video_idx: Index into self.video_files.
        """
        # Release previous video
        if self.cap is not None:
            self.cap.release()

        # Get video path
        video_path = self.video_files[video_idx]
        video_name = os.path.basename(video_path)

        # Load video
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")

        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Load gaze data
        if video_name in self.video_gaze_map:
            raw_gazes = self.video_gaze_map[video_name]

            # CRITICAL FIX: Extract only x, y coordinates (ignore timestamp t)
            # Data loader returns (x, y, t) tuples or (N, 3) array
            if isinstance(raw_gazes, np.ndarray):
                if raw_gazes.ndim == 2 and raw_gazes.shape[1] >= 2:
                    # Take only first 2 columns (x, y)
                    self.current_gazes = raw_gazes[:, :2].astype(np.float32)
                elif raw_gazes.ndim == 1:
                    # Single gaze point
                    self.current_gazes = raw_gazes[:2].reshape(1, 2).astype(np.float32)
                else:
                    self.current_gazes = raw_gazes.astype(np.float32)
            else:
                # List of tuples: [(x, y, t), ...]
                self.current_gazes = np.array(
                    [[gaze[0], gaze[1]] for gaze in raw_gazes],
                    dtype=np.float32
                )

            # Ensure shape is (N, 2)
            if self.current_gazes.ndim == 1:
                self.current_gazes = self.current_gazes.reshape(-1, 2)

            # Validate coordinates are in [0, 1] range
            if self.current_gazes.shape[1] != 2:
                raise ValueError(f"Gaze data must have shape (N, 2), got {self.current_gazes.shape}")

            # Clip to valid range
            self.current_gazes = np.clip(self.current_gazes, 0.0, 1.0)

        else:
            raise ValueError(f"No gaze data for video: {video_name}")

        logger.info(f"Loaded video: {video_name}")
        logger.info(f"Total frames: {self.total_frames}")
        logger.info(f"Gaze points: {len(self.current_gazes)} (shape: {self.current_gazes.shape})")

    def _preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """Preprocesses a video frame.

        Args:
            frame: Raw BGR frame from cv2.

        Returns:
            Preprocessed frame in correct shape.
        """
        # Resize first
        frame = cv2.resize(frame, (self.target_width, self.target_height))

        # Convert to grayscale if needed
        if self.grayscale:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # Shape is now (H, W) - no channel dimension for grayscale
        else:
            # Convert BGR to RGB and transpose to channels-first
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = np.transpose(frame, (2, 0, 1))  # (H, W, C) -> (C, H, W)

        return frame.astype(np.uint8)

    def _get_expert_gaze(self, frame_idx: int) -> np.ndarray:
        """Gets the expert gaze for current frame.

        Since we have sparse gaze points, we interpolate or use nearest.

        Args:
            frame_idx: Current frame index.

        Returns:
            Expert gaze (x, y) in [0, 1] range as shape (2,).
        """
        if len(self.current_gazes) == 0:
            # No gaze data, return center
            return np.array([0.5, 0.5], dtype=np.float32)

        # Simple strategy: Use modulo to cycle through gaze points
        # This assumes gazes are sequential
        gaze_idx = min(frame_idx, len(self.current_gazes) - 1)

        # Return only (x, y), ensure shape is (2,)
        gaze = self.current_gazes[gaze_idx]
        return gaze.copy().flatten()[:2]  # Ensure exactly 2 values

    def _calculate_reward(self, predicted: np.ndarray, expert: np.ndarray) -> float:
        """Calculates reward based on prediction accuracy.

        Args:
            predicted: Predicted gaze (x, y) - shape (2,).
            expert: Expert gaze (x, y) - shape (2,).

        Returns:
            Reward value.
        """
        cfg = self.config['reward']

        # Ensure both are (2,) arrays
        predicted = predicted.flatten()[:2]
        expert = expert.flatten()[:2]

        # Calculate Euclidean distance
        distance = np.linalg.norm(predicted - expert)

        # Base reward: inverse of distance
        reward = (1.0 - distance) * cfg['distance_scale']

        # Bonus for excellent accuracy
        if distance < cfg['excellent_threshold']:
            reward += cfg['excellent_bonus']
        elif distance < cfg['good_threshold']:
            reward += cfg['good_bonus']
        elif distance < cfg['fair_threshold']:
            reward += cfg['fair_bonus']

        # Penalty for edge hugging
        edge_inner = cfg['edge_inner']
        edge_outer = cfg['edge_outer']

        if (predicted[0] < edge_inner or predicted[0] > edge_outer or
                predicted[1] < edge_inner or predicted[1] > edge_outer):
            reward -= cfg['edge_penalty_light']

        # Movement rewards
        if len(self.gaze_history) > 0:
            last_pred = np.array(self.gaze_history[-1], dtype=np.float32).flatten()[:2]
            movement = np.linalg.norm(predicted - last_pred)

            if movement > cfg['movement_good_threshold']:
                reward += cfg['movement_reward']
            elif movement < cfg['stagnation_threshold']:
                reward -= cfg['stagnation_penalty']

        # Temporal consistency: penalize large jumps from expert
        if len(self.gaze_history) > 0 and self.current_frame_idx > 0:
            last_gaze = np.array(self.gaze_history[-1], dtype=np.float32).flatten()[:2]
            last_expert = self._get_expert_gaze(self.current_frame_idx - 1).flatten()[:2]

            # Calculate if prediction is consistent with trajectory
            expert_velocity = expert - last_expert
            pred_velocity = predicted - last_gaze

            velocity_diff = np.linalg.norm(expert_velocity - pred_velocity)

            if velocity_diff < 0.05:
                reward += cfg['consistency_reward']
            elif velocity_diff > 0.2:
                reward -= cfg['jitter_penalty']

        return float(reward)

    def reset(self, seed: Optional[int] = None,
              options: Optional[Dict] = None) -> Tuple[Dict, Dict]:
        """Resets the environment to initial state.

        Args:
            seed: Random seed for reproducibility.
            options: Additional options.

        Returns:
            Tuple of (observation, info).
        """
        super().reset(seed=seed)

        # Load next video (cycle through videos)
        self._load_video(self.current_video_idx)
        self.current_video_idx = (self.current_video_idx + 1) % len(self.video_files)

        # Reset frame index
        self.current_frame_idx = 0

        # Clear buffers
        self.frame_buffer.clear()
        self.gaze_history.clear()

        # Initialize buffers with first frames
        for i in range(self.frame_stack):
            ret, frame = self.cap.read()

            if ret:
                processed = self._preprocess_frame(frame)
                self.frame_buffer.append(processed)

                # Initialize gaze history with expert gazes (ensure (2,) shape)
                expert_gaze = self._get_expert_gaze(i)
                self.gaze_history.append(expert_gaze)
            else:
                # If video is shorter than frame_stack, pad with black frames
                if self.grayscale:
                    black_frame = np.zeros(
                        (self.target_height, self.target_width),
                        dtype=np.uint8
                    )
                else:
                    black_frame = np.zeros(
                        (3, self.target_height, self.target_width),
                        dtype=np.uint8
                    )
                self.frame_buffer.append(black_frame)
                self.gaze_history.append(np.array([0.5, 0.5], dtype=np.float32))

        # Reset statistics
        self.episode_rewards = []
        self.episode_distances = []

        return self._get_observation(), {}

    def _get_observation(self) -> Dict:
        """Constructs observation dictionary.

        Returns:
            Observation dictionary.
        """
        # Stack frames - shape will be (frame_stack, H, W) for grayscale
        # or (frame_stack, C, H, W) for RGB
        frames = np.array(list(self.frame_buffer), dtype=np.uint8)

        # Ensure gaze_history is (frame_stack, 2)
        gaze_hist = []
        for gaze in self.gaze_history:
            gaze_2d = np.array(gaze, dtype=np.float32).flatten()[:2]
            gaze_hist.append(gaze_2d)
        gaze_hist = np.array(gaze_hist, dtype=np.float32)

        # Ensure shape is (frame_stack, 2)
        if gaze_hist.shape[0] < self.frame_stack:
            # Pad if needed
            padding = np.tile([0.5, 0.5], (self.frame_stack - gaze_hist.shape[0], 1))
            gaze_hist = np.vstack([gaze_hist, padding])

        # Normalized frame index
        frame_idx_norm = np.array(
            [self.current_frame_idx / max(self.total_frames, 1)],
            dtype=np.float32
        )

        return {
            'frames': frames,
            'gaze_history': gaze_hist,
            'frame_index_normalized': frame_idx_norm
        }

    def step(self, action: np.ndarray) -> Tuple[Dict, float, bool, bool, Dict]:
        """Executes one step in the environment.

        Args:
            action: Predicted gaze coordinates (x, y) - shape (2,).

        Returns:
            Tuple of (observation, reward, terminated, truncated, info).
        """
        # Clip action to valid range and ensure (2,) shape
        action = np.array(action, dtype=np.float32).flatten()[:2]
        action = np.clip(action, 0.0, 1.0)

        # Get expert gaze for current frame (ensure (2,) shape)
        expert_gaze = self._get_expert_gaze(self.current_frame_idx)

        # Calculate reward
        reward = self._calculate_reward(action, expert_gaze)
        distance = np.linalg.norm(action - expert_gaze)

        # Store statistics
        self.episode_rewards.append(reward)
        self.episode_distances.append(distance)

        # Update gaze history (store as (2,) array)
        self.gaze_history.append(action.copy())

        # Read next frame
        ret, frame = self.cap.read()

        if ret:
            processed = self._preprocess_frame(frame)
            self.frame_buffer.append(processed)
            self.current_frame_idx += 1
        else:
            # Video ended
            terminated = True
            truncated = False

            info = {
                'episode': {
                    'r': np.sum(self.episode_rewards),
                    'l': len(self.episode_rewards)
                },
                'mean_distance': np.mean(self.episode_distances),
                'expert_gaze': expert_gaze,
                'predicted_gaze': action,
                'distance': distance
            }

            return self._get_observation(), reward, terminated, truncated, info

        # Check if we've reached the end
        terminated = self.current_frame_idx >= min(self.total_frames, len(self.current_gazes)) - 1
        truncated = False

        info = {
            'expert_gaze': expert_gaze,
            'predicted_gaze': action,
            'distance': distance,
            'frame_idx': self.current_frame_idx
        }

        return self._get_observation(), reward, terminated, truncated, info

    def render(self):
        """Renders the environment (optional)."""
        if self.render_mode is None:
            return

        # TODO: Implement rendering if needed
        pass

    def close(self):
        """Closes the environment and releases resources."""
        if self.cap is not None:
            self.cap.release()

        if self.render_mode == "human":
            cv2.destroyAllWindows()


def create_env(video_files: List[str],
               video_gaze_map: Dict[str, np.ndarray],
               config: Dict) -> WebGazerGazeEnv:
    """Factory function to create environment.

    Args:
        video_files: List of video file paths.
        video_gaze_map: Dictionary mapping filenames to gaze arrays.
        config: Configuration dictionary.

    Returns:
        WebGazerGazeEnv instance.
    """
    return WebGazerGazeEnv(video_files, video_gaze_map, config)


if __name__ == "__main__":
    # Test the environment

    # Load config
    config = load_config("config.yaml")

    # Load data
    loader = WebGazerDataLoader(
        json_folder=config['data']['json_folder'],
        video_folder=config['data']['video_folder'],
        config=config
    )

    video_gaze_map = loader.load_all_mappings()
    available_videos, _ = loader.verify_video_files(video_gaze_map)

    if len(available_videos) == 0:
        logger.info("No videos available for testing!")
        exit(1)

    # Get full paths
    video_paths = [os.path.join(config['data']['video_folder'], v)
                   for v in available_videos[:1]]  # Test with first video only

    # Create environment
    env = create_env(video_paths, video_gaze_map, config)

    logger.info("\n" + "=" * 80)
    logger.info("TESTING ENVIRONMENT")
    logger.info("=" * 80 + "\n")

    # Test reset
    obs, info = env.reset()
    logger.info(f"Observation space: {env.observation_space}")
    logger.info(f"Action space: {env.action_space}")
    logger.info(f"Frames shape: {obs['frames'].shape}")
    logger.info(f"Gaze history shape: {obs['gaze_history'].shape}")

    # Verify shapes
    logger.info("\nShape verification:")
    logger.info(f"  Expected frames: {env.observation_space['frames'].shape}")
    logger.info(f"  Actual frames: {obs['frames'].shape}")
    logger.info(f"  Expected gaze_history: ({env.frame_stack}, 2)")
    logger.info(f"  Actual gaze_history: {obs['gaze_history'].shape}")

    # Test a few steps
    logger.info("\nTesting 5 steps:")
    for i in range(5):
        # Random action
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)

        logger.info(f"Step {i + 1}:")
        logger.info(f"Action shape: {action.shape}, values: ({action[0]:.4f}, {action[1]:.4f})")
        logger.info(f"Expert shape: {info['expert_gaze'].shape}, values: ({info['expert_gaze'][0]:.4f}, {info['expert_gaze'][1]:.4f})")
        logger.info(f"Distance: {info['distance']:.4f}")
        logger.info(f"Reward: {reward:.4f}")
        logger.info(f"Frames shape: {obs['frames'].shape}")

        if terminated or truncated:
            logger.info("    Episode ended")
            break

    env.close()

    logger.info("\n" + "=" * 80)
    logger.info("✓ Environment test completed!")
    logger.info("=" * 80 + "\n")
