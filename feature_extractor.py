"""Feature extractor combining CNN and LSTM for gaze prediction.

This module implements the neural network architecture that processes
video frames with CNN and tracks temporal information with LSTM.

Author: Sudhanshu Anand
Date: 2025-10-30
"""

import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from gymnasium import spaces
from typing import Dict


class CNNLSTMGazeExtractor(BaseFeaturesExtractor):
    """Feature extractor combining CNN for spatial and LSTM for temporal processing.

    Architecture:
        1. CNN processes stacked frames to extract spatial features
        2. LSTM tracks gaze history for temporal patterns
        3. Features are combined and passed through FC layers

    Args:
        observation_space: The observation space of the environment.
        config: Configuration dictionary from YAML.
        features_dim: Output dimension of the feature extractor.
    """

    def __init__(self,
                 observation_space: spaces.Dict,
                 features_dim: int = 256,
                 config: Dict = None):
        """Initializes the feature extractor.

        Args:
            observation_space: Gym observation space.
            features_dim: Output feature dimension.
            config: Configuration dictionary (must be provided as kwarg).
        """
        super(CNNLSTMGazeExtractor, self).__init__(observation_space, features_dim)

        if config is None:
            raise ValueError("config must be provided to CNNLSTMGazeExtractor")

        self.config = config
        cnn_cfg = config['cnn']
        lstm_cfg = config['lstm']
        policy_cfg = config['policy']

        # Get input dimensions
        frame_stack = config['video']['frame_stack']
        channels = 1 if config['video']['grayscale'] else 3

        # CNN for spatial feature extraction
        self.cnn = nn.Sequential(
            # Conv1
            nn.Conv2d(
                frame_stack * channels,
                cnn_cfg['conv1_filters'],
                kernel_size=cnn_cfg['conv1_kernel'],
                stride=cnn_cfg['conv1_stride']
            ),
            nn.ReLU(),

            # Conv2
            nn.Conv2d(
                cnn_cfg['conv1_filters'],
                cnn_cfg['conv2_filters'],
                kernel_size=cnn_cfg['conv2_kernel'],
                stride=cnn_cfg['conv2_stride']
            ),
            nn.ReLU(),

            # Conv3
            nn.Conv2d(
                cnn_cfg['conv2_filters'],
                cnn_cfg['conv3_filters'],
                kernel_size=cnn_cfg['conv3_kernel'],
                stride=cnn_cfg['conv3_stride']
            ),
            nn.ReLU(),

            # Conv4
            nn.Conv2d(
                cnn_cfg['conv3_filters'],
                cnn_cfg['conv4_filters'],
                kernel_size=cnn_cfg['conv4_kernel'],
                stride=cnn_cfg['conv4_stride']
            ),
            nn.ReLU(),
        )

        # Adaptive pooling to fixed size
        if cnn_cfg['pooling'] == 'adaptive':
            self.pool = nn.AdaptiveAvgPool2d(tuple(cnn_cfg['adaptive_pool_size']))
            cnn_output_dim = cnn_cfg['conv4_filters']
        else:
            self.pool = nn.Identity()
            # Calculate output size (would need to compute based on input size)
            cnn_output_dim = cnn_cfg['conv4_filters']

        # LSTM for temporal feature extraction from gaze history
        # Input: (batch, seq_len, 2) where 2 is (x, y)
        self.gaze_lstm = nn.LSTM(
            input_size=2,  # x, y coordinates
            hidden_size=lstm_cfg['hidden_size'],
            num_layers=lstm_cfg['num_layers'],
            batch_first=True,
            dropout=lstm_cfg['dropout'] if lstm_cfg['num_layers'] > 1 else 0,
            bidirectional=lstm_cfg['bidirectional']
        )

        # LSTM output size
        lstm_output_dim = lstm_cfg['hidden_size']
        if lstm_cfg['bidirectional']:
            lstm_output_dim *= 2

        # Frame index embedding
        self.frame_index_fc = nn.Sequential(
            nn.Linear(1, 32),
            nn.ReLU(),
            nn.Linear(32, 32)
        )

        # Combine all features
        combined_dim = cnn_output_dim + lstm_output_dim + 32

        # Final FC layers
        hidden_layers = policy_cfg['shared_layers']
        layers = []

        in_dim = combined_dim
        for hidden_dim in hidden_layers:
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(policy_cfg['dropout'])
            ])
            in_dim = hidden_dim

        # Final output layer
        layers.append(nn.Linear(in_dim, features_dim))
        layers.append(nn.ReLU())

        self.combined_fc = nn.Sequential(*layers)

        self._features_dim = features_dim
        self.grayscale = config['video']['grayscale']

        print(f"CNNLSTMGazeExtractor initialized")
        print(f"  CNN input channels: {frame_stack * channels}")
        print(f"  CNN output dim: {cnn_output_dim}")
        print(f"  LSTM output dim: {lstm_output_dim}")
        print(f"  Combined dim: {combined_dim}")
        print(f"  Features dim: {features_dim}")
        print(f"  Grayscale: {self.grayscale}")

    def forward(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Forward pass through the network.

        Args:
            observations: Dictionary with keys:
                - 'frames': (batch, frame_stack, H, W) for grayscale
                           OR (batch, frame_stack, C, H, W) for RGB
                - 'gaze_history': (batch, frame_stack, 2)
                - 'frame_index_normalized': (batch, 1)

        Returns:
            Feature tensor of shape (batch, features_dim).
        """
        frames = observations['frames']
        gaze_history = observations['gaze_history']
        frame_idx = observations['frame_index_normalized']

        batch_size = frames.shape[0]

        # Process frames with CNN
        # Normalize to [0, 1]
        frames = frames.float() / 255.0

        # Handle different input shapes
        if self.grayscale:
            # Input: (batch, frame_stack, H, W)
            # Need: (batch, frame_stack, H, W) for CNN
            # CNN expects (batch, channels, H, W) where channels = frame_stack
            # frames is already in correct format!
            pass
        else:
            # Input: (batch, frame_stack, C, H, W)
            # Need: (batch, frame_stack*C, H, W)
            frame_stack = frames.shape[1]
            channels = frames.shape[2]
            height = frames.shape[3]
            width = frames.shape[4]

            # Reshape to merge frame_stack and channels
            frames = frames.reshape(batch_size, frame_stack * channels, height, width)

        # CNN forward
        cnn_features = self.cnn(frames)
        cnn_features = self.pool(cnn_features)
        cnn_features = cnn_features.view(batch_size, -1)  # Flatten

        # Process gaze history with LSTM
        # gaze_history: (batch, seq_len, 2)
        lstm_out, (h_n, c_n) = self.gaze_lstm(gaze_history)

        # Take the last hidden state
        # h_n shape: (num_layers * num_directions, batch, hidden_size)
        if self.gaze_lstm.bidirectional:
            # Concatenate forward and backward
            lstm_features = torch.cat([h_n[-2], h_n[-1]], dim=1)
        else:
            lstm_features = h_n[-1]  # Last layer

        # Process frame index
        frame_features = self.frame_index_fc(frame_idx)

        # Combine all features
        combined = torch.cat([cnn_features, lstm_features, frame_features], dim=1)

        # Final FC layers
        features = self.combined_fc(combined)

        return features


if __name__ == "__main__":
    # Test the feature extractor
    import yaml
    import numpy as np
    from gymnasium import spaces

    # Load config
    with open("config.yaml", 'r') as f:
        config = yaml.safe_load(f)

    # Create dummy observation space
    frame_stack = config['video']['frame_stack']
    height = config['video']['target_height']
    width = config['video']['target_width']
    grayscale = config['video']['grayscale']

    # Use the correct shape based on grayscale setting
    if grayscale:
        frames_shape = (frame_stack, height, width)
    else:
        frames_shape = (frame_stack, 3, height, width)

    obs_space = spaces.Dict({
        'frames': spaces.Box(
            low=0, high=255,
            shape=frames_shape,
            dtype=np.uint8
        ),
        'gaze_history': spaces.Box(
            low=0.0, high=1.0,
            shape=(frame_stack, 2),
            dtype=np.float32
        ),
        'frame_index_normalized': spaces.Box(
            low=0.0, high=1.0,
            shape=(1,),
            dtype=np.float32
        )
    })

    # Create feature extractor
    extractor = CNNLSTMGazeExtractor(obs_space, features_dim=256, config=config)

    print("\n" + "=" * 80)
    print("TESTING FEATURE EXTRACTOR")
    print("=" * 80 + "\n")

    # Create dummy batch
    batch_size = 4

    if grayscale:
        dummy_frames = torch.randint(0, 255, (batch_size, frame_stack, height, width), dtype=torch.uint8)
    else:
        dummy_frames = torch.randint(0, 255, (batch_size, frame_stack, 3, height, width), dtype=torch.uint8)

    dummy_obs = {
        'frames': dummy_frames,
        'gaze_history': torch.rand(batch_size, frame_stack, 2, dtype=torch.float32),
        'frame_index_normalized': torch.rand(batch_size, 1, dtype=torch.float32)
    }

    print("Input shapes:")
    for key, val in dummy_obs.items():
        print(f"  {key}: {val.shape}")

    # Forward pass
    features = extractor(dummy_obs)

    print(f"\nOutput shape: {features.shape}")
    print(f"Expected: ({batch_size}, 256)")

    print("\n" + "=" * 80)
    print("✓ Feature extractor test completed!")
    print("=" * 80 + "\n")