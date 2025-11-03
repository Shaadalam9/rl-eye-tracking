"""Training pipeline for WebGazer gaze prediction using PPO.

This script implements the complete training workflow:
1. Load JSON files and extract gaze data
2. Split into train/test sets
3. Create RL environment
4. Train PPO agent with CNN+LSTM architecture
5. Save model and results

Author: Sudhanshu Anand
Date: 2025-10-30
"""

import os
import numpy as np
import torch
from datetime import datetime
from typing import Dict
import json
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import (
    BaseCallback, CheckpointCallback, EvalCallback
)
from data_loader import WebGazerDataLoader, load_config
from gaze_environment import create_env
from feature_extractor import CNNLSTMGazeExtractor
from custom_logger import CustomLogger           # structured logging
from logmod import logs
import common

logs(show_level=common.get_configs("logger_level"), show_color=True)
logger = CustomLogger(__name__)


class TrainingLogger(BaseCallback):
    """Custom callback for logging training metrics.

    Logs:
        - Learning rate
        - Episode rewards and lengths
        - Custom gaze metrics
    """

    def __init__(self, config: Dict, verbose: int = 1):
        """Initializes the training logger.

        Args:
            config: Configuration dictionary.
            verbose: Verbosity level.
        """
        super(TrainingLogger, self).__init__(verbose)
        self.config = config
        self.log_frequency = config['training']['log_frequency']

        # Storage
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_distances = []
        self.step_count = 0

    def _on_step(self) -> bool:
        """Called at each training step.

        Returns:
            True to continue training, False to stop.
        """
        self.step_count += 1

        # Log episode info
        if 'infos' in self.locals:
            for info in self.locals['infos']:
                if 'episode' in info:
                    self.episode_rewards.append(info['episode']['r'])
                    self.episode_lengths.append(info['episode']['l'])

                    if 'mean_distance' in info:
                        self.episode_distances.append(info['mean_distance'])

        # Periodic logging
        if self.step_count % self.log_frequency == 0:
            if len(self.episode_rewards) > 0:
                mean_reward = np.mean(self.episode_rewards[-10:])
                mean_length = np.mean(self.episode_lengths[-10:])

                logger.info(f"\nStep {self.step_count}:")
                logger.info(f"  Mean reward (last 10): {mean_reward:.2f}")
                logger.info(f"  Mean length (last 10): {mean_length:.1f}")

                if len(self.episode_distances) > 0:
                    mean_dist = np.mean(self.episode_distances[-10:])
                    logger.info(f"  Mean distance (last 10): {mean_dist:.4f}")

        return True

    def save_metrics(self, save_path: str):
        """Saves training metrics to JSON.

        Args:
            save_path: Path to save metrics.
        """
        # Convert numpy types to Python native types for JSON serialization
        metrics = {
            'episode_rewards': [float(x) for x in self.episode_rewards],
            'episode_lengths': [int(x) for x in self.episode_lengths],
            'episode_distances': [float(x) for x in self.episode_distances],
            'total_steps': int(self.step_count)
        }

        with open(save_path, 'w') as f:
            json.dump(metrics, f, indent=2)

        logger.info(f"✓ Metrics saved to: {save_path}")


def create_lr_schedule(config: Dict):
    """Creates learning rate schedule based on config.

    Args:
        config: Configuration dictionary.

    Returns:
        Learning rate schedule function or constant value.
    """
    lr_cfg = config['lr_schedule']
    initial_lr = config['ppo']['learning_rate']

    if lr_cfg['type'] == 'constant':
        return initial_lr

    elif lr_cfg['type'] == 'linear':
        def linear_schedule(progress_remaining: float) -> float:
            """Linear decay from initial_lr to min_lr."""
            min_lr = initial_lr * lr_cfg['min_lr_factor']
            return min_lr + (initial_lr - min_lr) * progress_remaining

        return linear_schedule

    elif lr_cfg['type'] == 'exponential':
        decay_rate = lr_cfg['exponential_decay']
        min_lr_factor = lr_cfg['min_lr_factor']

        def exponential_schedule(progress_remaining: float) -> float:
            """Exponential decay from initial_lr to min_lr."""
            progress_made = 1.0 - progress_remaining
            lr = initial_lr * np.exp(-decay_rate * progress_made)
            min_lr = initial_lr * min_lr_factor
            return max(lr, min_lr)

        return exponential_schedule

    else:
        logger.info(f"Unknown LR schedule type: {lr_cfg['type']}, using constant")
        return initial_lr


def setup_training(config: Dict) -> Dict:
    """Sets up training environment and data.

    Args:
        config: Configuration dictionary.

    Returns:
        Dictionary with training components.
    """
    logger.info("\n" + "="*80)
    logger.info("SETUP TRAINING")
    logger.info("="*80 + "\n")

    # Set random seeds
    if config['reproducibility']['set_seed']:
        seed = config['reproducibility']['seed']
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
        logger.info(f"✓ Random seed set to: {seed}")

    # Create output directory
    output_dir = config['data']['output_folder']
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"✓ Output directory: {output_dir}")

    # Load data
    logger.info("\nLoading data...")
    loader = WebGazerDataLoader(
        json_folder=config['data']['json_folder'],
        video_folder=config['data']['video_folder'],
        config=config
    )

    video_gaze_map = loader.load_all_mappings()

    # Verify videos
    available_videos, missing_videos = loader.verify_video_files(video_gaze_map)

    if len(available_videos) == 0:
        raise ValueError("No videos found! Check your video folder.")

    # Split train/test
    train_videos, test_videos = loader.split_train_test(available_videos, video_gaze_map)

    # Save split
    split_path = os.path.join(output_dir, "data_split.yaml")
    loader.save_split(train_videos, test_videos, video_gaze_map, split_path)

    # Get full video paths
    video_folder = config['data']['video_folder']
    train_paths = [os.path.join(video_folder, v) for v in train_videos]
    test_paths = [os.path.join(video_folder, v) for v in test_videos]

    return {
        'loader': loader,
        'video_gaze_map': video_gaze_map,
        'train_videos': train_videos,
        'test_videos': test_videos,
        'train_paths': train_paths,
        'test_paths': test_paths,
        'output_dir': output_dir
    }


def create_ppo_model(env, config: Dict, output_dir: str):
    """Creates PPO model with custom feature extractor.

    Args:
        env: Vectorized environment.
        config: Configuration dictionary.
        output_dir: Directory for saving model.

    Returns:
        PPO model instance.
    """
    logger.info("\n" + "="*80)
    logger.info("CREATING PPO MODEL")
    logger.info("="*80 + "\n")

    policy_kwargs = dict(
        features_extractor_class=CNNLSTMGazeExtractor,
        features_extractor_kwargs=dict(
            config=config,
            features_dim=config['policy']['features_dim']
        ),
        net_arch=dict(
            pi=config['policy']['hidden_layers'],
            vf=config['policy']['hidden_layers']
        )
    )

    # Learning rate schedule
    lr_schedule = create_lr_schedule(config)

    # PPO config
    ppo_cfg = config['ppo']

    # Device
    device = config['training']['device']
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    logger.info(f"Device: {device}")
    logger.info(f"Initial learning rate: {ppo_cfg['learning_rate']}")
    logger.info(f"Total timesteps: {config['training']['total_timesteps']:,}")

    # Create model
    model = PPO(
        policy="MultiInputPolicy",
        env=env,
        learning_rate=lr_schedule,
        n_steps=ppo_cfg['n_steps'],
        batch_size=ppo_cfg['batch_size'],
        n_epochs=ppo_cfg['n_epochs'],
        gamma=ppo_cfg['gamma'],
        gae_lambda=ppo_cfg['gae_lambda'],
        clip_range=ppo_cfg['clip_range'],
        clip_range_vf=ppo_cfg['clip_range_vf'],
        ent_coef=ppo_cfg['ent_coef'],
        vf_coef=ppo_cfg['vf_coef'],
        max_grad_norm=ppo_cfg['max_grad_norm'],
        target_kl=ppo_cfg['target_kl'],
        policy_kwargs=policy_kwargs,
        verbose=config['logging']['verbose'],
        device=device,
        tensorboard_log=config['logging']['tensorboard_log'] if config['logging']['use_tensorboard'] else None
    )

    logger.info("✓ PPO model created successfully")

    return model


def train_model(setup_data: Dict, config: Dict):
    """Trains the PPO model.

    Args:
        setup_data: Dictionary from setup_training().
        config: Configuration dictionary.

    Returns:
        Trained PPO model.
    """
    logger.info("\n" + "="*80)
    logger.info("STARTING TRAINING")
    logger.info("="*80 + "\n")

    # Create training environment
    def make_env():
        return create_env(
            setup_data['train_paths'],
            setup_data['video_gaze_map'],
            config
        )

    env = DummyVecEnv([make_env])

    # Create model
    model = create_ppo_model(env, config, setup_data['output_dir'])

    # Setup callbacks
    callbacks = []

    # Checkpoint callback
    checkpoint_dir = os.path.join(setup_data['output_dir'], "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    checkpoint_callback = CheckpointCallback(
        save_freq=config['training']['save_frequency'],
        save_path=checkpoint_dir,
        name_prefix="gaze_model"
    )
    callbacks.append(checkpoint_callback)

    # Training logger
    training_logger = TrainingLogger(config, verbose=1)
    callbacks.append(training_logger)

    # Evaluation callback (optional)
    if config['training']['eval_frequency'] > 0 and len(setup_data['test_paths']) > 0:
        # Create eval environment
        def make_eval_env():
            return create_env(
                setup_data['test_paths'][:1],  # Use first test video
                setup_data['video_gaze_map'],
                config
            )

        eval_env = DummyVecEnv([make_eval_env])

        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=setup_data['output_dir'],
            log_path=setup_data['output_dir'],
            eval_freq=config['training']['eval_frequency'],
            n_eval_episodes=config['training']['eval_episodes'],
            deterministic=True
        )
        callbacks.append(eval_callback)

    # Train
    timestamp_start = datetime.now()
    logger.info(f"Training started at: {timestamp_start}")
    logger.info(f"Total timesteps: {config['training']['total_timesteps']:,}\n")

    try:
        model.learn(
            total_timesteps=config['training']['total_timesteps'],
            callback=callbacks,
            progress_bar=True
        )
    except KeyboardInterrupt:
        logger.error("\n\nTraining interrupted by user!")

    timestamp_end = datetime.now()
    duration = timestamp_end - timestamp_start

    logger.info(f"\n{'='*80}")
    logger.info("TRAINING COMPLETED")
    logger.info(f"{'='*80}")
    logger.info(f"Duration: {duration}")
    logger.info(f"{'='*80}\n")

    # Save final model
    final_model_path = os.path.join(setup_data['output_dir'], "final_model")
    model.save(final_model_path)
    logger.info(f"✓ Final model saved to: {final_model_path}")

    # Save training metrics
    metrics_path = os.path.join(setup_data['output_dir'], "training_metrics.json")
    training_logger.save_metrics(metrics_path)

    # Close environments
    env.close()
    if 'eval_env' in locals():
        eval_env.close()

    return model


def main():
    """Main training function."""
    logger.info("\n" + "="*80)
    logger.info("WEBGAZER GAZE PREDICTION TRAINING")
    logger.info("="*80)
    logger.info("Using PPO + LSTM + CNN")
    logger.info("="*80 + "\n")

    # Load configuration
    config = load_config("config.yaml")

    # Print configuration summary
    logger.info("Configuration:")
    logger.info(f"  JSON folder: {config['data']['json_folder']}")
    logger.info(f"  Video folder: {config['data']['video_folder']}")
    logger.info(f"  Output folder: {config['data']['output_folder']}")
    logger.info(f"  Total timesteps: {config['training']['total_timesteps']:,}")
    logger.info(f"  Frame stack: {config['video']['frame_stack']}")
    logger.info(f"  Frame size: {config['video']['target_width']}x{config['video']['target_height']}")

    # Setup training
    setup_data = setup_training(config)

    # Train model
    model = train_model(setup_data, config)

    logger.info("\n" + "="*80)
    logger.info("✓ TRAINING PIPELINE COMPLETED SUCCESSFULLY!")
    logger.info("="*80)
    logger.info(f"\nResults saved in: {setup_data['output_dir']}")
    logger.info("\nNext steps:")
    logger.info("1. Run test.py to evaluate on test videos")
    logger.info("2. Check tensorboard logs:")
    logger.info(f"tensorboard --logdir={config['logging']['tensorboard_log']}")
    logger.info("="*80 + "\n")


if __name__ == "__main__":
    main()
