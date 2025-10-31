"""Testing and evaluation script for trained gaze prediction model.

This script evaluates the trained PPO model on test videos and generates
comprehensive metrics and visualizations.

Author: Sudhanshu Anand
Date: 2025-10-30
"""

import os
import yaml
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
import json
from datetime import datetime

from stable_baselines3 import PPO

from data_loader import WebGazerDataLoader, load_config
from gaze_environment import create_env


class GazeEvaluator:
    """Evaluator for gaze prediction model.

    Computes metrics and generates visualizations for model performance.
    """

    def __init__(self, config: Dict):
        """Initializes the evaluator.

        Args:
            config: Configuration dictionary.
        """
        self.config = config
        self.results = {
            'per_video': [],
            'overall': {}
        }

    def evaluate_video(self,
                       model,
                       video_path: str,
                       video_gaze_map: Dict,
                       num_episodes: int = 1) -> Dict:
        """Evaluates model on a single video.

        Args:
            model: Trained PPO model.
            video_path: Path to video file.
            video_gaze_map: Gaze data mapping.
            num_episodes: Number of episodes to run.

        Returns:
            Dictionary with evaluation metrics.
        """
        video_name = os.path.basename(video_path)
        print(f"\nEvaluating: {video_name}")

        # Create environment for this video
        env = create_env([video_path], video_gaze_map, self.config)

        # Storage
        all_predicted = []
        all_expert = []
        all_distances = []
        all_rewards = []

        for episode in range(num_episodes):
            obs, _ = env.reset()
            done = False

            while not done:
                # Get prediction
                action, _ = model.predict(
                    obs,
                    deterministic=self.config['testing']['deterministic']
                )

                # Step environment
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated

                # Record
                all_predicted.append(info['predicted_gaze'])
                all_expert.append(info['expert_gaze'])
                all_distances.append(info['distance'])
                all_rewards.append(reward)

        env.close()

        # Convert to arrays
        predicted = np.array(all_predicted)
        expert = np.array(all_expert)
        distances = np.array(all_distances)
        rewards = np.array(all_rewards)

        # Compute metrics
        metrics = self._compute_metrics(predicted, expert, distances, rewards)
        metrics['video_name'] = video_name
        metrics['num_frames'] = len(predicted)

        # Print summary
        print(f"  Frames: {metrics['num_frames']}")
        print(f"  Mean distance: {metrics['mean_distance']:.4f}")
        print(f"  Median distance: {metrics['median_distance']:.4f}")
        print(f"  Accuracy @0.1: {metrics['accuracy_0.1']:.1f}%")

        return metrics

    def _compute_metrics(self,
                         predicted: np.ndarray,
                         expert: np.ndarray,
                         distances: np.ndarray,
                         rewards: np.ndarray) -> Dict:
        """Computes comprehensive metrics.

        Args:
            predicted: Predicted gaze coordinates (N, 2).
            expert: Expert gaze coordinates (N, 2).
            distances: Distances between predicted and expert (N,).
            rewards: Rewards received (N,).

        Returns:
            Dictionary of metrics.
        """
        metrics = {}

        # Distance metrics
        metrics['mean_distance'] = float(distances.mean())
        metrics['median_distance'] = float(np.median(distances))
        metrics['std_distance'] = float(distances.std())
        metrics['min_distance'] = float(distances.min())
        metrics['max_distance'] = float(distances.max())

        # Accuracy at thresholds
        for threshold in [0.05, 0.1, 0.15, 0.2]:
            accuracy = 100 * (distances < threshold).mean()
            metrics[f'accuracy_{threshold}'] = float(accuracy)

        # Reward metrics
        metrics['mean_reward'] = float(rewards.mean())
        metrics['total_reward'] = float(rewards.sum())

        # Coordinate-wise metrics
        metrics['mean_x_error'] = float(np.abs(predicted[:, 0] - expert[:, 0]).mean())
        metrics['mean_y_error'] = float(np.abs(predicted[:, 1] - expert[:, 1]).mean())

        # Temporal consistency
        if len(predicted) > 1:
            pred_movement = np.linalg.norm(np.diff(predicted, axis=0), axis=1)
            expert_movement = np.linalg.norm(np.diff(expert, axis=0), axis=1)

            metrics['mean_predicted_movement'] = float(pred_movement.mean())
            metrics['mean_expert_movement'] = float(expert_movement.mean())
            metrics['movement_correlation'] = float(
                np.corrcoef(pred_movement, expert_movement)[0, 1]
            )

        # Edge avoidance
        edge_min = 0.15
        edge_max = 0.85
        in_center = (
                (predicted[:, 0] > edge_min) & (predicted[:, 0] < edge_max) &
                (predicted[:, 1] > edge_min) & (predicted[:, 1] < edge_max)
        )
        metrics['center_percentage'] = float(100 * in_center.mean())

        return metrics

    def evaluate_all(self,
                     model,
                     video_paths: List[str],
                     video_gaze_map: Dict) -> Dict:
        """Evaluates model on all test videos.

        Args:
            model: Trained PPO model.
            video_paths: List of test video paths.
            video_gaze_map: Gaze data mapping.

        Returns:
            Complete evaluation results.
        """
        print("\n" + "=" * 80)
        print("EVALUATING MODEL ON TEST SET")
        print("=" * 80)

        num_episodes = self.config['testing']['num_episodes']

        # Evaluate each video
        for video_path in video_paths:
            metrics = self.evaluate_video(
                model, video_path, video_gaze_map, num_episodes
            )
            self.results['per_video'].append(metrics)

        # Compute overall metrics
        self._compute_overall_metrics()

        return self.results

    def _compute_overall_metrics(self):
        """Computes overall metrics across all videos."""
        per_video = self.results['per_video']

        if len(per_video) == 0:
            return

        overall = {}

        # Average metrics across videos
        metric_keys = [k for k in per_video[0].keys() if k != 'video_name' and k != 'num_frames']

        for key in metric_keys:
            values = [v[key] for v in per_video]
            overall[f'mean_{key}'] = float(np.mean(values))
            overall[f'std_{key}'] = float(np.std(values))

        # Total frames
        overall['total_frames'] = sum(v['num_frames'] for v in per_video)
        overall['num_videos'] = len(per_video)

        self.results['overall'] = overall

        # Print summary
        print("\n" + "=" * 80)
        print("OVERALL RESULTS")
        print("=" * 80)
        print(f"Videos tested: {overall['num_videos']}")
        print(f"Total frames: {overall['total_frames']}")
        print(f"\nMean distance: {overall['mean_mean_distance']:.4f} ± {overall['std_mean_distance']:.4f}")
        print(f"Median distance: {overall['mean_median_distance']:.4f}")
        print(f"Accuracy @0.1: {overall['mean_accuracy_0.1']:.1f}%")
        print(f"Accuracy @0.2: {overall['mean_accuracy_0.2']:.1f}%")
        print("=" * 80 + "\n")

    def save_results(self, save_path: str):
        """Saves evaluation results to JSON.

        Args:
            save_path: Path to save results.
        """
        with open(save_path, 'w') as f:
            json.dump(self.results, f, indent=2)

        print(f"✓ Results saved to: {save_path}")

    def create_visualizations(self, save_dir: str):
        """Creates visualization plots.

        Args:
            save_dir: Directory to save plots.
        """
        print("\nCreating visualizations...")

        os.makedirs(save_dir, exist_ok=True)

        # 1. Distance distribution
        self._plot_distance_distribution(os.path.join(save_dir, "distance_distribution.png"))

        # 2. Accuracy bar chart
        self._plot_accuracy_bars(os.path.join(save_dir, "accuracy_bars.png"))

        # 3. Per-video comparison
        self._plot_per_video_comparison(os.path.join(save_dir, "per_video_comparison.png"))

        print(f"✓ Visualizations saved to: {save_dir}")

    def _plot_distance_distribution(self, save_path: str):
        """Plots histogram of distances."""
        per_video = self.results['per_video']

        fig, ax = plt.subplots(figsize=(10, 6))

        all_distances = []
        for v in per_video:
            all_distances.append(v['mean_distance'])

        ax.hist(all_distances, bins=20, alpha=0.7, edgecolor='black')
        ax.axvline(np.mean(all_distances), color='red', linestyle='--',
                   linewidth=2, label=f'Mean: {np.mean(all_distances):.4f}')

        ax.set_xlabel('Mean Distance', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.set_title('Distribution of Mean Distances Across Videos',
                     fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_accuracy_bars(self, save_path: str):
        """Plots accuracy at different thresholds."""
        overall = self.results['overall']

        thresholds = [0.05, 0.1, 0.15, 0.2]
        accuracies = [overall[f'mean_accuracy_{t}'] for t in thresholds]

        fig, ax = plt.subplots(figsize=(10, 6))

        bars = ax.bar([str(t) for t in thresholds], accuracies,
                      alpha=0.7, edgecolor='black', color='steelblue')

        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height,
                    f'{height:.1f}%',
                    ha='center', va='bottom', fontsize=10)

        ax.set_xlabel('Distance Threshold', fontsize=12)
        ax.set_ylabel('Accuracy (%)', fontsize=12)
        ax.set_title('Accuracy at Different Distance Thresholds',
                     fontsize=14, fontweight='bold')
        ax.set_ylim(0, 100)
        ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_per_video_comparison(self, save_path: str):
        """Plots comparison across videos."""
        per_video = self.results['per_video']

        video_names = [v['video_name'] for v in per_video]
        mean_distances = [v['mean_distance'] for v in per_video]
        median_distances = [v['median_distance'] for v in per_video]

        fig, ax = plt.subplots(figsize=(12, 6))

        x = np.arange(len(video_names))
        width = 0.35

        ax.bar(x - width / 2, mean_distances, width, label='Mean',
               alpha=0.7, edgecolor='black')
        ax.bar(x + width / 2, median_distances, width, label='Median',
               alpha=0.7, edgecolor='black')

        ax.set_xlabel('Video', fontsize=12)
        ax.set_ylabel('Distance', fontsize=12)
        ax.set_title('Mean vs Median Distance Per Video',
                     fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(video_names, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()


def main():
    """Main testing function."""
    print("\n" + "=" * 80)
    print("WEBGAZER GAZE PREDICTION TESTING")
    print("=" * 80 + "\n")

    # Load configuration
    config = load_config("config.yaml")

    # Load data split
    output_dir = config['data']['output_folder']
    split_path = os.path.join(output_dir, "data_split.yaml")

    if not os.path.exists(split_path):
        print(f"❌ Data split not found: {split_path}")
        print("Please run train.py first!")
        return

    with open(split_path, 'r') as f:
        split_data = yaml.safe_load(f)

    test_videos = [v['name'] for v in split_data['test_videos']]

    print(f"Found {len(test_videos)} test videos:")
    for v in test_videos:
        print(f"  - {v}")

    # Load model
    model_path = os.path.join(output_dir, "final_model")

    if not os.path.exists(model_path + ".zip"):
        print(f"\n❌ Model not found: {model_path}.zip")
        print("Please run train.py first!")
        return

    print(f"\nLoading model from: {model_path}")
    model = PPO.load(model_path)
    print("✓ Model loaded successfully")

    # Load gaze data
    print("\nLoading gaze data...")
    loader = WebGazerDataLoader(
        json_folder=config['data']['json_folder'],
        video_folder=config['data']['video_folder'],
        config=config
    )

    video_gaze_map = loader.load_all_mappings()

    # Get full test video paths
    video_folder = config['data']['video_folder']
    test_paths = [os.path.join(video_folder, v) for v in test_videos]

    # Evaluate
    evaluator = GazeEvaluator(config)
    results = evaluator.evaluate_all(model, test_paths, video_gaze_map)

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = os.path.join(output_dir, f"test_results_{timestamp}.json")
    evaluator.save_results(results_path)

    # Create visualizations
    vis_dir = os.path.join(output_dir, f"visualizations_{timestamp}")
    evaluator.create_visualizations(vis_dir)

    print("\n" + "=" * 80)
    print("✓ TESTING COMPLETED SUCCESSFULLY!")
    print("=" * 80)
    print(f"\nResults: {results_path}")
    print(f"Visualizations: {vis_dir}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()