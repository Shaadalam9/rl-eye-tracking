"""WebGazer data loader for memory-efficient JSON processing.

This module handles loading large JSONL files (heroku_0.json, heroku_1.json)
and extracting gaze coordinates efficiently.

Author: Sudhanshu Anand
Date: 2025-10-30
"""

import json
import os
import re
from typing import Dict, List, Tuple, Optional
import numpy as np
from collections import defaultdict
import yaml
from custom_logger import CustomLogger           # structured logging
from logmod import logs
import common

logs(show_level=common.get_configs("logger_level"), show_color=True)
logger = CustomLogger(__name__)


class WebGazerDataLoader:
    """Loads and processes WebGazer experiment data from JSONL files.

    This class handles the large JSONL files where each line contains a complete
    experiment session with multiple trials.

    Attributes:
        json_folder: Path to folder containing JSONL files.
        video_folder: Path to folder containing video files.
        config: Configuration dictionary loaded from YAML.
    """

    def __init__(self, json_folder: str, video_folder: str, config: Dict):
        """Initializes the data loader.

        Args:
            json_folder: Path to folder with heroku_*.json files.
            video_folder: Path to folder with video files.
            config: Configuration dictionary from YAML.
        """
        self.json_folder = json_folder
        self.video_folder = video_folder
        self.config = config

        # Screen dimensions for normalization
        self.screen_width = config['screen']['width']
        self.screen_height = config['screen']['height']

        # Data quality thresholds
        self.min_gaze_points = config['split']['min_gaze_points']

        logger.info("Initialized WebGazerDataLoader")
        logger.info(f"JSON folder: {json_folder}")
        logger.info(f"Video folder: {video_folder}")
        logger.info(f"Screen size: {self.screen_width}x{self.screen_height}")

    def load_json_files(self) -> List[str]:
        """Loads list of JSON files in the folder.

        Returns:
            List of JSON file paths.
        """
        json_files = []
        for filename in os.listdir(self.json_folder):
            if filename.endswith('.json') and filename.startswith('heroku_'):
                json_files.append(os.path.join(self.json_folder, filename))

        json_files.sort()
        logger.info(f"Found {len(json_files)} JSON files: {[os.path.basename(f) for f in json_files]}")
        return json_files

    def extract_video_name(self, trial_data: Dict) -> Optional[str]:
        """Extracts video filename from trial data.

        Args:
            trial_data: Single trial dictionary from JSON.

        Returns:
            Video filename if found, None otherwise.
        """
        # Check if this is a video trial
        trial_type = trial_data.get('trial_type', '')

        # Look for video-related trial types
        if 'video' not in trial_type.lower():
            return None

        # Look for stimulus field (contains video path or filename)
        stimulus = trial_data.get('stimulus', '')

        if not isinstance(stimulus, str):
            # Check for list of stimuli
            if isinstance(stimulus, list) and len(stimulus) > 0:
                stimulus = stimulus[0]
            else:
                return None

        # Extract filename from path using regex
        # Handles: "videos/video_0.mp4", "C:\...\video_0.mp4", etc.
        match = re.search(r'([^/\\]+\.mp4)', stimulus)
        if match:
            return match.group(1)

        return None

    def extract_gaze_coordinates(self, trial_data: Dict) -> List[Tuple[float, float, float]]:
        """Extracts gaze coordinates from trial data.

        Args:
            trial_data: Single trial dictionary from JSON.

        Returns:
            List of (x_normalized, y_normalized, timestamp) tuples.
        """
        gazes = []

        # Check if this trial has webgazer_data
        if 'webgazer_data' not in trial_data:
            return gazes

        webgazer_data = trial_data['webgazer_data']

        # webgazer_data should be a list of gaze points
        if not isinstance(webgazer_data, list):
            return gazes

        # Iterate through all gaze points
        for gaze_point in webgazer_data:
            if not isinstance(gaze_point, dict):
                continue

            # Extract x, y, and timestamp
            if 'x' in gaze_point and 'y' in gaze_point and 't' in gaze_point:
                try:
                    x_raw = float(gaze_point['x'])
                    y_raw = float(gaze_point['y'])
                    timestamp = float(gaze_point['t'])

                    # Normalize coordinates to [0, 1]
                    x_norm = x_raw / self.screen_width
                    y_norm = y_raw / self.screen_height

                    # Clip to valid range
                    x_norm = np.clip(x_norm, 0.0, 1.0)
                    y_norm = np.clip(y_norm, 0.0, 1.0)

                    gazes.append((x_norm, y_norm, timestamp))
                except (ValueError, TypeError):
                    continue

        return gazes

    def process_json_file(self, json_path: str) -> Dict[str, List[Tuple[float, float, float]]]:
        """Processes a single JSONL file and extracts video-gaze mappings.

        Args:
            json_path: Path to JSONL file.

        Returns:
            Dictionary mapping video_name -> list of (x, y, t) tuples.
        """
        logger.info(f"\nProcessing: {os.path.basename(json_path)}")

        video_gaze_map = defaultdict(list)

        try:
            total_sessions = 0
            valid_sessions = 0
            total_trials = 0
            video_trials = 0
            gaze_points_found = 0

            # Read JSONL file line by line (each line is one experiment session)
            with open(json_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    total_sessions += 1
                    line = line.strip()

                    # Skip empty lines
                    if not line:
                        continue

                    try:
                        # Parse each line as a separate JSON object
                        session_data = json.loads(line)
                        valid_sessions += 1

                        # Each session has a 'data' array with trials
                        if 'data' not in session_data:
                            continue

                        trials = session_data['data']
                        if not isinstance(trials, list):
                            continue

                        total_trials += len(trials)

                        # Process each trial in the session
                        for trial in trials:
                            if not isinstance(trial, dict):
                                continue

                            # Extract video name
                            video_name = self.extract_video_name(trial)

                            # Extract gaze coordinates
                            gazes = self.extract_gaze_coordinates(trial)

                            if video_name and len(gazes) > 0:
                                video_gaze_map[video_name].extend(gazes)
                                video_trials += 1
                                gaze_points_found += len(gazes)

                    except json.JSONDecodeError as e:
                        logger.error(f"⚠️  Session {line_num}: JSON decode error - {e}")
                        continue
                    except Exception as e:
                        logger.error(f"⚠️  Session {line_num}: Error - {e}")
                        continue

            logger.info(f"Read {total_sessions} sessions ({valid_sessions} valid)")
            logger.info(f"Processed {total_trials} trials ({video_trials} video trials)")
            logger.info(f"✓ Extracted gaze data for {len(video_gaze_map)} unique videos")
            logger.info(f"✓ Total gaze points: {gaze_points_found}")

            # Filter videos with too few points
            filtered_map = {
                video: gazes for video, gazes in video_gaze_map.items()
                if len(gazes) >= self.min_gaze_points
            }

            if len(filtered_map) < len(video_gaze_map):
                removed = len(video_gaze_map) - len(filtered_map)
                logger.info(f"ℹ️  Filtered out {removed} videos with < {self.min_gaze_points} gaze points")

            return dict(filtered_map)

        except Exception as e:
            logger.info(f"❌ Error processing {json_path}: {e}")
            import traceback
            traceback.print_exc()
            return {}

    def load_all_mappings(self) -> Dict[str, List[Tuple[float, float, float]]]:
        """Loads gaze mappings from all JSONL files.

        Returns:
            Dictionary mapping video_name -> list of (x, y, t) tuples.
        """
        logger.info("\n" + "=" * 80)
        logger.info("LOADING WEBGAZER DATA")
        logger.info("=" * 80)

        json_files = self.load_json_files()

        if len(json_files) == 0:
            raise ValueError(f"No JSON files found in {self.json_folder}")

        # Combine all mappings
        all_mappings = {}

        for json_file in json_files:
            mappings = self.process_json_file(json_file)

            # Merge with existing mappings
            for video_name, gazes in mappings.items():
                if video_name in all_mappings:
                    all_mappings[video_name].extend(gazes)
                else:
                    all_mappings[video_name] = gazes

        # Sort gazes by timestamp for each video
        for video_name in all_mappings:
            all_mappings[video_name].sort(key=lambda x: x[2])  # Sort by timestamp

        logger.info("\n" + "=" * 80)
        logger.info("SUMMARY")
        logger.info("=" * 80)
        logger.info(f"Total unique videos: {len(all_mappings)}")

        for video_name, gazes in sorted(all_mappings.items()):
            logger.info(f"  {video_name}: {len(gazes)} gaze points")

        logger.info("=" * 80 + "\n")

        return all_mappings

    def verify_video_files(self, video_gaze_map: Dict) -> Tuple[List[str], List[str]]:
        """Verifies which videos exist in the video folder.

        Args:
            video_gaze_map: Dictionary from load_all_mappings().

        Returns:
            Tuple of (available_videos, missing_videos).
        """
        logger.info("\n" + "=" * 80)
        logger.info("VERIFYING VIDEO FILES")
        logger.info("=" * 80 + "\n")

        available = []
        missing = []

        for video_name in video_gaze_map.keys():
            video_path = os.path.join(self.video_folder, video_name)

            if os.path.exists(video_path):
                available.append(video_name)
                logger.info(f"✓ {video_name}")
            else:
                missing.append(video_name)
                logger.info(f"✗ {video_name} (not found)")

        logger.info(f"\n{'=' * 80}")
        logger.info(f"Available: {len(available)}/{len(video_gaze_map)}")
        if missing:
            logger.info(f"Missing: {len(missing)} videos")
            logger.info("  " + ", ".join(missing))
        logger.info("=" * 80 + "\n")

        return available, missing

    def get_gaze_for_video(self, video_name: str,
                           video_gaze_map: Dict) -> np.ndarray:
        """Gets gaze coordinates for a specific video.

        Args:
            video_name: Name of the video file.
            video_gaze_map: Dictionary from load_all_mappings().

        Returns:
            Numpy array of shape (N, 2) with normalized (x, y) coordinates.
        """
        if video_name not in video_gaze_map:
            return np.array([])

        gazes = video_gaze_map[video_name]

        # Extract only x, y (ignore timestamp for now)
        coords = np.array([[x, y] for x, y, t in gazes], dtype=np.float32)

        return coords

    def split_train_test(self, available_videos: List[str],
                         video_gaze_map: Dict) -> Tuple[List[str], List[str]]:
        """Splits available videos into train and test sets.

        Args:
            available_videos: List of video names that exist.
            video_gaze_map: Dictionary from load_all_mappings().

        Returns:
            Tuple of (train_videos, test_videos).
        """
        logger.info("\n" + "=" * 80)
        logger.info("SPLITTING TRAIN/TEST SETS")
        logger.info("=" * 80 + "\n")

        # Set random seed
        np.random.seed(self.config['split']['random_seed'])

        # Shuffle videos
        shuffled = available_videos.copy()
        np.random.shuffle(shuffled)

        # Calculate split point
        test_ratio = self.config['split']['test_ratio']
        n_test = max(1, int(len(shuffled) * test_ratio))

        test_videos = shuffled[:n_test]
        train_videos = shuffled[n_test:]

        logger.info(f"Total videos: {len(available_videos)}")
        logger.info(f"Train: {len(train_videos)} ({100 * (1 - test_ratio):.1f}%)")
        logger.info(f"Test: {len(test_videos)} ({100 * test_ratio:.1f}%)")
        logger.info(f"Random seed: {self.config['split']['random_seed']}")

        logger.info("\nTraining videos:")
        for v in train_videos:
            n_gazes = len(video_gaze_map[v])
            logger.info(f"  - {v} ({n_gazes} gaze points)")

        logger.info("\nTesting videos:")
        for v in test_videos:
            n_gazes = len(video_gaze_map[v])
            logger.info(f"  - {v} ({n_gazes} gaze points)")

        logger.info("\n" + "=" * 80 + "\n")

        return train_videos, test_videos

    def save_split(self, train_videos: List[str], test_videos: List[str],
                   video_gaze_map: Dict, output_path: str = "data_split.yaml"):
        """Saves train/test split to YAML file.

        Args:
            train_videos: List of training video names.
            test_videos: List of testing video names.
            video_gaze_map: Dictionary from load_all_mappings().
            output_path: Where to save the split info.
        """
        split_info = {
            'train_videos': [
                {
                    'name': v,
                    'gaze_points': len(video_gaze_map[v])
                }
                for v in train_videos
            ],
            'test_videos': [
                {
                    'name': v,
                    'gaze_points': len(video_gaze_map[v])
                }
                for v in test_videos
            ],
            'total_train': len(train_videos),
            'total_test': len(test_videos),
            'random_seed': self.config['split']['random_seed']
        }

        with open(output_path, 'w') as f:
            yaml.dump(split_info, f, default_flow_style=False)

        logger.info(f"✓ Split saved to: {output_path}")


def load_config(config_path: str = "config.yaml") -> Dict:
    """Loads configuration from YAML file.

    Args:
        config_path: Path to config.yaml file.

    Returns:
        Configuration dictionary.
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    logger.info("Configuration loaded successfully")
    return config


def main():
    """Main function to test the data loader."""
    # Load configuration
    config = load_config("config.yaml")

    # Initialize loader
    loader = WebGazerDataLoader(
        json_folder=config['data']['json_folder'],
        video_folder=config['data']['video_folder'],
        config=config
    )

    # Load all mappings
    video_gaze_map = loader.load_all_mappings()

    if len(video_gaze_map) == 0:
        logger.info("❌ No video-gaze mappings found! Check your JSON files.")
        return

    # Verify video files
    available_videos, missing_videos = loader.verify_video_files(video_gaze_map)

    if len(available_videos) == 0:
        logger.info("❌ No videos found! Check your video folder path.")
        return

    # Split train/test
    train_videos, test_videos = loader.split_train_test(available_videos, video_gaze_map)

    # Save split
    loader.save_split(train_videos, test_videos, video_gaze_map)

    # Show sample data
    logger.info("\n" + "=" * 80)
    logger.info("SAMPLE GAZE DATA")
    logger.info("=" * 80)

    sample_video = train_videos[0]
    sample_gazes = loader.get_gaze_for_video(sample_video, video_gaze_map)

    logger.info(f"\nVideo: {sample_video}")
    logger.info(f"Total gaze points: {len(sample_gazes)}")
    logger.info("\nFirst 5 gaze coordinates (normalized):")
    for i, (x, y) in enumerate(sample_gazes[:5]):
        logger.info(f"{i}: x={x:.4f}, y={y:.4f}")

    logger.info("\nGaze statistics:")
    logger.info(f"X range: [{sample_gazes[:, 0].min():.4f}, {sample_gazes[:, 0].max():.4f}]")
    logger.info(f"Y range: [{sample_gazes[:, 1].min():.4f}, {sample_gazes[:, 1].max():.4f}]")
    logger.info(f"X mean: {sample_gazes[:, 0].mean():.4f} ± {sample_gazes[:, 0].std():.4f}")
    logger.info(f"Y mean: {sample_gazes[:, 1].mean():.4f} ± {sample_gazes[:, 1].std():.4f}")

    logger.info("\n" + "=" * 80)
    logger.info("✓ Data loader test completed successfully!")
    logger.info("=" * 80 + "\n")


if __name__ == "__main__":
    main()
