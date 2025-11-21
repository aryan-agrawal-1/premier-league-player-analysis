import os
import sys
from pathlib import Path

import pandas as pd
import soccerdata as sd
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv(Path(__file__).parent / '.env')


STAT_TYPES = [
    "standard",
    "shooting",
    "passing",
    "passing_types",
    "goal_shot_creation",
    "defense",
    "possession",
    "playing_time",
    "misc",
    "keeper",
    "keeper_adv",
]


def _bool_env(var_name, default=False):
    value = os.getenv(var_name)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "on"}


class FBrefDataLoader:
    def __init__(
        self,
        league="ENG-Premier League",
        seasons=None,
        stat_types=None,
        data_root=None,
    ):
        self.league = league
        if seasons is None:
            seasons = ["2025"]
        if isinstance(seasons, (str, int)):
            seasons = [seasons]
        self.seasons = list(seasons)
        self.stat_types = stat_types or list(STAT_TYPES)
        root = data_root or Path(__file__).resolve().parent.parent / "data" / "raw"
        self.data_root = Path(root)
        self.data_root.mkdir(parents=True, exist_ok=True)
        self._client = None
        print(f"Initialized DataLoader for {self.league}, seasons: {self.seasons}")
        print(f"Data directory: {self.data_root}")

    def _client_kwargs(self):
        kwargs = {"leagues": [self.league], "seasons": self.seasons}
        data_dir = os.getenv("SOCCERDATA_DATA_DIR")
        if data_dir:
            kwargs["data_dir"] = data_dir
        if _bool_env("SOCCERDATA_NOCACHE"):
            kwargs["no_cache"] = True
        return kwargs

    def _fbref(self):
        if self._client is None:
            self._client = sd.FBref(**self._client_kwargs())
        return self._client

    def _target_path(self, stat_type):
        season_label = "-".join(map(str, self.seasons))
        stat_dir = self.data_root / season_label
        stat_dir.mkdir(parents=True, exist_ok=True)
        return stat_dir / f"{stat_type}.parquet"

    def fetch_stat(self, stat_type):
        if stat_type not in self.stat_types:
            raise ValueError(f"Unsupported stat_type: {stat_type}")
        target_path = self._target_path(stat_type)
        
        print(f"Fetching {stat_type} from FBref...")
        try:
            frame = self._fbref().read_player_season_stats(stat_type=stat_type)
        except ConnectionError as e:
            print(f"⚠ ConnectionError fetching {stat_type}: {e}")
            print(f"   Skipping {stat_type} and continuing with other stat types...")
            raise
        
        if isinstance(frame.index, pd.MultiIndex):
            frame = frame.reset_index()
        
        if isinstance(frame.columns, pd.MultiIndex):
            frame.columns = ['_'.join(filter(None, map(str, col))).strip('_') for col in frame.columns.values]
        
        print(f"Saving {stat_type} to {target_path} ({len(frame)} rows)")
        frame.to_parquet(target_path, index=False)
        return frame

    def refresh_all(self):
        print(f"\nRefreshing {len(self.stat_types)} stat types...")
        frames = {}
        failed_stats = []
        for i, stat_type in enumerate(self.stat_types, 1):
            print(f"\n[{i}/{len(self.stat_types)}] Processing {stat_type}")
            try:
                frames[stat_type] = self.fetch_stat(stat_type)
            except ConnectionError:
                failed_stats.append(stat_type)
                continue
        
        if failed_stats:
            print(f"\n⚠ Failed to fetch {len(failed_stats)} stat type(s): {', '.join(failed_stats)}")
            print(f"✓ Successfully refreshed {len(frames)}/{len(self.stat_types)} stat types")
        else:
            print(f"\n✓ Completed refresh for all {len(self.stat_types)} stat types")
        
        return frames


def refresh_all_stats():
    print("Starting FBref data refresh...")
    loader = FBrefDataLoader()
    try:
        frames = loader.refresh_all()
        if not frames:
            print("\n⚠ No data was successfully fetched. This may be due to connection issues.")
            return False
        return True
    except Exception as e:
        print(f"\n❌ Unexpected error during data refresh: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = refresh_all_stats()
    sys.exit(0 if success else 0)  # Always exit 0 to allow heartbeat update

