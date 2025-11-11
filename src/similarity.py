from pathlib import Path

import numpy as np
import pandas as pd


class PlayerVectorStore:
    def __init__(self, data_path=None):
        if data_path is None:
            data_path = Path(__file__).resolve().parent.parent / "data" / "processed" / "player_vectors.parquet"
        
        self.data_path = Path(data_path)
        self.df = None
        self.feature_cols = None
        self.feature_matrix = None
        self.metadata_cols = None
        
        self._load_data()
    
    def _is_stylistic_feature(self, col_name):
        # exclude playing time related features
        col_lower = col_name.lower()
        
        # Patterns that indicate playing time rather than style
        # be specific to avoid false positives (e.g., "cmp" shouldn't match "_mp")
        non_stylistic_patterns = [
            "playing time",  # all playing time columns
            "90s",  # keeping per90 columns removing playing time in 90s
            "starts_",  # starts related columns
            "subs_",  # substitutions related columns
            "mn/mp",  # minutes per match
            "mn/start",  # minutes per start
            "mn/sub",  # minutes per sub
        ]
        
        # Check if column name contains any non-stylistic pattern
        for pattern in non_stylistic_patterns:
            if pattern in col_lower:
                # Special handling for "90s" - only exclude if it's standalone or with "playing time"
                if pattern == "90s":
                    if col_lower == "90s" or col_lower.endswith("_90s") or "playing time" in col_lower:
                        return False
                else:
                    return False
        
        # Also exclude columns that end with _mp but are specifically playing time related
        # (e.g., "Playing Time_MP" but not "Cmp" which is completion)
        if col_lower.endswith("_mp") or col_lower.endswith("_mp_per90"):
            if "playing time" in col_lower or col_lower.startswith("mp"):
                return False
        
        # Exclude exact matches for common playing time columns
        non_stylistic_exact = {
            "playing time_mp", "playing time_min", "playing time_starts",
            "playing time_90s", "playing time_mn/mp",
            "starts_starts", "starts_compl", "starts_mn/start",
            "subs_subs", "subs_unsub", "subs_mn/sub"
        }
        
        if col_lower in non_stylistic_exact:
            return False
        
        return True
    
    def _load_data(self):
        print(f"Loading player vectors from {self.data_path}...")
        self.df = pd.read_parquet(self.data_path)
        
        # Identify metadata columns
        metadata_cols = {"league", "season", "team", "player", "nation", "pos", "age", "born", "position", 
                         "Playing Time_Min", "Playing Time_MP", "Playing Time_Starts"}
        self.metadata_cols = [col for col in self.df.columns if col in metadata_cols]
        
        # Feature columns are all numeric columns that aren't metadata and are stylistic
        self.feature_cols = [
            col for col in self.df.columns
            if col not in self.metadata_cols
            and pd.api.types.is_numeric_dtype(self.df[col])
            and self._is_stylistic_feature(col)
        ]
        
        # extract feature matrix as numpy array
        self.feature_matrix = self.df[self.feature_cols].values
        
        # Handle NaN values by filling with 0
        self.feature_matrix = np.nan_to_num(self.feature_matrix, nan=0.0)
        
        # normalise feature vectors for efficient cosine similarity
        # L2 normalise each row 
        norms = np.linalg.norm(self.feature_matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1  # Avoid division by zero
        self.feature_matrix_norm = self.feature_matrix / norms
        
        print(f"Loaded {len(self.df)} player entries with {len(self.feature_cols)} features")
    
    def get_player_index(self, player_name, team=None):
        # Find index of player in dataframe
        mask = self.df["player"] == player_name
        if team:
            mask = mask & (self.df["team"] == team)
        
        indices = self.df[mask].index.tolist()
        return indices
    
    def get_player_vector(self, player_name, team=None, position=None):
        # Get feature vector for specific player
        mask = self.df["player"] == player_name
        if team:
            mask = mask & (self.df["team"] == team)
        if position:
            mask = mask & (self.df["position"] == position)
        
        matches = self.df[mask]
        if len(matches) == 0:
            return None, None
        
        # Return first match
        idx = matches.index[0]
        return self.feature_matrix_norm[idx], idx
    
    def cosine_similarity(self, vec1, vec2):
        # Compute cosine similarity between two vectors

        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
    
    def find_similar_players(
        self,
        player_name,
        team=None,
        position=None,
        top_n=5,
        filter_position=None,
        filter_team=None,
        exclude_player=True
    ):
        """
        Find players most similar to the given player using cosine similarity.
        
        Args:
            player_name: Name of the player to find similar players for
            team: Optional team name to disambiguate player (not to select specific team)
            position: Optional position to select specific player variant if they play fw and md for example
            top_n: Number of similar players to return
            filter_position: Only return players with this position
            filter_team: Only return players from this team (or exclude if filter_team is list)
            exclude_player: exclude player from results (cause why would they be there)
        
        Returns:
            DataFrame with similar players and similarity scores
        """
        # Get the query player's vector
        query_vec, query_idx = self.get_player_vector(player_name, team, position)
        
        if query_vec is None:
            return pd.DataFrame()
        
        # Compute cosine similarities with all players
        similarities = np.dot(self.feature_matrix_norm, query_vec)
        
        # Build results dataframe
        results = self.df.copy()
        results["similarity"] = similarities
        
        # Apply filters
        if exclude_player:
            # Exclude all entries for this player (all positions)
            results = results[results["player"] != player_name]
        
        if filter_position:
            results = results[results["position"] == filter_position]
        
        if filter_team:
            if isinstance(filter_team, list):
                # Exclude these teams
                results = results[~results["team"].isin(filter_team)]
            else:
                # Only include this team
                results = results[results["team"] == filter_team]
        
        # Remove duplicates - keep highest similarity for each unique player
        results = results.sort_values("similarity", ascending=False)
        results = results.drop_duplicates(subset=["player", "team"], keep="first")
        
        # Return top N
        return results.head(top_n)[self.metadata_cols + ["similarity"]]
    
    def get_all_players(self):
        # Get list of all unique players
        return self.df[["player", "team", "position"]].drop_duplicates()


def find_similar_players(player_name, top_n=5, **kwargs):
    # easy function to find similar players
    store = PlayerVectorStore()
    return store.find_similar_players(player_name, top_n=top_n, **kwargs)


if __name__ == "__main__":
    # Test the similarity module
    store = PlayerVectorStore()
    
    print("\n" + "=" * 60)
    print("Testing similarity search...")
    print("=" * 60)
    
    # Get a sample player
    sample_player = "Alexandros Kyziridis"
    sample_team = "Hearts"
    
    print(f"\nFinding players similar to: {sample_player} ({sample_team})")
    similar = store.find_similar_players(sample_player, team=sample_team, top_n=5)
    
    print(f"\nTop 5 similar players:")
    print(similar[["player", "team", "position", "similarity"]])

