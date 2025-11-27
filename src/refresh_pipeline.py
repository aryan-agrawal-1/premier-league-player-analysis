"""
Unified data refresh pipeline.

Runs data fetching, preprocessing, and clustering atomically to ensure
all derived artifacts use consistent indices.
"""
import sys
from pathlib import Path
from data_loader import refresh_all_stats
from preprocess import preprocess_pipeline
from clustering import recompute_all_projections


def run_pipeline():
    print("UNIFIED DATA REFRESH PIPELINE")
    
    # Step 1: Fetch fresh data from FBref
    print("\n[1/3] Fetching data from FBref...")
    
    data_success = refresh_all_stats()
    if not data_success:
        print("Warning: Data fetch had issues, continuing with existing data...")
    
    # Step 2: Preprocess and create player vectors
    print("\n[2/3] Running preprocessing pipeline...")
    
    try:
        preprocess_pipeline()
    except Exception as e:
        print(f"Preprocessing failed: {e}")
        return False
    
    # Step 3: Compute PCA and clustering (uses freshly created player_vectors.parquet)
    print("\n[3/3] Computing PCA and clustering...")
    
    try:
        recompute_all_projections()
    except Exception as e:
        print(f"Clustering failed: {e}")
        return False
    
    print("Pipeline Complete")
    return True


if __name__ == "__main__":
    success = run_pipeline()
    sys.exit(0 if success else 1)

