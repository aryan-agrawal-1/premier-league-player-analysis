from pathlib import Path
import numpy as np
import pandas as pd
import yaml
import json
from similarity import PlayerVectorStore


def compute_pca(feature_matrix, n_components=2):
    """
    Compute Principal Component Analysis using numpy SVD.
    
    Args:
        feature_matrix: numpy array of shape (n_samples, n_features) - feature matrix (will be standardized)
        n_components: number of principal components to compute (default: 2 for visualization)
    
    Returns:
        components: numpy array of shape (n_components, n_features) - principal component vectors
        explained_variance: numpy array of shape (n_components,) - variance explained by each component
        explained_variance_ratio: numpy array of shape (n_components,) - percentage of variance explained
        projection: numpy array of shape (n_samples, n_components) - data projected onto principal components
    """
    # Standardize features (mean=0, std=1) for PCA
    mean_vec = np.mean(feature_matrix, axis=0)
    std_vec = np.std(feature_matrix, axis=0)
    std_vec[std_vec == 0] = 1.0  # Avoid division by zero
    standardized_vec = (feature_matrix - mean_vec) / std_vec
    
    # Center the standardized data (should already be ~0 mean, but ensure it)
    centered_vec = standardized_vec - np.mean(standardized_vec, axis=0)
    n_samples = feature_matrix.shape[0]

    # Perform SVD decomposition on the centered_matrix
    # SVD decomposes any matrix into three simpler matrices
    # V_t's rows are the pricipal component vectors
    # prinicpal component vectors are rows of length features (number of stats)
    # These vectors tell you how to combine the original stats to form the new, uncorrelated PC axes.
    # Looks like linear regression simple e.g. you could have PC1 = 0.4 x G/90 + 0.5 x G+A/90 - 0.1 x Tck/90
    # he coefficients in this vector tell you that PC 1 is highly associated with players who score and shoot a lot, and negatively associated with players who tackle a lot. This might define an "Attacking Productivity" component.

    # Singular values (S) reflect the importance of each corresponding Principal Component
    # High Singular Value: The corresponding PC captures a large amount of the total variance in the player stats

    U, S, V_t = np.linalg.svd(centered_vec)

    # Extract principal components
    components = V_t[:n_components].copy()

    # Eigenvalues of covariance matrix are S^2 / (n-1)
    eigenvalues = S**2 / (n_samples - 1)
    explained_var = eigenvalues[:n_components]
    explained_var_ratio = explained_var / np.sum(eigenvalues)

    # Project data onto principal components
    # result here is a n_samples x n_components
    # It gives you the player's new coordinates in the reduced PCA space.
    # Kyziridis could be (4, 0.5) for example
    projection = np.dot(centered_vec, components.T)

    # Ensure the dominant feature on each component has a positive loading for interpretability
    for idx, comp_vec in enumerate(components):
        dominant_idx = np.argmax(np.abs(comp_vec))
        if comp_vec[dominant_idx] < 0:
            components[idx] *= -1
            projection[:, idx] *= -1

    return components, explained_var, explained_var_ratio, projection


def fit_pca(vector_store, n_components=2, top_feature_count=4):
    """
    Args:
        vector_store: PlayerVectorStore instance containing feature_matrix
        n_components: number of principal components to compute per position
        top_feature_count: number of dominant features to retain for axis summaries
    
    Returns:
        dict keyed by position with projection coordinates and component metadata
    """
    # need to get non normalised data
    feature_matrix = vector_store.df[vector_store.feature_cols].values
    feature_matrix = np.nan_to_num(feature_matrix, nan=0.0)

    position_series = vector_store.df['position'].fillna('Unknown')
    unique_positions = sorted(position_series[position_series != 'Unknown'].unique())

    positional_payload = {}

    for position in unique_positions:
        mask = position_series == position
        indices = vector_store.df.index[mask].to_numpy()
        position_matrix = feature_matrix[mask]

        if len(position_matrix) < n_components:
            print(f"Warning: Not enough samples for {position}, skipping PCA...")
            continue

        components, explained_variance, explained_variance_ratio, projection = compute_pca(position_matrix, n_components=n_components)

        axis_features = []
        for comp_vec in components:
            ranked_idx = np.argsort(np.abs(comp_vec))[::-1][:top_feature_count]
            features = []
            for idx in ranked_idx:
                features.append({
                    'name': vector_store.feature_cols[idx],
                    'weight': float(comp_vec[idx])
                })
            axis_features.append(features)

        positional_payload[position] = {
            'components': components,
            'explained_variance': explained_variance,
            'explained_variance_ratio': explained_variance_ratio,
            'projection': projection,
            'indices': indices,
            'axis_features': axis_features
        }

    return {
        'positions': positional_payload,
        'feature_cols': list(vector_store.feature_cols),
        'top_feature_count': top_feature_count
    }
    

def save_pca_results(pca_results, output_path=None):
    """
    Save PCA results to disk.
    
    Args:
        pca_results: dict returned from fit_pca()
        output_path: optional path to save results (default: data/processed/pca_projection.parquet)
    """
    if output_path is None:
        output_path = Path(__file__).resolve().parent.parent / "data" / "processed" / "pca_projection.parquet"
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    projection_rows = []
    component_store = {}
    axis_metadata = {}

    for position, payload in pca_results['positions'].items():
        projection = payload['projection']
        indices = payload['indices']

        for idx, coords in zip(indices, projection):
            projection_rows.append({
                'index': int(idx),
                'position': position,
                'pc1': coords[0],
                'pc2': coords[1]
            })

        key_prefix = f"{position}__"
        component_store[f"{key_prefix}components"] = payload['components']
        component_store[f"{key_prefix}explained_variance"] = payload['explained_variance']
        component_store[f"{key_prefix}explained_variance_ratio"] = payload['explained_variance_ratio']

        axis_metadata[position] = []
        for idx, features in enumerate(payload['axis_features'], start=1):
            axis_metadata[position].append({
                'component': idx,
                'features': features
            })

    if not projection_rows:
        raise ValueError('No PCA projections available to save')

    projection_df = pd.DataFrame(projection_rows).set_index('index')
    projection_df.to_parquet(output_path)

    components_path = output_path.parent / "pca_components.npz"
    np.savez(components_path, **component_store)

    axis_path = output_path.parent / "pca_axis_metadata.json"
    with open(axis_path, 'w') as f:
        json.dump(axis_metadata, f, indent=2)

    print(f"Saved PCA projection to {output_path}")
    print(f"Saved PCA components to {components_path}")
    print(f"Saved PCA axis metadata to {axis_path}")


def load_pca_results(projection_path=None):
    """
    Load PCA results from disk.
    
    Args:
        projection_path: optional path to projection file
    
    Returns:
        dict with 'projection_df' and 'components_data'
    """
    if projection_path is None:
        projection_path = Path(__file__).resolve().parent.parent / "data" / "processed" / "pca_projection.parquet"
    
    projection_path = Path(projection_path)
    projection_df = pd.read_parquet(projection_path)

    components_path = projection_path.parent / "pca_components.npz"
    components_data = np.load(components_path)

    axis_path = projection_path.parent / "pca_axis_metadata.json"
    axis_metadata = {}
    if axis_path.exists():
        with open(axis_path, 'r') as f:
            axis_metadata = json.load(f)

    return {
        'projection_df': projection_df,
        'components_data': components_data,
        'axis_metadata': axis_metadata
    }

# K-means++ is designed to pick initial centers that are far away from each other 
# reduces the chance of the K-means algorithm converging to a locally optimal poor solution
def kmeans_plusplus_init(feature_matrix, n_clusters, random_state=None):
    """
    Args:
        feature_matrix: numpy array of shape (n_samples, n_features)
        n_clusters: number of clusters
        random_state: seed for reproducibility
    
    Returns:
        centroids: numpy array of shape (n_clusters, n_features) - initial cluster centers
    """
    if random_state:
        np.random.seed(random_state)
    
    n_samples = feature_matrix.shape[0]

    # Initialise first centroid randomly
    first_idx = np.random.randint(0, n_samples)
    centroids = [feature_matrix[first_idx]]

    # Compute squared distances from all points to nearest existing centroid
    # find distance to closest centroid
    # append min distance to min distances array
    for k in range(1, n_clusters):

        # Compute squared distances from all points to nearest existing centroid
        min_distances = []
        for point in feature_matrix:
            # Squared distance from 'point' to ALL currently chosen 'centroids'
            distances_to_centroids = [np.sum((point - centroid)**2) for centroid in centroids]
            # Find the distance to the closest centroi
            min_distances.append(min(distances_to_centroids))

        min_distances = np.array(min_distances)
        # convert distances to probabilities
        # Add a small epsilon to the denominator to prevent division by zero if all distances are 0
        # if player is far from all previous centroids, min_distances will be large
        # denominator the sum of all points' squared distances to their nearest center, normalises the distribution
        # Players who are currently far from all centroids are given a high probability of being selected as the next one
        probabilities = min_distances / np.sum(min_distances + 1e-8)


        next_idx = np.random.choice(n_samples, p=probabilities)
        centroids.append(feature_matrix[next_idx])

    centroids = np.array(centroids)
    return centroids


# partition N data points into K clusters such that the SSE is minimised
# The algorithm alternates between two steps, Assignment and Update until convergence
# minimizing the Inertia (or Within-Cluster Sum of Squares, WCSS):
def compute_kmeans(feature_matrix, n_clusters, max_iters=100, tol=1e-4, random_state=None):
    """
    
    Args:
        feature_matrix: numpy array of shape (n_samples, n_features)
        n_clusters: number of clusters
        max_iters: maximum iterations
        tol: tolerance for convergence (stop if centroid movement < tol)
        random_state: seed for reproducibility
    
    Returns:
        labels: numpy array of shape (n_samples,) - cluster assignment for each point
        centroids: numpy array of shape (n_clusters, n_features) - final cluster centers
        inertia: float - sum of squared distances to nearest centroid
    
    """
    # initialise with kmeans++
    centroids = kmeans_plusplus_init(feature_matrix, n_clusters, random_state=random_state)
    n_samples = feature_matrix.shape[0]
    labels = np.zeros(n_samples, dtype=int)

    for i in range(max_iters):
        # ASSIGNMENT STEP
        # get distance of points to all 
        distances = np.linalg.norm(feature_matrix[:, np.newaxis, :] - centroids[np.newaxis, :, :], axis=2)
        # assign point to nearest centroid
        labels = np.argmin(distances, axis=1)

        # UPDATE STEP
        # Goal: Keep the assignments fixed and move the centroids to minimize the WCSS for the current assignment
        new_centroids = np.zeros_like(centroids) # same shape and size as centroids

        for cluster in range(n_clusters):
            mask = labels == cluster
            # If no players are assigned to a cluster reinitialize that centroid randomly to prevent the algorithm from failing
            if np.sum(mask) > 0:
                # compute the average stats vector for all players assigned to cluster
                # This is the point that minimises the WCSS for the assigned points within the cluster
                new_centroids[cluster] = np.mean(feature_matrix[mask], axis=0)
            else:
                new_centroids[cluster] = feature_matrix[np.random.randint(0, n_samples)]

        # CONVERGENCE STEP
        # Goal: Stop iterating when the algorithm has found a stable solution
        # The algorithm converges when the centroids stop moving (or movement less than tolerance)        
        movement = np.max(np.linalg.norm(new_centroids - centroids, axis=1))
        if movement < tol:
            break
        else:
            centroids = new_centroids
    

    distances_to_assigned = np.linalg.norm(feature_matrix - centroids[labels], axis=1)

    # The final minimised value of the WCSS (Sum of Squared Errors)
    # A lower inertia means the clusters are tighter and the player stats within each cluster are more similar to each other
    # Inertia is typically used to evaluate the quality of the clustering

    inertia = np.sum(distances_to_assigned ** 2)

    return labels, centroids, inertia

# We will cluster separately for each position so we get stylistic differences within each position
def fit_positional_kmeans(vector_store, cluster_config=None):
    """
    
    Args:
        vector_store: PlayerVectorStore instance
        cluster_config: dict mapping position -> n_clusters (default: config from file or hardcoded defaults)
    
    Returns:
        dict mapping position -> {
            'labels': cluster assignments,
            'centroids': cluster centers,
            'inertia': sum of squared distances,
            'indices': row indices in original dataframe
        }
    
    Step-by-step implementation:
    1. Define default cluster config if not provided:
       - defaults = {
           'Attacker': 4,
           'Midfielder': 4,
           'Defender': 4,
           'Keeper': 2
         }
       - If cluster_config is None, try to load from config/clusters.yaml using yaml
       - If file doesn't exist, use defaults
    
    2. Get feature matrix and metadata from vector_store:
       - feature_matrix = vector_store.df[vector_store.feature_cols].values
       - Handle NaN: feature_matrix = np.nan_to_num(feature_matrix, nan=0.0)
       - positions = vector_store.df['position'].values
    
    3. For each position in cluster_config:
       a. Filter data for this position:
          - mask = positions == position
          - position_indices = np.where(mask)[0]
          - position_matrix = feature_matrix[position_indices]
       
       b. Skip if not enough samples (need at least n_clusters):
          - if len(position_matrix) < cluster_config[position]:
              print(f"Warning: Not enough samples for {position}, skipping...")
              continue
       
       c. Compute K-Means:
          - labels, centroids, inertia = compute_kmeans(
              position_matrix, 
              n_clusters=cluster_config[position],
              random_state=42
            )
       
       d. Store results with original indices:
          - results[position] = {
              'labels': labels,
              'centroids': centroids,
              'inertia': inertia,
              'indices': position_indices
            }
    
    4. Return results dict
    """
    
    defaults = {
        'Attacker': 4,
        'Midfielder': 4,
        'Defender': 4,
        'Keeper': 2
    }
    
    if cluster_config is None:
        config_path = Path(__file__).resolve().parent.parent / "config" / "clusters.yaml"
        if config_path.exists() and yaml is not None:
            with open(config_path, 'r') as f:
                cluster_config = yaml.safe_load(f)
        else:
            cluster_config = defaults
    else:
        cluster_config = cluster_config
    
    results = {}
    
    feature_matrix = vector_store.df[vector_store.feature_cols].values
    feature_matrix = np.nan_to_num(feature_matrix, nan=0.0)
    positions = vector_store.df['position'].values

    for pos, ks in cluster_config.items():
        # filter out each position
        mask = positions == pos
        position_indices = np.where(mask)[0]
        position_matrix = feature_matrix[position_indices]

        if len(position_matrix) < ks:
            print(f"Warning: Not enough samples for {pos}, skipping...")
            continue

        labels, centroids, inertia = compute_kmeans(
              position_matrix, 
              n_clusters=ks,
              random_state=42
        )

        results[pos] = {
              'labels': labels,
              'centroids': centroids,
              'inertia': inertia,
              'indices': position_indices
            }

    return results


def save_clustering_results(clustering_results, output_path=None):
    """  
    Args:
        clustering_results: dict returned from fit_positional_kmeans()
        output_path: optional path to save results
    """
    if output_path is None:
        output_path = Path(__file__).resolve().parent.parent / "data" / "processed" / "clustering_results.parquet"
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    rows = []
    for position, results in clustering_results.items():
        labels = results['labels']
        indices = results['indices']
        for local_idx, idx in enumerate(indices):
            rows.append({
                'index': idx,
                'position': position,
                'cluster': labels[local_idx]
            })
    
    cluster_df = pd.DataFrame(rows)
    cluster_df.to_parquet(output_path, index=False)
    
    centroids_path = output_path.parent / "cluster_centroids.npz"
    centroids_dict = {f"{pos}_centroids": clustering_results[pos]['centroids'] 
                      for pos in clustering_results}
    np.savez(centroids_path, **centroids_dict)
    
    metadata_path = output_path.parent / "clustering_metadata.json"
    metadata = {}
    for position, results in clustering_results.items():
        metadata[position] = {
            'inertia': float(results['inertia']),
            'n_clusters': len(np.unique(results['labels']))
        }
    
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Saved clustering assignments to {output_path}")
    print(f"Saved cluster centroids to {centroids_path}")
    print(f"Saved clustering metadata to {metadata_path}")


def load_clustering_results(cluster_path=None):
    """
    Load clustering results from disk.
    
    Args:
        cluster_path: optional path to cluster assignments file
    
    Returns:
        dict with cluster assignments DataFrame and centroids data
    """
    if cluster_path is None:
        cluster_path = Path(__file__).resolve().parent.parent / "data" / "processed" / "clustering_results.parquet"
    
    cluster_path = Path(cluster_path)
    cluster_df = pd.read_parquet(cluster_path)
    
    centroids_path = cluster_path.parent / "cluster_centroids.npz"
    centroids_data = np.load(centroids_path)
    
    metadata_path = cluster_path.parent / "clustering_metadata.json"
    metadata = {}
    if metadata_path.exists():
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
    
    return {
        'cluster_df': cluster_df,
        'centroids_data': centroids_data,
        'metadata': metadata
    }


def recompute_all_projections(vector_store_path=None):
    """
    Args:
        vector_store_path: optional path to player_vectors.parquet
    """
    if vector_store_path:
        store = PlayerVectorStore(vector_store_path)
    else:
        store = PlayerVectorStore()
    
    pca_res = fit_pca(store)
    save_pca_results(pca_res)
    
    clustering_res = fit_positional_kmeans(store)
    save_clustering_results(clustering_res)
    
    print("\n=== Summary ===")
    for position, payload in pca_res['positions'].items():
        ratio = payload['explained_variance_ratio']
        print(f"PCA {position}: {np.round(ratio, 3)}")
    print("\nClustering results:")
    for position, results in clustering_res.items():
        print(f"  {position}: {len(np.unique(results['labels']))} clusters, inertia: {results['inertia']:.2f}")
    


if __name__ == "__main__":
    # Test PCA and clustering
    print("Loading vector store...")
    store = PlayerVectorStore()
    
    print("\nFitting PCA...")
    pca_results = fit_pca(store)
    for position, payload in pca_results['positions'].items():
        ratio = payload['explained_variance_ratio']
        print(f"Explained variance ratio for {position}: {np.round(ratio, 3)}")
    
    print("\nSaving PCA results...")
    save_pca_results(pca_results)
    
    print("\nFitting K-Means clustering...")
    clustering_results = fit_positional_kmeans(store)
    
    print("\nClustering summary:")
    for position, results in clustering_results.items():
        print(f"  {position}: {len(np.unique(results['labels']))} clusters, inertia: {results['inertia']:.2f}")
    
    print("\nSaving clustering results...")
    save_clustering_results(clustering_results)
    
    print("\nDone!")

