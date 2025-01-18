import numpy as np
from scipy.optimize import linear_sum_assignment 
import pandas as pd 
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D

def balanced_kmeans(X, n_clusters, max_iterations=100):
    print("Starting balanced k-means...")
    n_samples = X.shape[0]
    points_per_cluster = n_samples // n_clusters
    
    # Initialize centroids randomly
    centroids = X[np.random.choice(n_samples, n_clusters, replace=False)]
    labels = np.zeros(n_samples, dtype=int)
    
    for iteration in range(max_iterations):
        print(f"Iteration {iteration + 1}/{max_iterations}")
        
        # Calculate distances
        distances = np.zeros((n_samples, n_clusters))
        for i in range(n_clusters):
            distances[:, i] = np.sum((X - centroids[i])**2, axis=1)
        
        # Sort points by distance to each centroid
        sorted_indices = np.argsort(distances, axis=0)
        
        # Assign points to clusters ensuring equal size
        labels = np.full(n_samples, -1)
        assigned_counts = np.zeros(n_clusters, dtype=int)
        
        # First pass: assign points to their closest available centroid
        for i in range(n_samples):
            for j in range(n_clusters):
                point_idx = sorted_indices[i, j]
                if assigned_counts[j] < points_per_cluster and labels[point_idx] == -1:
                    labels[point_idx] = j
                    assigned_counts[j] += 1
                    break
        
        # Assign any remaining points
        unassigned = labels == -1
        if np.any(unassigned):
            remaining_points = np.where(unassigned)[0]
            for i, point_idx in enumerate(remaining_points):
                cluster = i % n_clusters
                labels[point_idx] = cluster
        
        # Update centroids
        new_centroids = np.array([X[labels == i].mean(axis=0) for i in range(n_clusters)])
        
        # Check convergence
        if np.allclose(centroids, new_centroids):
            print("Converged!")
            break
            
        centroids = new_centroids
        
        # Print current cluster sizes
        for i in range(n_clusters):
            print(f"Cluster {i} size: {np.sum(labels == i)}")
    
    return labels, centroids

def process_and_cluster(csv_path, n_clusters):
    # Read CSV file
    print("Reading CSV file...")
    df = pd.read_csv(csv_path, nrows=8000)  # Only read first 8000 rows
    
    # Extract x, y, z coordinates and scale Z by 0.1
    print("Extracting coordinates...")
    X = df[['X', 'Y', 'Z']].values
    X[:, 2] = X[:, 2] * 0.1  # Scale Z (population) by 0.1
    
    # Apply balanced k-means
    print(f"Clustering {len(X)} points into {n_clusters} clusters...")
    labels, centroids = balanced_kmeans(X, n_clusters)
    
    # Print final cluster sizes
    print("\nFinal cluster sizes:")
    for i in range(n_clusters):
        print(f"Cluster {i} size: {np.sum(labels == i)}")
    
    # Visualize results
    print("\nCreating visualization...")
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # For visualization, use original Z values (unscaled)
    visualization_points = df[['X', 'Y', 'Z']].values
    
    scatter = ax.scatter(visualization_points[:, 0], 
                        visualization_points[:, 1], 
                        visualization_points[:, 2],
                        c=labels, cmap='viridis', alpha=0.6)
    
    # For centroids, unscale the Z coordinate for visualization
    centroids[:, 2] = centroids[:, 2] * 10  # Unscale Z for visualization
    ax.scatter(centroids[:, 0], centroids[:, 1], centroids[:, 2],
              color='red', marker='x', s=200, linewidth=3, label='Centroids')
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z (Population)')
    plt.colorbar(scatter)
    plt.legend()
    plt.title(f'3D Balanced K-means Clustering (k={n_clusters})')
    plt.show()
    
    return labels, centroids, X

csv_path = 'usa_pd_2020_1km_ASCII_XYZ.csv'
n_clusters = 3  # Number of desired clusters
labels, centroids, points = process_and_cluster(csv_path, n_clusters)