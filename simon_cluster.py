import pandas as pd
import numpy as np
from sklearn.cluster import KMeans

import data_processing as dp



# Constants
KM_PER_DEGREE = 111.32
MAX_DISTANCE_KM = 15
PEOPLE_PER_MW_YEAR = 625

def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculate haversine distance between points"""
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    return 6371 * 2 * np.arcsin(np.sqrt(a))

def calculate_population(data: pd.DataFrame):
    """Calculate population using area method"""
    if len(data) < 2:
        return 0
    square_area = (0.008333333299987 * 111.32) * (0.0083333333000013 * 111.32)
    pop = data["Z"].sum() * square_area
    return pop

def calculate_clusters(data, n_stations, power_per_station_mw=10):
    """
    Optimize geothermal pump placement and generate visualization
    
    Parameters:
    -----------
    data : pandas.DataFrame
        DataFrame with columns 'X' (longitude), 'Y' (latitude), 'Z' (population density)
    n_stations : int
        Number of stations to place
    power_per_station_mw : float
        Power output per station in megawatts
    plot : bool, optional
        Whether to generate and show the plot
    figsize : tuple, optional
        Figure size for the plot
        
    Returns:
    --------
    tuple
        (centers DataFrame, filtered data DataFrame, populations Series, matplotlib figure if plot=True)
    """
    max_people = power_per_station_mw * PEOPLE_PER_MW_YEAR
    
    

    # data['within_range'] = False
    # data['distance'] = 0.0
    
    # Normalize coordinates and density for weighted clustering
    coords = np.column_stack([
        data[['Y', 'X']].values,
        0.1 * (data['Z'] - data['Z'].min()) / (data['Z'].max() - data['Z'].min())
    ])
    
    # Cluster with density weights
    weights = np.clip(data['Z'], 50, 5000) / data['Z'].sum()
    weights = -weights + max(weights) + 1
    #weights = 1 / weights
    kmeans = KMeans(n_clusters=n_stations, random_state=42)
    kmeans.fit(coords, sample_weight=weights)
    
    # Process results
    # data['cluster'] = kmeans.labels_
    # centers = pd.DataFrame(kmeans.cluster_centers_[:, :2], columns=['Y', 'X'])
    
    # # Calculate distances and filter points
    # populations = []
    
    # for i in range(len(centers)):
    #     mask = data['cluster'] == i
    #     cluster_data = data[mask]
        
    #     distances = [haversine_distance(row['Y'], row['X'], centers.iloc[i]['Y'], centers.iloc[i]['X'])
    #                 for _, row in cluster_data.iterrows()]
        
    #     data.loc[mask, 'distance'] = distances
    #     valid_mask = np.array(distances) <= MAX_DISTANCE_KM
    #     data.loc[mask & pd.Series(valid_mask, index=mask[mask].index), 'within_range'] = True
        
    #     valid_points = cluster_data[valid_mask]
    #     populations.append(min(calculate_population(valid_points), max_people))
    
    # filtered_data = data[data['within_range']]
    # populations = pd.Series(populations)
    

    
    return kmeans.cluster_centers_[:, :2], kmeans.labels_