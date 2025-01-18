import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import folium
from streamlit_folium import folium_static
import geopandas as gpd
from scipy.stats import gaussian_kde

def load_population_data():
    """
    Placeholder function to load population density data.
    In practice, you would load real data from a CSV or database.
    """
    # Example data structure
    return pd.DataFrame({
        'latitude': np.random.uniform(40.0, 42.0, 1000),
        'longitude': np.random.uniform(-74.0, -72.0, 1000),
        'population_density': np.random.exponential(100, 1000)
    })

def calculate_optimal_locations(data, n_clusters):
    """
    Use K-means clustering to determine optimal pump locations
    based on population density
    """
    # Weight points by population density
    weights = data['population_density']
    
    # Normalize weights
    weights = weights / weights.sum()
    
    # Perform K-means clustering
    coords = data[['latitude', 'longitude']].values
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    
    # Fit K-means with sample weights
    kmeans.fit(coords, sample_weight=weights)
    
    # Get cluster centers and assign clusters
    optimal_locations = pd.DataFrame(
        kmeans.cluster_centers_,
        columns=['latitude', 'longitude']
    )
    
    data['cluster'] = kmeans.labels_
    
    return optimal_locations, data

def calculate_service_radius(cluster_data, percentile=90):
    """
    Calculate recommended service radius for each cluster
    based on distance distribution
    """
    radii = []
    for cluster in cluster_data['cluster'].unique():
        cluster_points = cluster_data[cluster_data['cluster'] == cluster]
        center = cluster_points[['latitude', 'longitude']].mean()
        
        # Calculate distances from center to all points in cluster
        distances = np.sqrt(
            (cluster_points['latitude'] - center['latitude'])**2 +
            (cluster_points['longitude'] - center['longitude'])**2
        )
        
        # Use nth percentile as service radius
        radius = np.percentile(distances, percentile)
        radii.append(radius)
    
    return radii

def create_map(data, optimal_locations, service_radii):
    """
    Create an interactive map with population density heatmap
    and optimal pump locations
    """
    # Create base map centered on data
    center_lat = data['latitude'].mean()
    center_lon = data['longitude'].mean()
    m = folium.Map(location=[center_lat, center_lon], zoom_start=8)
    
    # Add population density heatmap
    heat_data = [[row['latitude'], row['longitude'], row['population_density']] 
                 for idx, row in data.iterrows()]
    folium.plugins.HeatMap(heat_data).add_to(m)
    
    # Add optimal pump locations with service radius circles
    for idx, row in optimal_locations.iterrows():
        folium.CircleMarker(
            location=[row['latitude'], row['longitude']],
            radius=8,
            color='red',
            fill=True,
            popup=f'Pump Station {idx+1}'
        ).add_to(m)
        
        # Add service radius circle
        folium.Circle(
            location=[row['latitude'], row['longitude']],
            radius=service_radii[idx] * 111000,  # Convert to meters
            color='blue',
            fill=False,
            popup=f'Service Radius: {service_radii[idx]:.2f} degrees'
        ).add_to(m)
    
    return m

def main():
    st.title('Geothermal Pump Placement Optimizer')
    
    st.sidebar.header('Parameters')
    n_clusters = st.sidebar.slider('Number of Pump Stations', 3, 15, 5)
    service_percentile = st.sidebar.slider(
        'Service Radius Percentile', 
        50, 
        95, 
        90
    )
    
    # Load data
    data = load_population_data()
    
    # Calculate optimal locations
    optimal_locations, clustered_data = calculate_optimal_locations(
        data, 
        n_clusters
    )
    
    # Calculate service radii
    service_radii = calculate_service_radius(
        clustered_data, 
        percentile=service_percentile
    )
    
    # Create and display map
    st.subheader('Optimal Pump Locations and Service Areas')
    m = create_map(data, optimal_locations, service_radii)
    folium_static(m)
    
    # Display statistics
    st.subheader('Station Statistics')
    stats_df = pd.DataFrame({
        'Station': range(1, len(optimal_locations) + 1),
        'Latitude': optimal_locations['latitude'],
        'Longitude': optimal_locations['longitude'],
        'Service Radius (deg)': service_radii,
        'Population Served': [
            clustered_data[clustered_data['cluster'] == i]['population_density'].sum()
            for i in range(len(optimal_locations))
        ]
    })
    st.dataframe(stats_df)
    
    # Add download button for results
    st.download_button(
        label="Download Results CSV",
        data=stats_df.to_csv(index=False),
        file_name="pump_stations.csv",
        mime="text/csv"
    )

if __name__ == "__main__":
    main()