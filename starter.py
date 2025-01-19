import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import folium
from streamlit_folium import folium_static
import geopandas as gpd
from scipy.stats import gaussian_kde
import data_processing as dp

# Constants for energy conversion
AVG_HOUSEHOLD_SIZE = 2.5  # people per household
AVG_HOUSEHOLD_ENERGY = 10000  # kWh per year per household

def convert_power_to_people(power_output_mw):
    """
    Convert power output in MW to number of people that can be served
    """
    power_kwh_year = power_output_mw * 1000 * 8760  # Convert MW to kWh/year (8760 hours in a year)
    households_served = power_kwh_year / AVG_HOUSEHOLD_ENERGY
    people_served = households_served * AVG_HOUSEHOLD_SIZE
    return int(people_served)

def load_population_data():
    csv_filename = "usa_pd_2020_1km_ASCII_XYZ.csv"
    data = pd.read_csv(csv_filename)
    filtered = dp.filter_data(data, -150, -148, 65, 68)
    return filtered

def calculate_optimal_locations(data, n_clusters, max_people_per_pump):
    """
    Use K-means clustering to determine optimal pump locations
    with capacity constraints
    """
    weights = data['Z']
    weights = weights / weights.sum()
    coords = data[['Y', 'X']].values
    
    # Initial clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(coords, sample_weight=weights)
    
    # Assign clusters and calculate population per cluster
    data['cluster'] = kmeans.labels_
    cluster_populations = data.groupby('cluster')['Z'].sum()
    
    # Adjust clusters to respect capacity constraints
    while any(cluster_populations > max_people_per_pump):
        # Find clusters that exceed capacity
        overloaded_clusters = cluster_populations[cluster_populations > max_people_per_pump].index
        
        # Add more clusters to split overloaded ones
        n_clusters += len(overloaded_clusters)
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        kmeans.fit(coords, sample_weight=weights)
        
        # Recalculate populations
        data['cluster'] = kmeans.labels_
        cluster_populations = data.groupby('cluster')['Z'].sum()
    
    optimal_locations = pd.DataFrame(
        kmeans.cluster_centers_,
        columns=['Y', 'X']
    )
    
    return optimal_locations, data

def create_map(data, optimal_locations, cluster_populations):
    """
    Create an interactive map with population density heatmap
    and optimal pump locations
    """
    center_lat = data['Y'].mean()
    center_lon = data['X'].mean()
    m = folium.Map(location=[center_lat, center_lon], zoom_start=8)
    
    # Add population density heatmap
    heat_data = [[row['Y'], row['X'], row['Z']] 
                 for idx, row in data.iterrows()]
    folium.plugins.HeatMap(heat_data).add_to(m)
    
    # Add optimal pump locations
    for idx, row in optimal_locations.iterrows():
        folium.CircleMarker(
            location=[row['Y'], row['X']],
            radius=8,
            color='red',
            fill=True,
            popup=f'Pump Station {idx+1}<br>Population Served: {int(cluster_populations[idx])}'
        ).add_to(m)
    
    return m

def main():
    st.title('Geothermal Pump Placement Optimizer')
    
    st.sidebar.header('Parameters')
    power_output = st.sidebar.number_input(
        'Power Output per Pump (MW)',
        min_value=1.0,
        max_value=100.0,
        value=10.0,
        step=0.5
    )
    
    max_people_per_pump = convert_power_to_people(power_output)
    st.sidebar.info(f'Each pump can serve approximately {max_people_per_pump:,} people')
    
    n_clusters = st.sidebar.slider('Initial Number of Pump Stations', 3, 15, 5)
    
    # Load and process data
    data = load_population_data()
    
    # Calculate optimal locations with capacity constraints
    optimal_locations, clustered_data = calculate_optimal_locations(
        data, 
        n_clusters,
        max_people_per_pump
    )
    
    cluster_populations = clustered_data.groupby('cluster')['Z'].sum()
    
    # Create and display map
    st.subheader('Optimal Pump Locations')
    m = create_map(data, optimal_locations, cluster_populations)
    folium_static(m)
    
    # Display statistics
    st.subheader('Station Statistics')
    stats_df = pd.DataFrame({
        'Station': range(1, len(optimal_locations) + 1),
        'Latitude': optimal_locations['Y'],
        'Longitude': optimal_locations['X'],
        'Population Served': cluster_populations.values,
        'Capacity Utilization (%)': (cluster_populations.values / max_people_per_pump * 100).round(1)
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