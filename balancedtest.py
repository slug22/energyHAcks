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

def count_population(pop_data: pd.DataFrame):
    """
    Calculate population using area and density method
    """
    if len(pop_data) < 2:  # Need at least 2 points to calculate range
        return 0
        
    x_range = pop_data["X"].max() - pop_data["X"].min()
    x_dist = x_range * 111.32 * np.cos(np.mean(pop_data["Y"]) * np.pi / 180)
    
    y_range = pop_data["Y"].max() - pop_data["Y"].min()
    y_dist = y_range * 111.32
    
    area = x_dist * y_dist
    average_density = pop_data["Z"].mean()
    return area * average_density

def load_population_data():
    csv_filename = "usa_pd_2020_1km_ASCII_XYZ.csv"
    data = pd.read_csv(csv_filename)
    filtered = dp.filter_data(data, -105, -104, 39, 40)
    return filtered
def calculate_optimal_locations(data, n_clusters, max_people_per_pump):
    """
    Use K-means clustering to determine optimal pump locations with:
    - Strict population caps per pump
    - Inverse density weighting (favoring less dense areas)
    - Urban density cap to avoid oversaturated areas
    """
    if len(data) == 0:
        st.error("No data points found in the selected region. Please adjust the latitude/longitude bounds.")
        return pd.DataFrame(columns=['Y', 'X']), data, pd.Series([])
    
    # Constants for density calculations
    URBAN_DENSITY_CAP = 5000  # people per sq km - typical suburban density
    MIN_DENSITY = 50  # people per sq km - minimum density to consider
    
    # Calculate inverse density weights with caps
    weights = data['Z'].copy()
    weights = np.clip(weights, MIN_DENSITY, URBAN_DENSITY_CAP)  # Cap density range
    weights = 1 / weights  # Inverse weighting
    weights = weights / weights.sum()  # Normalize weights
    
    coords = data[['Y', 'X']].values
    
    # Initial clustering with user-specified number of clusters
    n_clusters = min(n_clusters, len(data))
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(coords, sample_weight=weights)
    
    # Assign clusters
    data['cluster'] = kmeans.labels_
    
    # Calculate and cap populations for each cluster
    cluster_populations = []
    uncapped_populations = []  # Store original populations for reporting
    for cluster_id in range(kmeans.n_clusters):
        cluster_data = data[data['cluster'] == cluster_id]
        uncapped_pop = count_population(cluster_data)
        capped_pop = min(uncapped_pop, max_people_per_pump)
        
        cluster_populations.append(capped_pop)
        uncapped_populations.append(uncapped_pop)
        
        # If significant capping occurred, warn the user
        if uncapped_pop > max_people_per_pump:
            excess = uncapped_pop - max_people_per_pump
            service_ratio = (max_people_per_pump / uncapped_pop * 100)
            st.warning(
                f"Station {cluster_id + 1} can only serve {service_ratio:.1f}% of its area's population. "
                f"Excess: {int(excess):,} people. Consider adding more stations in this region."
            )
    
    cluster_populations = pd.Series(cluster_populations)
    uncapped_populations = pd.Series(uncapped_populations)
    
    # Create optimal locations dataframe
    optimal_locations = pd.DataFrame(
        kmeans.cluster_centers_,
        columns=['Y', 'X']
    )
    
    # Add density information to the data
    data['original_population'] = data['Z']
    data['weighted_density'] = weights
    
    return optimal_locations, data, cluster_populations, uncapped_populations
def create_map(data, optimal_locations, cluster_populations):
    """
    Create an interactive map with:
    - Colored dots for each cluster
    - Optimal pump locations marked as larger circles
    """
    if len(data) == 0:
        # Create empty map centered on Alaska
        m = folium.Map(location=[66.5, -149], zoom_start=6)
        return m
        
    center_lat = data['Y'].mean()
    center_lon = data['X'].mean()
    m = folium.Map(location=[center_lat, center_lon], zoom_start=8)
    
    # Define a color palette for clusters
    colors = [
        '#e6194B', '#3cb44b', '#ffe119', '#4363d8', '#f58231', 
        '#911eb4', '#42d4f4', '#f032e6', '#bfef45', '#fabed4',
        '#469990', '#dcbeff', '#9A6324', '#fffac8', '#800000'
    ]
    
    # Add data points colored by cluster
    for idx, row in data.iterrows():
        cluster_id = int(row['cluster'])
        color = colors[cluster_id % len(colors)]
        folium.CircleMarker(
            location=[row['Y'], row['X']],
            radius=3,  # Small dots for data points
            color=color,
            fill=True,
            fillOpacity=0.7,
            weight=1,
            popup=f'Cluster {cluster_id + 1}<br>Population Density: {row["Z"]:.1f}'
        ).add_to(m)
    
    # Add optimal pump locations with larger markers
    for idx, row in optimal_locations.iterrows():
        folium.CircleMarker(
            location=[row['Y'], row['X']],
            radius=10,  # Larger circles for pump stations
            color='red',
            fill=True,
            fillOpacity=0.9,
            weight=2,
            popup=f'Pump Station {idx+1}<br>Population Served: {int(cluster_populations[idx]):,}'
        ).add_to(m)
        
        # Add a label for each pump station
        folium.Popup(
            f'Station {idx+1}',
            permanent=True
        ).add_to(folium.CircleMarker(
            location=[row['Y'], row['X']],
            radius=0,
            weight=0,
            fillOpacity=0
        ).add_to(m))
    
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
    
# In main():
    try:
        # Load and process data
        data = load_population_data()
        
        # Calculate optimal locations with capacity constraints
        optimal_locations, clustered_data, cluster_populations, uncapped_populations = calculate_optimal_locations(
            data, 
            n_clusters,
            max_people_per_pump
        )
        
        # Create and display map
        st.subheader('Optimal Pump Locations')
        m = create_map(data, optimal_locations, cluster_populations)
        folium_static(m)
        
        # Only display statistics if we have data
        if len(optimal_locations) > 0:
            # Display statistics
            st.subheader('Station Statistics')
            stats_df = pd.DataFrame({
                'Station': range(1, len(optimal_locations) + 1),
                'Latitude': optimal_locations['Y'],
                'Longitude': optimal_locations['X'],
                'Population Served': cluster_populations.values.round(0),
                'Total Population in Area': uncapped_populations.values.round(0),
                'Service Coverage (%)': (cluster_populations.values / uncapped_populations.values * 100).round(1),
                'Capacity Utilization (%)': (cluster_populations.values / max_people_per_pump * 100).round(1)
            })
            st.dataframe(stats_df)
            
            total_served = cluster_populations.sum()
            total_population = uncapped_populations.sum()
            st.info(f"Total population served: {int(total_served):,} out of {int(total_population):,} " +
                   f"({total_served/total_population*100:.1f}% coverage)")
            
            # Add download button for results
            st.download_button(
                label="Download Results CSV",
                data=stats_df.to_csv(index=False),
                file_name="pump_stations.csv",
                mime="text/csv"
            )
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.error("Please check your input parameters and try again.")

if __name__ == "__main__":
    main()
