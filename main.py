import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import folium
from folium.plugins import HeatMap
from streamlit_folium import folium_static, st_folium
import geopandas as gpd
from scipy.stats import gaussian_kde

import data_processing as dp
import simon_cluster as sc

def load_density_data():
    csv_filename = "usa_pd_2020_1km_ASCII_XYZ.csv"
    data = pd.read_csv(csv_filename)
    return data

def create_map(data, labels, optimal_locations):
    # points at stations
    # heatmap for clusters

    center = [data['Y'].mean(), data['X'].mean()]
    m = folium.Map(location = center, zoom_start = 9)
    
    # # heat_data = [[row['Y'], row['X'], row['Z']] for idx, row in data.iterrows()]

    # simplified_data = data.iloc[::5]
    # heat_data = simplified_data[['Y', 'X', 'Z']].values
    # HeatMap(heat_data, blur = 30, min_opacity = 5).add_to(m)
    
    colors = [
        '#1a237e',  # Deep indigo
        '#880e4f',  # Dark magenta
        '#1b5e20',  # Forest green
        '#4a148c',  # Deep purple
        '#b71c1c',  # Dark red
        '#006064',  # Dark cyan
        '#3e2723',  # Dark brown
        '#264653',  # Dark slate blue
        '#7b1fa2',  # Rich purple
        '#004d40',  # Dark teal
        '#bf360c',  # Deep orange
        '#0d47a1',  # Strong blue
        '#2e7d32',  # Rich green
        '#827717',  # Dark olive
        '#4a148c'   # Deep violet
    ]
    for idx, (_, row) in enumerate(data.iterrows()):
        if idx % 2: # TODO: 2 const?
            continue
        cluster_id = labels[idx]
        color = colors[cluster_id % len(colors)]
        folium.CircleMarker(
            location=[row['Y'], row['X']],
            radius=3,  # Small dots for data points
            color=color,
            fill=True,
            fillOpacity=0.5,
            weight=1,
            #popup=f'Cluster {cluster_id + 1}<br>Population Density: {row["Z"]:.1f}'
        ).add_to(m)
    
    for idx, (lat, lon) in enumerate(optimal_locations):
        folium.CircleMarker(location=[lat, lon], radius = 10, color = 'green',
                            fill = True, popup=f'Pump Station {idx+1}').add_to(m)
        # folium.Circle(location=[row['Y'], row['X']], radius=service_radii[idx] * 111000, color='blue', fill=False, popup=f'Service Radius: {service_radii[idx]:.2f} degrees').add_to(m)
    
    return m

def calculate_efficiency(data, labels):
    populations = []
    
    num_clusters = len(set(labels))
    for i in range(num_clusters):
        labels_mask = labels == i
        cluster_data = data[labels_mask]
        populations.append(int(sc.calculate_population(cluster_data)))
    
    return populations

atl_lat_lon = [33.78, -84.40]

def main():
    st.title('Thermos')

    st.sidebar.header('Parameters')
    station_output = st.sidebar.number_input('Station Output (MW)', min_value=1.0, max_value=50.0, value=20.0, step=0.1)
    power_usage_per_person = st.sidebar.number_input('Power Usage per Person (kW)', min_value=0.1, max_value=5.0, value=0.5, step=0.1)

    # TODO: handle too big areas, and if south is norther than north etc
    start_inp = st.sidebar.text_input('Southwest Coordinates (lat, lon)', '33.75, -84.25')
    end_inp = st.sidebar.text_input('Northeast Coordinates (lat, lon)', '34, -84')

    st.subheader('Optimal Pump Locations and Service Areas')
    
    data = load_density_data()
    
    
    with st.spinner("Loading..."):
        if st.sidebar.button("Optimize"):
            
            try:
                start_x, start_y = dp.coords_to_xy(start_inp)
                end_x, end_y = dp.coords_to_xy(end_inp)
            except ValueError:
                st.error('Error: Invalid coordinates')
                return
            if start_x >= end_x or start_y >= end_y:
                st.error('Error: Invalid coordinates')
                return

            filtered_data = dp.filter_data(data, start_x, end_x, start_y, end_y)
            filtered_data = dp.remove_lower_densities(filtered_data)

            population = dp.count_population(filtered_data)
            # TODO: output served population?
            power_required = population * power_usage_per_person
            n_clusters = int(power_required / (station_output * 1000))
            print(n_clusters)
            
            print("starting clustering")
            optimal_locations, labels = sc.calculate_clusters(filtered_data, n_clusters)
            print("Optimal locations:", optimal_locations)

            map = create_map(filtered_data, labels, optimal_locations)
            folium_static(map)
            
            cluster_pops = calculate_efficiency(filtered_data, labels)
            print(cluster_pops)
            print(sum(cluster_pops) / population)
            
            # for i, cluster_pop in enumerate(cluster_pops):
            #     efficiency = (cluster_pop * power_usage_per_person) / (station_output * 1000)
            #     percent_efficiency = min(efficiency, 100.0) * 100##
            #     st.write(f"Station {i + 1} has {int(percent_efficiency)}% efficiency")

            # TODO: csv of optimal locations and populations served
            st.subheader('Station Statistics')
            stats_df = pd.DataFrame({
                'Station': range(1, len(optimal_locations) + 1),
                'Coordinates': [str(location) for location in optimal_locations],
                'Population Served': cluster_pops
            })
            st.dataframe(stats_df)
            st.download_button(label="Download Results CSV", data=stats_df.to_csv(index=False), file_name="pump_stations.csv", mime="text/csv")

if __name__ == "__main__":
    main()
