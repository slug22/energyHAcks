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
        '#e6194B', '#3cb44b', '#ffe119', '#4363d8', '#f58231', 
        '#911eb4', '#42d4f4', '#f032e6', '#bfef45', '#fabed4',
        '#469990', '#dcbeff', '#9A6324', '#fffac8', '#800000'
    ]
    for idx, (_, row) in enumerate(data.iterrows()):
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
    
    for idx, (lon, lat) in enumerate(optimal_locations):
        folium.CircleMarker(location=[lat, lon], radius = 10, color = 'green',
                            fill = True, popup=f'Pump Station {idx+1}').add_to(m)
        # folium.Circle(location=[row['Y'], row['X']], radius=service_radii[idx] * 111000, color='blue', fill=False, popup=f'Service Radius: {service_radii[idx]:.2f} degrees').add_to(m)
    
    return m


atl_lat_lon = [33.78, -84.40]

def main():
    st.title('Geothermal Pump Placement Optimizer') # TODO: better title

    st.sidebar.header('Parameters')
    # TODO: default values, ranges???
    station_output = st.sidebar.number_input('Station Output (kW)', min_value=0.1, max_value=10.0, value=1.0, step=0.1)
    power_usage_per_person = st.sidebar.number_input('Power Usage per Person (kW)', min_value=0.1, max_value=10.0, value=1.0, step=0.1)

    # TODO: handle too big areas
    start_inp = st.sidebar.text_input('Southwest Coordinates (lat, lon)', '33.75, -84.25')
    end_inp = st.sidebar.text_input('Northeast Coordinates (lat, lon)', '34, -84')

    st.subheader('Optimal Pump Locations and Service Areas')
    
    data = load_density_data()
    
    
    with st.spinner():
        if st.sidebar.button("Optimize"):
            
            try:
                start_x, start_y = dp.coords_to_xy(start_inp)
                end_x, end_y = dp.coords_to_xy(end_inp)
            except ValueError:
                st.error('Invalid coordinates') # TODO: error message
                return
            
            filtered_data = dp.filter_data(data, start_x, end_x, start_y, end_y)
            population = dp.count_population(filtered_data)
            # TODO: output served population?
            power_required = population * power_usage_per_person
            n_clusters = int(power_required / station_output)
            
            n_clusters = 4 ## TODO: fix the above number being too big
            
            print("starting clustering")
            optimal_locations, labels = sc.calculate_clusters(filtered_data, n_clusters)
            print("Optimal locations:", optimal_locations)

            map = create_map(filtered_data, labels, optimal_locations)
            folium_static(map)


    # st.subheader('Station Statistics')
    # stats_df = pd.DataFrame({
    #     'Station': range(1, len(optimal_locations) + 1),
    #     'X': optimal_locations['X'],
    #     'Y': optimal_locations['Y'],
    #     'Service Radius (deg)': service_radii,
    #     'Population Served': [clustered_data[clustered_data['cluster'] == i]['Z'].sum() for i in range(len(optimal_locations))]
    # })
    # st.dataframe(stats_df)
    # st.download_button(label="Download Results CSV", data=stats_df.to_csv(index=False), file_name="pump_stations.csv", mime="text/csv")

if __name__ == "__main__":
    main()
