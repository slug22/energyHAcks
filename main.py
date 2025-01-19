import streamlit as st
import pandas as pd
import numpy as np
import folium
from streamlit_folium import folium_static, st_folium
import plotly.graph_objects as go

import data_processing as dp
import simon_cluster as sc

# streamlit_folium
import warnings
warnings.filterwarnings("ignore", category = DeprecationWarning)

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
        '#1f77b4',  # Strong blue
        '#d62728',  # Crimson red
        '#2ca02c',  # Forest green
        '#9467bd',  # Medium purple
        '#8c564b',  # Brown
        '#e377c2',  # Deep pink
        '#17becf',  # Teal
        '#ff7f0e',  # Dark orange
        '#7f3b08',  # Dark brown
        '#d4436f',  # Dark raspberry
        '#2d004b',  # Deep purple
        '#084081',  # Navy blue
        '#b30000',  # Dark red
        '#006837',  # Deep green
        '#662506'   # Rust brown
    ]
    for idx, (_, row) in enumerate(data.iterrows()):
        if idx % 1: # TODO: 2 const?
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
                            fill = True, popup = f'Pump Station {idx+1}').add_to(m)
        
    return m


COAL_CO2_PER_MWH = 820  # kg CO2 per MWh for coal power
GAS_CO2_PER_MWH = 490   # kg CO2 per MWh for natural gas
GEOTHERMAL_CO2_PER_MWH = 38  # kg CO2 per MWh for geothermal

def calculate_environmental_impact(total_power_mw, hours=8760):  # 8760 hours in a year
    total_mwh = total_power_mw * hours
    
    # Calculate CO2 savings compared to fossil fuels (in metric tons)
    coal_savings = (COAL_CO2_PER_MWH - GEOTHERMAL_CO2_PER_MWH) * total_mwh / 1000
    gas_savings = (GAS_CO2_PER_MWH - GEOTHERMAL_CO2_PER_MWH) * total_mwh / 1000
    
    # Convert to equivalent metrics
    trees_equivalent = int(coal_savings * 0.12)  # Rough estimate: 1 tree absorbs 8.3 kg CO2 per year
    cars_equivalent = int(coal_savings * 0.2)    # Rough estimate: 1 car emits 5 metric tons CO2 per year
    
    return {
        'coal_savings': int(coal_savings),
        'gas_savings': int(gas_savings),
        'trees_equivalent': trees_equivalent,
        'cars_equivalent': cars_equivalent
    }

atl_lat_lon = [33.78, -84.40]

def main():
    st.title('Thermos')
    st.subheader('Sustainably Optimizing Geothermal Pumps')

    st.image("https://i.ytimg.com/vi/Hu3I0LfgMiM/maxresdefault.jpg",
         caption="Community geothermal heat pump system. Credit: Eversource")

    st.sidebar.header('Parameters')
    station_output = st.sidebar.number_input('Station Output (MW)', min_value=1.0, max_value=50.0, value=20.0, step=0.1)
    power_usage_per_person = st.sidebar.number_input('Power Usage per Person (kW)', min_value=0.1, max_value=5.0, value=0.5, step=0.1)

    start_inp = st.sidebar.text_input('Southwest Coordinates (lat, lon)', '33.75, -84.50')
    end_inp = st.sidebar.text_input('Northeast Coordinates (lat, lon)', '34.00, -84.25')
    
    data = load_density_data()

    if st.sidebar.button("Optimize"):
        with st.spinner("Loading..."):
            
            st.subheader('Optimal Geothermal Pump Locations')
            
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
            power_required = population * power_usage_per_person
            n_clusters = int(power_required / (station_output * 1000))
            print("Using", n_clusters, "clusters")
            
            print("starting clustering")
            optimal_locations, labels = sc.calculate_clusters(filtered_data, n_clusters)
            print("finished clustering")
            
            map = create_map(filtered_data, labels, optimal_locations)
            folium_static(map)
            
            cluster_pops = dp.calculate_efficiency(filtered_data, labels)
            print(cluster_pops)
            print(f"Population reached: {int(sum(cluster_pops) / population * 100)}%")
            
            utilization_rates = []
            for i, cluster_pop in enumerate(cluster_pops):
                efficiency = (cluster_pop * power_usage_per_person) / (station_output * 1000)
                percent_efficiency = int(min(efficiency * 100.0, 100.0))
                utilization_rates.append(int(percent_efficiency))

            # Environmental impact
            if 'utilization_rates' in locals():
                total_power = station_output * len(optimal_locations)
                impact = calculate_environmental_impact(total_power)
                
                st.subheader('Environmental Impact')
                
                # Create three columns for metrics
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Annual CO₂ Savings vs Coal",
                            f"{impact['coal_savings']:,} tons",
                            "Compared to coal power plants")
                    
                with col2:
                    st.metric("Equivalent to Trees Planted",
                            f"{impact['trees_equivalent']:,} trees",
                            "Annual CO₂ absorption")
                    
                with col3:
                    st.metric("Cars Removed",
                            f"{impact['cars_equivalent']:,} cars",
                            "Annual emissions equivalent")
                
                # Add comparison chart
                fig = go.Figure()
                
                energy_sources = ['Coal', 'Natural Gas', 'Geothermal']
                emissions = [COAL_CO2_PER_MWH, GAS_CO2_PER_MWH, GEOTHERMAL_CO2_PER_MWH]
                colors = ['#cf6679', '#ffd740', '#69f0ae']
                
                fig.add_trace(go.Bar(
                    x=energy_sources,
                    y=emissions,
                    marker_color=colors,
                    text=emissions,
                    textposition='auto',
                ))
                
                fig.update_layout(
                    title='CO₂ Emissions by Energy Source',
                    yaxis_title='kg CO₂ per MWh',
                    showlegend=False,
                    height=400,
                    margin=dict(t=30, b=0)
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                st.info("""
                💡 Geothermal power stations significantly reduce carbon emissions compared to traditional fossil fuel power plants. 
                The environmental benefits shown above represent the annual impact of switching to geothermal energy in your selected region.
                """)
                
                st.markdown(
                    '[Learn more about geothermal energy](https://www.eversource.com/content/residential/save-money-energy/clean-energy-options/geothermal-energy/geothermal-pilot-framingham)',
                    unsafe_allow_html=True
                )
                
            # Data download
            st.subheader('Station Statistics')
            stats_df = pd.DataFrame({
                'Station': range(1, len(optimal_locations) + 1),
                'Coordinates (lat, lon)': [tuple(np.round(location, 3)) for location in optimal_locations],
                'Population Served': cluster_pops,
                'Utilization Rate (%)': utilization_rates
            })
            st.dataframe(stats_df, hide_index = True)
            st.download_button(label = "Download Station Data as CSV",
                               data = stats_df.to_csv(index=False),
                               file_name = "pump_stations.csv", mime = "text/csv")

if __name__ == "__main__":
    main()
