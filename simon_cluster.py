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


def calculate_clusters(data: pd.DataFrame, n_clusters: int):
    # REPLACE THIS FUNCTION BODY
    weights = 1 / (data['Z'] + 1)
    weights = data['Z'] / data['Z'].sum()
    coords = data[['X', 'Y']].values

    kmeans = KMeans(n_clusters = n_clusters, random_state = 4)
    kmeans.fit(coords, sample_weight = weights)

    return kmeans.cluster_centers_, kmeans.labels_
