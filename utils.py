import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

@st.cache_data
def load_data(uploaded_file, sample_size=10000):
    """Loads CSV data into a Pandas DataFrame."""
    try:
        df = pd.read_csv(uploaded_file)
        if len(df) > sample_size:
            st.info(f"Dataset is large ({len(df)} rows). Randomly sampling {sample_size} rows for performance.")
            df = df.sample(n=sample_size, random_state=42).reset_index(drop=True)
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

@st.cache_data
def preprocess_data(df, features):
    """Handles missing values and scales features."""
    try:
        # Filter available features from dataset
        available_features = [f for f in features if f in df.columns]
        if not available_features:
            return None, None
            
        data_subset = df[available_features].dropna()
        
        # Scale features
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(data_subset)
        scaled_df = pd.DataFrame(scaled_features, columns=available_features, index=data_subset.index)
        
        return scaled_df, data_subset.index
    except Exception as e:
        st.error(f"Error preprocessing data: {e}")
        return None, None

@st.cache_data
def compute_wcss(scaled_df, max_clusters=10):
    """Computes Within-Cluster Sum of Square to find the Elbow method curve."""
    wcss = []
    for i in range(1, max_clusters + 1):
        kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42, n_init=10)
        kmeans.fit(scaled_df)
        wcss.append(kmeans.inertia_)
    return wcss

def perform_clustering(df, scaled_df, valid_indices, num_clusters=4):
    """Performs K-Means clustering and attaches results to the original DataFrame."""
    kmeans = KMeans(n_clusters=num_clusters, init='k-means++', random_state=42, n_init=10)
    clusters = kmeans.fit_predict(scaled_df)
    
    # Create a copy to attach clusters safely
    result_df = df.copy()
    
    # Initialize 'Cluster' column with NaN, then assign computed clusters to valid rows
    result_df['Cluster'] = np.nan
    result_df.loc[valid_indices, 'Cluster'] = clusters
    
    # Convert active clusters to visually appealing strings for plotting
    result_df.loc[valid_indices, 'Cluster Name'] = result_df.loc[valid_indices, 'Cluster'].apply(lambda x: f"Cluster {int(x)}")
    return result_df, clusters

def perform_pca_2d(scaled_df):
    """Reduces dimensional spaces to 2D for plotting."""
    pca = PCA(n_components=2)
    pca_features = pca.fit_transform(scaled_df)
    return pd.DataFrame(data=pca_features, columns=['PC1', 'PC2'], index=scaled_df.index)

def perform_pca_3d(scaled_df):
    """Reduces dimensional spaces to 3D for plotting."""
    pca = PCA(n_components=3)
    pca_features = pca.fit_transform(scaled_df)
    return pd.DataFrame(data=pca_features, columns=['PC1', 'PC2', 'PC3'], index=scaled_df.index)

def create_elbow_plot(wcss):
    """Generates an aesthetic Plotly line chart for the Elbow Method."""
    fig = go.Figure(data=go.Scatter(
        x=list(range(1, len(wcss) + 1)), 
        y=wcss, 
        mode='lines+markers',
        line=dict(color='#1DB954', width=3),
        marker=dict(size=10, color='#FF6B6B', line=dict(width=2, color='white'))
    ))
    fig.update_layout(
        title='Optimal Clusters: The Elbow Method',
        xaxis_title='Number of Clusters (K)',
        yaxis_title='WCSS (Within-Cluster Sum of Squares)',
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0.1)',
        font=dict(family="Outfit")
    )
    return fig

def create_2d_scatter(result_df, pca_df, valid_indices):
    """Generates a 2D Scatter plot of the clusters."""
    plot_data = pca_df.copy()
    plot_data['Cluster Name'] = result_df.loc[valid_indices, 'Cluster Name']
    
    # Add track names if available for hover text
    hover_data = {}
    if 'track_name' in result_df.columns:
        plot_data['Track'] = result_df.loc[valid_indices, 'track_name']
        hover_data = {'Track': True}
    elif 'name' in result_df.columns:
        plot_data['Track'] = result_df.loc[valid_indices, 'name']
        hover_data = {'Track': True}

    fig = px.scatter(
        plot_data, 
        x='PC1', 
        y='PC2', 
        color='Cluster Name',
        color_discrete_sequence=px.colors.qualitative.Pastel,
        hover_data=hover_data,
        title='2D Cluster Visualization'
    )
    fig.update_traces(marker=dict(size=8, opacity=0.8, line=dict(width=1, color='DarkSlateGrey')))
    fig.update_layout(
        template='plotly_dark', 
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0.1)',
        font=dict(family="Outfit")
    )
    return fig

def create_3d_scatter(result_df, pca_df, valid_indices):
    """Generates an interactive 3D Scatter plot of the clusters."""
    plot_data = pca_df.copy()
    plot_data['Cluster Name'] = result_df.loc[valid_indices, 'Cluster Name']
    
    # Add track names if available for hover text
    hover_name = None
    if 'track_name' in result_df.columns:
        plot_data['Track'] = result_df.loc[valid_indices, 'track_name']
        hover_name = 'Track'
    elif 'name' in result_df.columns:
        plot_data['Track'] = result_df.loc[valid_indices, 'name']
        hover_name = 'Track'

    fig = px.scatter_3d(
        plot_data, 
        x='PC1', 
        y='PC2', 
        z='PC3',
        color='Cluster Name',
        color_discrete_sequence=px.colors.qualitative.Pastel,
        hover_name=hover_name,
        title='3D Global Analysis map'
    )
    fig.update_traces(marker=dict(size=6, opacity=0.9))
    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        scene=dict(
            xaxis=dict(backgroundcolor="rgba(0,0,0,0.1)", gridcolor="rgba(255,255,255,0.2)"),
            yaxis=dict(backgroundcolor="rgba(0,0,0,0.1)", gridcolor="rgba(255,255,255,0.2)"),
            zaxis=dict(backgroundcolor="rgba(0,0,0,0.1)", gridcolor="rgba(255,255,255,0.2)"),
        ),
        font=dict(family="Outfit"),
        margin=dict(l=0, r=0, b=0, t=40)
    )
    return fig

def create_radar_chart(result_df, valid_indices, features, num_clusters):
    """Generates a radar chart analyzing feature averages per cluster."""
    clusters_data = result_df.loc[valid_indices]
    
    # Calculate means
    cluster_means = clusters_data.groupby('Cluster Name')[features].mean().reset_index()
    
    # Scale feature means strictly between 0 and 1 for the radar chart representation
    scaler = StandardScaler()
    scaled_values = scaler.fit_transform(cluster_means[features])
    # MinMaxScaler logic to make radar look good
    scaled_values = (scaled_values - scaled_values.min(axis=0)) / (scaled_values.max(axis=0) - scaled_values.min(axis=0) + 1e-10)
    
    fig = go.Figure()
    
    colors = px.colors.qualitative.Pastel
    
    for i, row in cluster_means.iterrows():
        cluster_name = row['Cluster Name']
        values = scaled_values[i].tolist()
        # Close the loop
        values.append(values[0])
        feature_labels = features + [features[0]]
        
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=feature_labels,
            fill='toself',
            name=cluster_name,
            line_color=colors[i % len(colors)],
            opacity=0.7
        ))
        
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=False, range=[0, 1]),
            angularaxis=dict(tickfont=dict(color='white', size=12))
        ),
        showlegend=True,
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(family="Outfit"),
        title="Cluster Centroids (Scaled)"
    )
    return fig
