import streamlit as st
import pandas as pd
import pathlib

# Import utility functions from utils.py
from utils import (
    load_data, 
    preprocess_data, 
    compute_wcss, 
    perform_clustering, 
    perform_pca_2d, 
    perform_pca_3d,
    create_elbow_plot, 
    create_2d_scatter, 
    create_3d_scatter, 
    create_radar_chart
)

# Page configuration MUST be the first Streamlit command
st.set_page_config(
    page_title="Spotify clustering project",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded"
)

def load_css(file_path):
    """Loads external CSS for custom styling."""
    try:
        with open(file_path, "r") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        st.warning(f"CSS file not found at {file_path}")

# Load custom CSS
css_path = pathlib.Path(__file__).parent / "style.css"
load_css(css_path)

def main():
    # Hero Section
    st.markdown("<h1 class='title-text'>Spotify Track Clusters</h1>", unsafe_allow_html=True)
    st.markdown("<p class='subtitle-text'>Discovering patterns in global music through advanced K-Means clustering and Plotly 3D visualisations.</p>", unsafe_allow_html=True)

    # Sidebar
    st.sidebar.header("Data Source")
    uploaded_file = st.sidebar.file_uploader("Upload spotify_tracks.csv", type=["csv"])
    
    if uploaded_file is None:
        st.markdown(
            '''
            <div class='glass-container' style='text-align: center; margin-top: 50px;'>
                <h3 style='color: #FF6B6B;'>No Dataset Selected</h3>
                <p>Please upload the <b>spotify_tracks.csv</b> file via the sidebar to start exploring the clusters.</p>
                <div style='font-size: 5rem; margin-top: 1rem;'>📊</div>
            </div>
            ''', 
            unsafe_allow_html=True
        )
        return

    # Load Data
    with st.spinner('Loading track records...'):
        df = load_data(uploaded_file)
        
    if df is None:
        return
        
    st.sidebar.success(f"Dataset Loaded: {len(df)} rows")
    
    # Feature Selection Configuration
    st.sidebar.header("Clustering Parameters")
    
    # Provide commonly used spotify features for default selection
    default_features = ['danceability', 'energy', 'loudness', 'tempo', 'valence']
    available_columns = df.select_dtypes(include=['number']).columns.tolist()
    
    selected_features = st.sidebar.multiselect(
        "Select features for K-Means Clustering:",
        options=available_columns,
        default=[f for f in default_features if f in available_columns]
    )
    
    if len(selected_features) < 2:
        st.warning("Please select at least two numeric features for clustering.")
        return

    num_clusters = st.sidebar.slider("Number of Clusters (K):", min_value=2, max_value=10, value=4, step=1)
    
    # Preprocessing
    with st.spinner('Scaling features...'):
        scaled_df, valid_indices = preprocess_data(df, selected_features)
        
    if scaled_df is None:
        return
        
    # Generate Tabs
    tab1, tab2, tab3 = st.tabs(["📚 Data Overview", "📈 Cluster Analysis", "🌌 3D/2D Visualizations"])

    with tab1:
        st.markdown("<h3 style='margin-bottom: 20px;'>Raw Dataset Preview</h3>", unsafe_allow_html=True)
        st.dataframe(df.head(100), use_container_width=True)
        
        st.markdown("<h3 style='margin-bottom: 20px; margin-top: 40px;'>Feature Distributions</h3>", unsafe_allow_html=True)
        st.markdown("<p>Summary statistics of the selected numeric features used for AI clustering.</p>", unsafe_allow_html=True)
        st.dataframe(scaled_df.describe(), use_container_width=True)

    with tab2:
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("<h3>The Elbow Method</h3>", unsafe_allow_html=True)
            st.markdown("<p style='color: #b3b3b3; font-size: 0.9em;'>This helps find the optimal number of clusters by minimizing the within-cluster sum of squares (WCSS).</p>", unsafe_allow_html=True)
            with st.spinner('Computing WCSS...'):
                wcss = compute_wcss(scaled_df, max_clusters=10)
                elbow_fig = create_elbow_plot(wcss)
                st.plotly_chart(elbow_fig, use_container_width=True)
                
        with col2:
            st.markdown("<h3>Radar Map / Centroids</h3>", unsafe_allow_html=True)
            st.markdown("<p style='color: #b3b3b3; font-size: 0.9em;'>Observe the characteristics of each cluster across multiple feature dimensions.</p>", unsafe_allow_html=True)
            with st.spinner('Running K-Means...'):
                result_df, clusters = perform_clustering(df, scaled_df, valid_indices, num_clusters)
                radar_fig = create_radar_chart(result_df, valid_indices, selected_features, num_clusters)
                st.plotly_chart(radar_fig, use_container_width=True)
                
        # Show cluster sizes
        st.markdown("<h3>Cluster Demographics</h3>", unsafe_allow_html=True)
        cluster_counts = result_df.loc[valid_indices, 'Cluster Name'].value_counts().reset_index()
        cluster_counts.columns = ['Cluster', 'Count']
        st.dataframe(cluster_counts.set_index('Cluster'), use_container_width=True)

    with tab3:
        st.markdown("<h3>Global Dimensionality Reduction</h3>", unsafe_allow_html=True)
        st.markdown("<p style='margin-bottom: 20px; color: #b3b3b3;'>We use Principal Component Analysis (PCA) to map multi-dimensional music features into interactive 2D and 3D spaces.</p>", unsafe_allow_html=True)
        
        # Ensure we have the result_df computed
        if 'result_df' not in locals():
             result_df, clusters = perform_clustering(df, scaled_df, valid_indices, num_clusters)
             
        # Calculate PCA only when tab is viewed to save time
        col1, col2 = st.columns([1, 1])
        with col1:
            with st.spinner('Generating 2D Layout...'):
                pca_2d_df = perform_pca_2d(scaled_df)
                fig_2d = create_2d_scatter(result_df, pca_2d_df, valid_indices)
                st.plotly_chart(fig_2d, use_container_width=True)
                
        with col2:
            if len(selected_features) >= 3:
                with st.spinner('Generating 3D Space...'):
                    pca_3d_df = perform_pca_3d(scaled_df)
                    fig_3d = create_3d_scatter(result_df, pca_3d_df, valid_indices)
                    st.plotly_chart(fig_3d, use_container_width=True)
            else:
                st.info("Select at least 3 features in the sidebar to view the interactive 3D Model.")

if __name__ == "__main__":
    main()
