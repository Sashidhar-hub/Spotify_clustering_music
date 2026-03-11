# 🎵 Spotify Clustering Music Analysis

![Streamlit UI Concept](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)

A stunning, glassmorphic Streamlit web application that utilizes Machine Learning (K-Means Clustering) and dimensionality reduction (PCA) to analyze and visualize Spotify track features in interactive 2D and 3D spaces. 

## 🚀 Features

*   **Premium Interactive UI:** Built with custom CSS, glassmorphism design traits, and smooth animations.
*   **Dynamic Data Upload:** Handles any structured `spotify_tracks.csv` dataset directly from the sidebar.
*   **Smart Feature Selection:** Choose combinations of numeric audio features (like `danceability`, `energy`, `tempo`) to drive the AI clustering.
*   **Automated Sampling:** Intelligently scales large datasets down to 10,000 rows to ensure blazing-fast computation without sacrificing statistical accuracy.
*   **The Elbow Method:** Automatically computes and graphs the Within-Cluster Sum of Squares (WCSS) to help you determine the optimal `K` clusters.
*   **Radar Centroid Analysis:** Understand the "audio fingerprint" of each AI-generated cluster using comparative radar maps.
*   **Interactive 3D/2D Scatter Plots:** Explore the global distribution of music through Plotly-powered Principal Component Analysis (PCA) models.

## 🛠️ Technology Stack

*   **Frontend:** [Streamlit](https://streamlit.io/) with raw CSS injection (`style.css`).
*   **Data Processing:** `pandas`, `numpy`
*   **Machine Learning:** `scikit-learn` (StandardScaler, PCA, KMeans)
*   **Data Visualization:** `plotly.express`, `plotly.graph_objects`

## 📦 Installation & Setup

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/Sashidhar-hub/Spotify_clustering_music.git
    cd Spotify_clustering_music
    ```

2.  **Install requirements:**
    Ensure you have Python 3.8+ installed. Run the following command to grab all necessary packages:
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the application locally:**
    ```bash
    python -m streamlit run app.py
    ```

4.  **Upload the Data:**
    Once the local server boots up (usually at `http://localhost:8501`), drag and drop your `spotify_tracks.csv` into the sidebar to begin analysis!

## 📂 Project Structure
```text
Spotify_clustering_music/
│
├── app.py               # Main Streamlit application and UI layout
├── utils.py             # Backend logic containing data processing, clustering, and charting functions
├── style.css            # Custom glassmorphic CSS styling
├── requirements.txt     # Python package dependencies
├── notebook.ipynb       # Original Colab notebook research reference
└── README.md            # You are here
```

## 🤝 Contributing
Contributions, issues, and feature requests are welcome! Feel free to check the issues page or submit a pull request if you want to improve the visualization modes or the ML logic.