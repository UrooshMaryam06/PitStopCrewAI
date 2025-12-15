import streamlit as st
import pandas as pd
import numpy as np
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

# ==============================================================================
# --- 1. CONFIGURATION & STYLING -----------------------------------------------
# ==============================================================================

# Define F1 Team Colors for consistent visualization
F1_TEAM_COLORS = {
    'Red Bull Racing': '#1E5BC6',   # Deep Blue (often represented as dark blue in broadcasts)
    'Ferrari': '#ED1C24',         # Ferrari Red
    'Mercedes': '#6CD3BF',        # Petronas Teal
    'McLaren': '#F58020',         # Papaya Orange
    'Alpine': '#2293D1',          # Alpine Blue
    'AlphaTauri': '#4E7C9B',      # Navy/Grey Blue
    'Aston Martin': '#2D826D',    # British Racing Green
    'Williams': '#37BEDD',        # Sky Blue
    'Alfa Romeo': '#B12039',      # Alfa Red/Maroon
    'Haas': '#B6BABD',            # White/Grey
    # Use generic colors for drivers if their team isn't listed (unlikely for F1 2022)
    'Other': '#777777'
}

# Use session state to store the initial dataframe
if 'ml_df_initial' not in st.session_state:
    st.session_state['ml_df_initial'] = None

# ==============================================================================
# --- 2. ML & DATA UTILITY FUNCTIONS -------------------------------------------
# ==============================================================================

@st.cache_data
def load_and_prepare_data(csv_path="f1_2022_ml_ready_dataset.csv"):
    """Loads, cleans, and engineers features for the ML model."""
    try:
        # Check if initial data is already loaded to avoid re-reading the file
        if st.session_state['ml_df_initial'] is not None:
            return st.session_state['ml_df_initial'].copy()

        # Assuming the CSV file is available in the execution environment
        ml_df = pd.read_csv(csv_path)
    except FileNotFoundError:
        st.error(f"Error: The file {csv_path} was not found. Please check the path.")
        st.stop()
    except Exception as e:
        st.error(f"Error loading CSV file: {e}")
        st.stop()
        
    real_2022_drivers = ml_df['Driver'].unique()
    ml_df = ml_df[ml_df['Driver'].isin(real_2022_drivers)].copy()
    
    # Critical sort for time-series features
    ml_df = ml_df.sort_values(by=['Race_Round']).reset_index(drop=True)

    # --- SIMULATION (For runnability when actual data is missing) ---
    np.random.seed(42)
    if 'Finish_Position' not in ml_df.columns:
        ml_df['Finish_Position'] = np.random.randint(1, 21, size=len(ml_df))
    if 'Qualifying_Position' not in ml_df.columns:
        ml_df['Qualifying_Position'] = np.random.randint(1, 21, size=len(ml_df))
    if 'Constructor' not in ml_df.columns:
        # Mock team name if missing
        ml_df['Constructor'] = ml_df['Driver'].apply(lambda x: x.split(' ')[-1] + ' Team') 

    # --- FEATURE ENGINEERING: Team Form ---
    ml_df['Team_Avg_Points_Last_5'] = (
        ml_df.groupby('Constructor')['Driver_Points'] 
        .transform(lambda x: x.shift(1).rolling(window=5, min_periods=1).mean())
    )
    ml_df['Team_Avg_Points_Last_5'] = ml_df['Team_Avg_Points_Last_5'].fillna(0)
    
    st.session_state['ml_df_initial'] = ml_df.copy()
    
    return ml_df

@st.cache_resource(hash_funcs={pd.DataFrame: lambda _: None})
def train_model(ml_df, n_estimators, max_depth):
    """Trains the LightGBM model with user-defined hyperparameters."""
    with st.spinner(f"Training Model: Estimators={n_estimators}, Depth={max_depth}..."):
        features = [
            'Driver_Points', 'Driver_Podiums', 'Sprint_Points', 'Sprint_Laps',
            'Circuit_Length_km', 'Circuit_Laps', 'Circuit_Turns', 'Circuit_DRS_Zones',
            'Qualifying_Position', 'Team_Avg_Points_Last_5'
        ]
        target = 'Finish_Position'

        ml_df_clean = ml_df.dropna(subset=features + [target])
        X = ml_df_clean[features]
        y = ml_df_clean[target].astype(int)

        lgb_model = lgb.LGBMClassifier(
            objective='multiclass', num_class=int(y.max()) + 1, n_estimators=n_estimators,
            max_depth=max_depth, learning_rate=0.05, random_state=42, n_jobs=-1, verbose=-1
        )
        lgb_model.fit(X, y)
    
    st.success("Model training complete.")
    return lgb_model, X

def predict_and_rank(model, X_data, full_df):
    """Predicts race positions, ranks them, and calculates podium/points."""
    # Ensure full_df is a fresh copy before modifications
    df = full_df.copy()
    
    y_proba = model.predict_proba(X_data)
    df['Predicted_Finish'] = np.argmax(y_proba, axis=1) + 1
    
    # Calculate the final ranked position based on the predicted score
    df['Race_Position'] = df.groupby('Race_Round')['Predicted_Finish'] \
                              .rank(method='first', ascending=True).astype(int)
                            
    # Create the predicted podium rank
    df['Predicted_Podium'] = df['Race_Position']
                                
    return df[df['Predicted_Podium'] <= 3].copy() # Return only podium finishers for specific graphs

# ==============================================================================
# --- 3. STREAMLIT APP LAYOUT --------------------------------------------------
# ==============================================================================

st.set_page_config(
    page_title="F1 Race Predictor (Pure ML)",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Theme Styling (Black and Red F1 Theme)
st.markdown(
    """
    <style>
    /* Main background */
    .stApp { background-color: #0d0d0d; color: #f1f1f1; }
    /* Sidebar background */
    .st-emotion-cache-1kyxreq, .st-emotion-cache-1jmve3k { background-color: #1a1a1a; }
    /* Headers and Titles */
    h1, h2, h3, .st-emotion-cache-1cpxdwj { color: #ff1e00; } /* F1 Red */
    /* Metric/Info Boxes */
    .stMetric { background-color: #1a1a1a; padding: 10px; border-radius: 10px; border-left: 5px solid #ff1e00; }
    </style>
    """,
    unsafe_allow_html=True
)

# --- Load Data and Train Model ---
ml_df_initial = load_and_prepare_data()

# --- Sidebar Controls ---
st.sidebar.markdown("# **🏎️ F1 ML Predictor**") # Race Car Icon!
st.sidebar.markdown("---")

# 1. Hyperparameter Tuning Feature
st.sidebar.header("🔧 ML Model Tuning")
st.sidebar.markdown("Adjust the parameters below. The model will retrain automatically.")

n_estimators = st.sidebar.slider("Number of Trees (`n_estimators`)", min_value=100, max_value=1000, step=100, value=500)
max_depth = st.sidebar.slider("Max Tree Depth (`max_depth`)", min_value=3, max_value=20, step=1, value=10)

# Train and Predict
lgb_model, X_data = train_model(ml_df_initial.copy(), n_estimators, max_depth)
# The predict_and_rank function now returns only podium finishers, 
# ensuring the Constructor column is available in the returned DataFrame.
podium_df = predict_and_rank(lgb_model, X_data, ml_df_initial.copy())


# 2. Race Selector & Analysis Controls
st.sidebar.header("📊 Race Selector & Analysis")

race_rounds = sorted(ml_df_initial['Race_Round'].unique())
selected_round = st.sidebar.selectbox(
    "Select Race Round to Analyze:",
    options=race_rounds,
    format_func=lambda x: f"Round {x} - {ml_df_initial[ml_df_initial['Race_Round'] == x]['Race_Name'].iloc[0]}"
)

all_drivers = sorted(ml_df_initial['Driver'].unique())
drivers_to_plot = st.sidebar.multiselect(
    "Select Drivers for Trend Plot:",
    options=all_drivers,
    default=['Max Verstappen', 'Lewis Hamilton', 'Charles Leclerc', 'Sergio Perez']
)

# --- Main Content ---
st.title("🏎️ F1 Race Predictor")
st.markdown("### Pure ML Prediction using LightGBM Classification and Feature Engineering")

# 1. Prediction for Selected Race
st.header(f"Race Prediction: {ml_df_initial[ml_df_initial['Race_Round'] == selected_round]['Race_Name'].iloc[0]}")

# Re-calculate full race results for the race table
full_race_df = ml_df_initial.copy()
full_race_df['Predicted_Finish'] = np.argmax(lgb_model.predict_proba(X_data), axis=1) + 1
full_race_df['Race_Position'] = full_race_df.groupby('Race_Round')['Predicted_Finish'] \
                                            .rank(method='first', ascending=True).astype(int)

race_results = full_race_df[full_race_df['Race_Round'] == selected_round] \
    .sort_values('Race_Position')[['Race_Position', 'Driver', 'Constructor', 'Predicted_Finish']] \
    .rename(columns={'Predicted_Finish': 'Predicted Score'})

# Highlight Podium finishers
def highlight_podium(s):
    # s is a series representing a row, so check if the value in the 'Race_Position' column is <= 3
    is_podium = s['Race_Position'] <= 3
    if is_podium:
        # Apply style to all cells in the row if it's a podium finish
        return ['background-color: #4a0505; color: white; font-weight: bold'] * len(s)
    return [''] * len(s)

st.dataframe(
    race_results.style.apply(highlight_podium, axis=1),
    use_container_width=True,
    height=500
)

# ==============================================================================
# --- 4. ADVANCED VISUALIZATIONS -----------------------------------------------
# ==============================================================================

st.header("Season Analysis")
col1, col2 = st.columns(2)

# --- Cumulative Predicted Points Chart (Col 1) ---
with col1:
    st.subheader("Cumulative Predicted Championship Points")
    
    # Map podium to points (1st=25, 2nd=18, 3rd=15)
    points_map = {1:25, 2:18, 3:15}
    podium_df['Predicted_Points'] = podium_df['Predicted_Podium'].map(points_map).fillna(0)

    # *** FIX APPLIED HERE: Include 'Constructor' in the groupby to retain the column. ***
    # Group by Driver, Constructor, and Race_Round, then calculate cumulative sum
    cumulative_points = (
        podium_df
        .groupby(['Driver', 'Constructor', 'Race_Round'])['Predicted_Points'].sum()
        .groupby(level=0).cumsum()
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor('#0d0d0d')
    ax.set_facecolor('#1a1a1a')
    ax.tick_params(colors='white')
    ax.spines['left'].set_color('white')
    ax.spines['bottom'].set_color('white')
    ax.xaxis.label.set_color('white')
    ax.yaxis.label.set_color('white')
    ax.title.set_color('#ff1e00')

    for driver in drivers_to_plot:
        driver_data = cumulative_points[cumulative_points['Driver'] == driver]
        
        # Get driver's team color, defaulting to grey
        # This line now works because 'Constructor' is present in driver_data
        team_color = F1_TEAM_COLORS.get(driver_data['Constructor'].iloc[0], '#777777') if not driver_data.empty else '#777777'

        ax.plot(driver_data['Race_Round'], driver_data['Predicted_Points'], 
                marker='o', label=driver, color=team_color, linewidth=2)

    ax.set_title("Cumulative Predicted Points Across Races")
    ax.set_xlabel("Race Round")
    ax.set_ylabel("Cumulative Points")
    ax.legend(facecolor='#1a1a1a', edgecolor='white', labelcolor='white')
    ax.grid(True, alpha=0.3, color='#333333')
    st.pyplot(fig)


# --- Predicted Podium Share Pie Chart (Col 2) ---
with col2:
    st.subheader("Predicted Podium Share")
    
    podium_counts = podium_df['Driver'].value_counts().head(8) # Top 8 only for cleaner pie chart
    
    # Helper DataFrame to map drivers to constructors for coloring
    driver_to_constructor = podium_df[['Driver', 'Constructor']].drop_duplicates().set_index('Driver')
    
    # Map colors to drivers for the pie chart
    driver_colors = [F1_TEAM_COLORS.get(driver_to_constructor.loc[driver, 'Constructor'], '#777777') 
                     for driver in podium_counts.index]

    fig, ax = plt.subplots(figsize=(8, 8))
    fig.patch.set_facecolor('#0d0d0d')

    wedges, texts, autotexts = ax.pie(
        podium_counts, 
        labels=podium_counts.index, 
        autopct='%1.1f%%', 
        startangle=90,
        colors=driver_colors,
        textprops={'color': 'white'}
    )
    
    # Ensure percentages are readable
    for autotext in autotexts:
        autotext.set_color('black')
        autotext.set_fontsize(10)
    
    ax.set_title("Predicted Podium Share - 2022 Season (Top 8)", color='#ff1e00')
    st.pyplot(fig)


# --- Race-by-Race Podium Position Scatter Plot (Full Width) ---
st.header("Race-by-Race Predicted Podium Positions")

if drivers_to_plot:
    fig, ax = plt.subplots(figsize=(14, 7))
    sns.set_style("whitegrid", {'axes.facecolor': '#1a1a1a', 'grid.color': '#333333'})
    
    # Configure Matplotlib for Dark Theme
    fig.patch.set_facecolor('#0d0d0d')
    ax.set_facecolor('#1a1a1a')
    ax.tick_params(colors='white')
    ax.spines['left'].set_color('white')
    ax.spines['bottom'].set_color('white')
    ax.xaxis.label.set_color('white')
    ax.yaxis.label.set_color('white')
    ax.title.set_color('#ff1e00')


    for driver in drivers_to_plot:
        driver_data = podium_df[podium_df['Driver'] == driver]
        
        if not driver_data.empty:
            # Determine driver's team color
            team_color = F1_TEAM_COLORS.get(driver_data['Constructor'].iloc[0], '#777777')
            
            # Scatter plot for predicted position
            ax.scatter(driver_data['Race_Round'], driver_data['Predicted_Podium'], 
                        color=team_color, s=150, alpha=0.9, label=driver, edgecolor='white', linewidth=1)
            
            # Line plot to show trend (optional)
            ax.plot(driver_data['Race_Round'], driver_data['Predicted_Podium'], 
                    color=team_color, linestyle='--', alpha=0.4, linewidth=1)

    ax.invert_yaxis() 
    ax.set_xticks(range(1, int(podium_df['Race_Round'].max()) + 1))
    ax.set_yticks([1, 2, 3])
    ax.set_yticklabels(['1st', '2nd', '3rd'])
    ax.set_xlabel("Race Round")
    ax.set_ylabel("Predicted Podium Position")
    ax.set_title("Race-by-Race Predicted Podium - 2022 Season")
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title="Driver", 
              facecolor='#1a1a1a', edgecolor='white', labelcolor='white')
    ax.grid(True, alpha=0.3, color='#333333')
    plt.tight_layout()
    st.pyplot(fig)
else:
    st.info("Please select at least one driver in the sidebar to view the trend plot.")