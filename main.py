# Version: 1.8.0 - Enhanced with Plotly visualizations, dataset statistics, and refined error handling
# This version reintroduces Plotly to visualize predicted data against training dataset statistics.

from flask import Flask, render_template, request
import numpy as np
import pickle
import io
import base64
import pandas as pd
import os # Import os module for path checking
import sys # Import sys for more robust error printing
import re # Import regex for 'Quarter' string extraction

# Plotly imports for visualizations
import plotly.express as px
import plotly.graph_objects as go
import json # To convert Plotly figures to JSON strings

# For model evaluation metrics
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
import xgboost as xgb # Only importing xgb as it's the chosen model

# --- Global Variables for Model, Metrics, Data, and Visualizations ---
model = None
GLOBAL_METRICS = None
GLOBAL_TRAINING_COLUMNS = None # To store the exact order of columns used in training
GLOBAL_DATASET_STATS = {} # To store mean, median, 75th, max for numeric columns for comparison

# Department mapping: String from UI to Integer for Model Input
DEPARTMENT_STRING_TO_INT_MAP = {
    'finishing': 0,
    'sweing': 1
}

def clean_and_prepare_data(data_df):
    """
    Performs comprehensive data cleaning and preprocessing for both training and prediction.
    Handles 'wip', 'date', 'month', 'department', and numeric column issues.
    """
    print("DEBUG: Starting data cleaning and preparation...")
    processed_df = data_df.copy() # Work on a copy to avoid modifying original DataFrame

    # Drop 'wip' column if it exists
    if 'wip' in processed_df.columns:
        processed_df = processed_df.drop(['wip'], axis=1)
        print("DEBUG: 'wip' column dropped.")
    else:
        print("DEBUG: 'wip' column not found, skipping drop.")

    # Convert 'date' to datetime and extract 'month'
    if 'date' in processed_df.columns:
        processed_df['date'] = pd.to_datetime(processed_df['date'], errors='coerce')
        processed_df['month'] = processed_df['date'].dt.month
        processed_df = processed_df.drop(['date'], axis=1)
        print("DEBUG: 'date' column processed and 'month' extracted.")
    else:
        print("DEBUG: 'date' column not found, skipping date processing.")


    # Handle 'quarter' column: Extract numeric part if it's a string like 'QuarterX'
    if 'quarter' in processed_df.columns:
        # Convert to string first to apply regex, then to numeric
        processed_df['quarter'] = processed_df['quarter'].astype(str).apply(lambda x: re.search(r'\d+', x).group(0) if re.search(r'\d+', x) else np.nan)
        processed_df['quarter'] = pd.to_numeric(processed_df['quarter'], errors='coerce')
        print("DEBUG: 'quarter' column processed for string values.")
    else:
        print("DEBUG: 'quarter' column not found.")


    # Department encoding: 'finishing' -> 0, 'sweing' -> 1
    if 'department' in processed_df.columns:
        processed_df['department'] = processed_df['department'].astype(str).apply(lambda x: DEPARTMENT_STRING_TO_INT_MAP.get(x.lower().strip(), np.nan))
        print("DEBUG: 'department' column encoded.")
    else:
        print("DEBUG: 'department' column not found.")


    # Identify columns that should be numeric (excluding 'actual_productivity' if present)
    numeric_cols = [col for col in processed_df.columns if col not in ['actual_productivity']]
    
    for col in numeric_cols:
        # Attempt to convert to numeric, coercing errors to NaN
        processed_df[col] = pd.to_numeric(processed_df[col], errors='coerce')
        
        # Handle NaN values: fill with mean
        if processed_df[col].isnull().any():
            mean_val = processed_df[col].mean()
            if pd.isna(mean_val): # If mean is NaN (e.g., column was all NaNs)
                mean_val = 0 # Fallback to 0 or another sensible default
            print(f"WARNING: NaN values found in column '{col}'. Filling with mean: {mean_val:.2f}", file=sys.stderr)
            processed_df[col] = processed_df[col].fillna(mean_val)
        
        # Handle Infinite values: replace with NaN then fill with mean
        if np.isinf(processed_df[col]).any():
            print(f"WARNING: Infinite values found in column '{col}'. Replacing with NaN then filling with mean.", file=sys.stderr)
            mean_val = processed_df[col].mean()
            if pd.isna(mean_val):
                mean_val = 0
            processed_df[col] = processed_df[col].replace([np.inf, -np.inf], np.nan).fillna(mean_val)
    
    print("DEBUG: Numeric columns processed for NaN/Inf values.")
    return processed_df

def calculate_dataset_stats(data_df):
    """
    Calculates min, 25th, mean, 50th, 75th, and max percentiles for numeric columns.
    """
    stats = {}
    # Include 'actual_productivity' in numeric_cols for its stats
    numeric_cols = [col for col in data_df.columns if pd.api.types.is_numeric_dtype(data_df[col])]
    
    for col in numeric_cols:
        if not data_df[col].empty and data_df[col].count() > 0: # Ensure column is not empty and has non-NaN values
            stats[col] = {
                'min': data_df[col].min(),
                'p25': data_df[col].quantile(0.25),
                'mean': data_df[col].mean(),
                'p50': data_df[col].quantile(0.50),
                'p75': data_df[col].quantile(0.75),
                'max': data_df[col].max()
            }
        else:
            # Default if no data or all NaNs
            stats[col] = {'min': 0, 'p25': 0, 'mean': 0, 'p50': 0, 'p75': 0, 'max': 0} 
    return stats


def train_and_save_models():
    """
    Trains the machine learning model (XGBoost) and saves it as a .pkl file.
    Also stores the order of training columns and calculates global metrics and dataset statistics.
    """
    global model, GLOBAL_METRICS, GLOBAL_TRAINING_COLUMNS, GLOBAL_DATASET_STATS

    data_file_path = './content/garments_worker_productivity.csv'
    print(f"DEBUG: Attempting to train and save models. Checking for data file: {data_file_path}")

    if not os.path.exists(data_file_path):
        print(f"ERROR: Data file '{data_file_path}' not found. Cannot train models.", file=sys.stderr)
        GLOBAL_TRAINING_COLUMNS = None
        GLOBAL_METRICS = None
        GLOBAL_DATASET_STATS = {}
        return # Exit function if data file is missing

    try:
        data = pd.read_csv(data_file_path)
        print("DEBUG: Raw training data loaded successfully.")

        # Clean and prepare the data
        processed_data_for_training = clean_and_prepare_data(data.copy()) # Use a copy to avoid modifying original data in place
        
        # Calculate and store global dataset statistics BEFORE splitting
        GLOBAL_DATASET_STATS = calculate_dataset_stats(processed_data_for_training.copy())
        print("DEBUG: Global dataset statistics calculated.")

        # Define features (x) and target (y)
        if 'actual_productivity' not in processed_data_for_training.columns:
            print("ERROR: 'actual_productivity' column not found in data after cleaning. Cannot train model.", file=sys.stderr)
            GLOBAL_TRAINING_COLUMNS = None
            GLOBAL_METRICS = None
            GLOBAL_DATASET_STATS = {}
            return

        x = processed_data_for_training.drop(['actual_productivity'], axis=1)
        y = processed_data_for_training['actual_productivity']

        # Store the column order for consistent prediction input
        GLOBAL_TRAINING_COLUMNS = x.columns.tolist()
        print(f"DEBUG: Training columns order set: {GLOBAL_TRAINING_COLUMNS}")

        # Split data for training and evaluation
        x_train_df, x_test_df, y_train_series, y_test_series = train_test_split(x, y, test_size=0.2, random_state=42)

        # Convert training features to NumPy array for model training
        x_train = x_train_df.to_numpy(dtype=np.float32) 
        print(f"DEBUG: Features x_train (NumPy array) shape: {x_train.shape}, dtype: {x_train.dtype}")

        # Train and save XGBoost model
        model_xgb = xgb.XGBRegressor(n_estimators=200, max_depth=5, learning_rate=0.1, random_state=42)
        model_xgb.fit(x_train, y_train_series) # Fit with NumPy array for features, Series for target
        with open('model_xgb.pkl', 'wb') as f:
            pickle.dump(model_xgb, f)
        print("DEBUG: model_xgb.pkl trained and saved.")

        # Set the global model to the newly trained XGBoost model
        model = model_xgb
        
        # Calculate and store global metrics
        y_pred_xgb_metrics = model.predict(x_test_df.to_numpy(dtype=np.float32)) # Predict using NumPy array
        GLOBAL_METRICS = {
            'mse': mean_squared_error(y_test_series, y_pred_xgb_metrics),
            'mae': mean_absolute_error(y_test_series, y_pred_xgb_metrics),
            'r2': r2_score(y_test_series, y_pred_xgb_metrics)
        }
        print("DEBUG: Global metrics calculated from training test set.")

    except Exception as e:
        print(f"ERROR: An error occurred during model training/saving: {e}", file=sys.stderr)
        GLOBAL_TRAINING_COLUMNS = None
        GLOBAL_METRICS = None
        GLOBAL_DATASET_STATS = {}


# --- Flask App Initialization ---
app = Flask(__name__)

# Attempt to load the model on application startup
print("DEBUG: Attempting to load model on application startup...")
try:
    with open('model_xgb.pkl', 'rb') as f:
        model = pickle.load(f)
    print("DEBUG: XGBoost model loaded successfully from 'model_xgb.pkl'.")

    # Load data to get column order and calculate metrics if model was pre-existing
    data_file_path_metrics = './content/garments_worker_productivity.csv'
    if not os.path.exists(data_file_path_metrics):
         print(f"WARNING: Data file '{data_file_path_metrics}' not found for metrics. Metrics and dataset stats will be unavailable.", file=sys.stderr)
         GLOBAL_TRAINING_COLUMNS = None
         GLOBAL_METRICS = None
         GLOBAL_DATASET_STATS = {}
    else:
        data_for_metrics = pd.read_csv(data_file_path_metrics)
        print("DEBUG: Data for metrics loaded successfully.")
        
        # Clean and prepare the data for metrics calculation
        processed_data_for_metrics = clean_and_prepare_data(data_for_metrics.copy())

        # Calculate dataset statistics from the full processed data
        GLOBAL_DATASET_STATS = calculate_dataset_stats(processed_data_for_metrics.copy())
        print("DEBUG: Global dataset statistics calculated from loaded data.")

        if 'actual_productivity' not in processed_data_for_metrics.columns:
            print("ERROR: 'actual_productivity' column not found in data for metrics. Cannot calculate metrics.", file=sys.stderr)
            GLOBAL_METRICS = None
            GLOBAL_TRAINING_COLUMNS = None
        else:
            x_metrics = processed_data_for_metrics.drop(['actual_productivity'], axis=1)
            y_metrics = processed_data_for_metrics['actual_productivity']
            
            # Set GLOBAL_TRAINING_COLUMNS here as well, in case training wasn't run
            GLOBAL_TRAINING_COLUMNS = x_metrics.columns.tolist()
            print(f"DEBUG: Training columns order set from metrics data: {GLOBAL_TRAINING_COLUMNS}")

            # Split data for global test sets
            _, x_test_metrics_df, _, y_test_metrics_series = train_test_split(x_metrics, y_metrics, test_size=0.2, random_state=42)

            if model and x_test_metrics_df is not None and y_test_metrics_series is not None:
                y_pred_xgb_metrics = model.predict(x_test_metrics_df.to_numpy(dtype=np.float32)) # Predict using NumPy array
                GLOBAL_METRICS = {
                    'mse': mean_squared_error(y_test_metrics_series, y_pred_xgb_metrics),
                    'mae': mean_absolute_error(y_test_metrics_series, y_pred_xgb_metrics),
                    'r2': r2_score(y_test_metrics_series, y_pred_xgb_metrics)
                }
                print("DEBUG: Global metrics loaded for performance display.")
            else:
                print("WARNING: Could not calculate global metrics after loading model (missing test data or model issue).", file=sys.stderr)

except FileNotFoundError:
    print("WARNING: Model 'model_xgb.pkl' not found. Attempting to train models now...", file=sys.stderr)
    train_and_save_models() # Call training function if model is not found
except Exception as e:
    print(f"ERROR: An unexpected error occurred during initial model loading: {e}", file=sys.stderr)
    model = None # Ensure model is None if loading fails
    GLOBAL_TRAINING_COLUMNS = None
    GLOBAL_METRICS = None
    GLOBAL_DATASET_STATS = {}

# --- Route Definitions ---
@app.route("/")
def home_page():
    """Renders the home page."""
    return render_template('home.html')

@app.route("/about")
def about_page():
    """Renders the about page."""
    return render_template('about.html')

@app.route("/predict")
def predict_page():
    """Renders the prediction input form page."""
    return render_template('predict.html')

@app.route("/pred", methods=['POST'])
def make_prediction():
    """
    Handles the prediction request, processes input, makes a prediction,
    and renders the result page with visualizations of predicted data vs. dataset stats.
    """
    # Check if the model was loaded or trained successfully
    if model is None:
        print("ERROR: Model is None, returning error error page.", file=sys.stderr)
        return render_template('submit.html',
                               prediction_text="Error: Prediction service is currently unavailable. Model not loaded or trained.",
                               raw_prediction_value="N/A",
                               graphs=[], # Empty list as no graphs will be generated
                               metrics=None,
                               dataset_stats={}) # Pass empty stats
    
    if GLOBAL_TRAINING_COLUMNS is None:
        print("ERROR: Training columns order not set. Cannot make prediction.", file=sys.stderr)
        return render_template('submit.html',
                               prediction_text="Error: Model configuration incomplete. Please ensure training data is available.",
                               raw_prediction_value="N/A",
                               graphs=[], # Empty list as no graphs will be generated
                               metrics=None,
                               dataset_stats={}) # Pass empty stats


    try:
        # Collect raw form data
        raw_form_data = {key: request.form[key].strip() for key in request.form}
        print(f"DEBUG: Raw form data received: {raw_form_data}", file=sys.stderr)

        # Create a DataFrame for the single prediction input
        # Ensure all expected columns are present, even if input is missing (will be NaN then filled by clean_and_prepare_data)
        input_data_for_df = {col: raw_form_data.get(col) for col in GLOBAL_TRAINING_COLUMNS}
        
        # Create a DataFrame from the single input row, ensuring correct column order
        input_df = pd.DataFrame([input_data_for_df])
        
        # Reorder columns to match training data
        input_df = input_df[GLOBAL_TRAINING_COLUMNS]

        print(f"DEBUG: Input DataFrame before cleaning:\n{input_df}", file=sys.stderr)

        # Apply the same cleaning and preprocessing as training data
        processed_input_df = clean_and_prepare_data(input_df)
        print(f"DEBUG: Input DataFrame after cleaning:\n{processed_input_df}", file=sys.stderr)

        # Convert to NumPy array
        features_array = processed_input_df.to_numpy(dtype=np.float32)
        
    except ValueError as e:
        print(f"ERROR: Invalid input data format during preprocessing: {e}", file=sys.stderr)
        return render_template('submit.html',
                               prediction_text=f"Invalid input provided. Please ensure all fields are filled correctly with numeric values. Detail: {e}",
                               raw_prediction_value="N/A",
                               graphs=[], # Empty list as no graphs will be generated
                               metrics=GLOBAL_METRICS,
                               dataset_stats=GLOBAL_DATASET_STATS)
    except KeyError as e:
        print(f"ERROR: Missing expected form data field during preprocessing: {e}", file=sys.stderr)
        return render_template('submit.html',
                               prediction_text=f"Missing form data. Please fill all required fields. Missing: {e}",
                               raw_prediction_value="N/A",
                               graphs=[], # Empty list as no graphs will be generated
                               metrics=GLOBAL_METRICS,
                               dataset_stats=GLOBAL_DATASET_STATS)
    except Exception as e:
        print(f"ERROR: An unexpected error occurred during input processing: {e}", file=sys.stderr)
        return render_template('submit.html',
                               prediction_text=f"An unexpected error occurred during data processing: {e}",
                               raw_prediction_value="N/A",
                               graphs=[], # Empty list as no graphs will be generated
                               metrics=GLOBAL_METRICS,
                               dataset_stats=GLOBAL_DATASET_STATS)

    # Check for NaN/Inf in the input features array before prediction
    if np.isnan(features_array).any():
        print("ERROR: NaN values detected in input features array before prediction.", file=sys.stderr)
        return render_template('submit.html',
                               prediction_text="Error: Input data contains missing or invalid numeric values. Please check your entries.",
                               raw_prediction_value="N/A",
                               graphs=[], # Empty list as no graphs will be generated
                               metrics=GLOBAL_METRICS,
                               dataset_stats=GLOBAL_DATASET_STATS)
    if np.isinf(features_array).any():
        print("ERROR: Infinite values detected in input features array before prediction.", file=sys.stderr)
        return render_template('submit.html',
                               prediction_text="Error: Input data contains infinite values. Please check your entries.",
                               raw_prediction_value="N/A",
                               graphs=[], # Empty list as no graphs will be generated
                               metrics=GLOBAL_METRICS,
                               dataset_stats=GLOBAL_DATASET_STATS)

    print(f"DEBUG: Final input features array shape: {features_array.shape}, dtype: {features_array.dtype}")
    print(f"DEBUG: Final input features array content: {features_array}", file=sys.stderr)

    try:
        prediction_value = model.predict(features_array)[0]
        print(f"DEBUG: Prediction successful: {prediction_value}")
    except Exception as e:
        print(f"ERROR: Error during model prediction: {e}", file=sys.stderr)
        return render_template('submit.html',
                               prediction_text=f"Error during prediction: {e}",
                               raw_prediction_value="N/A",
                               graphs=[], # Empty list as no graphs will be generated
                               metrics=GLOBAL_METRICS,
                               dataset_stats=GLOBAL_DATASET_STATS)

    # Determine productivity text based on prediction value
    if prediction_value <= 0.3:
        productivity_text = 'The employee is Averagely Productive.'
    elif 0.3 < prediction_value <= 0.8:
        productivity_text = 'The employee is Medium Productive.'
    else:
        productivity_text = 'The employee is Highly Productive.'

    # --- Plotly Graph Generation ---
    plotly_graphs = []

    # Columns to visualize (all input features + actual_productivity)
    # Ensure these keys exist in GLOBAL_DATASET_STATS
    columns_to_visualize = list(GLOBAL_TRAINING_COLUMNS) + ['actual_productivity']
    
    # Map raw form data to processed numeric values for visualization
    # This requires re-processing the single input row to get numerical values for comparison
    # We already have `processed_input_df` from earlier, which contains these.
    # For 'department', use the mapped integer value.
    processed_input_values = processed_input_df.iloc[0].to_dict()

    for col in columns_to_visualize:
        if col not in GLOBAL_DATASET_STATS:
            print(f"WARNING: Dataset statistics not available for column '{col}'. Skipping visualization.", file=sys.stderr)
            continue

        stats = GLOBAL_DATASET_STATS[col]
        
        # Determine the value to highlight in the graph
        if col == 'actual_productivity':
            highlight_value = prediction_value
            title_prefix = "Predicted Productivity"
        else:
            # Get the processed numeric value for the input feature
            highlight_value = processed_input_values.get(col, 0) # Default to 0 if not found

            # Special handling for 'department' to display string
            if col == 'department':
                # Reverse map the integer back to string for display
                department_int = int(highlight_value)
                department_string = next((k for k, v in DEPARTMENT_STRING_TO_INT_MAP.items() if v == department_int), "Unknown")
                title_prefix = f"Input {col.replace('_', ' ').title()} ({department_string})"
            else:
                title_prefix = f"Input {col.replace('_', ' ').title()}"

        # Create data for the bar chart
        categories = ['Min', '25th Percentile', 'Mean', '50th Percentile (Median)', '75th Percentile', 'Max']
        values = [stats['min'], stats['p25'], stats['mean'], stats['p50'], stats['p75'], stats['max']]
        colors = ['#636EFA', '#EF553B', '#00CC96', '#AB63FA', '#FFA15A', '#19D3F3'] # Plotly default colors

        # Add the input/predicted value to the data for highlighting
        # Find the appropriate position to insert it
        insert_idx = 0
        for i, val in enumerate(values):
            if highlight_value < val:
                insert_idx = i
                break
            insert_idx = i + 1 # If highlight_value is greater than all, place it at the end

        categories.insert(insert_idx, 'Your Value')
        colors.insert(insert_idx, '#FF6692') # Highlight color
        values.insert(insert_idx, highlight_value)
        

        fig = go.Figure(data=[go.Bar(x=categories, y=values, marker_color=colors)])

        fig.update_layout(
            title_text=f'{title_prefix} vs. Training Data Distribution',
            title_x=0.5, # Center title
            xaxis_title="Statistic",
            yaxis_title=col.replace('_', ' ').title(),
            template="plotly_white", # Clean white background
            margin=dict(l=20, r=20, t=50, b=20), # Adjust margins
            height=350 # Fixed height for consistency
        )

        plotly_graphs.append(json.loads(fig.to_json())) # Convert figure to JSON and append

    return render_template('submit.html',
                           prediction_text=productivity_text,
                           raw_prediction_value=prediction_value, # Pass the raw prediction value
                           graphs=plotly_graphs, # This will now contain Plotly JSONs
                           metrics=GLOBAL_METRICS, # Pass metrics here
                           dataset_stats=GLOBAL_DATASET_STATS) # Pass dataset stats here

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True) # debug=True for development, set to False for production
