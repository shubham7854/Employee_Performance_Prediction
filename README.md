Employee Productivity Prediction ApplicationThis is a Flask-based web application that leverages Machine Learning to predict employee productivity based on various input parameters. It provides a user-friendly interface for inputting data and receiving a productivity prediction, along with model performance metrics.FeaturesEmployee Productivity Prediction: Predicts employee productivity (Averagely Productive, Medium Productive, Highly Productive) based on user-provided inputs.Model Performance Metrics: Displays key evaluation metrics (MSE, MAE, R2 Score) of the trained machine learning model on its test data.Clean and Responsive UI: A modern, light-themed user interface built with pure HTML and CSS, designed to be responsive across various devices.Backend with Flask: A Python Flask backend handles data preprocessing, model inference, and serves the web pages.XGBoost Model: Utilizes an XGBoost Regressor for robust productivity prediction.Project Structure.


├── content/
│   └── garments_worker_productivity.csv  # Dataset for training the model
├── templates/
│   ├── home.html      # Home page of the application
│   ├── about.html     # About page with developer information
│   ├── predict.html   # Form for inputting prediction parameters
│   └── submit.html    # Page to display prediction results and metrics
├── model_xgb.pkl      # Trained XGBoost model (generated after first run if not present)
├── main.py            # Main Flask application logic
└── requirements.txt 


Setup and InstallationTo run this application locally, follow these steps:PrerequisitesPython 3.8+pip (Python package installer)1. Clone the Repository (or download files)If you have these files in a local directory, you can skip this step. Otherwise, imagine this project is in a Git repository:git clone <repository-url>
cd <repository-name>

2. Create a Virtual Environment (Recommended)It's good practice to use a virtual environment to manage dependencies:python -m venv venv

3. 
# On Windows:
.\venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate


3. Install DependenciesInstall the required Python libraries:pip install Flask pandas numpy scikit-learn xgboost
4. Place the DatasetEnsure you have the garments_worker_productivity.csv dataset inside a content/ directory in the root of your project. If you don't have it, you'll need to source it (e.g., from Kaggle) and place it there.5. Run the Applicationpython main.py


The application will start, and you should see output indicating that the Flask server is running, typically on http://127.0.0.1:8000/ or http://localhost:8000/.6. Access the ApplicationOpen your web browser and navigate to http://localhost:8000/.UsageHome Page (/): Provides an introduction to the application.About Page (/about): Contains information about the developer and technical projects.Predict Page (/predict): Fill out the form with the employee parameters and click "Predict Productivity".Result Page (/pred): Displays the predicted productivity level and the model's performance metrics.Model TrainingThe main.py script will automatically train the XGBoost model and save it as model_xgb.pkl if the file does not already exist when the application starts. It uses the garments_worker_productivity.csv dataset for this purpose.CustomizationStyling: All styling is done directly within the <style> tags in each HTML file. You can modify these CSS rules to change the appearance.Model: If you wish to use a different machine learning model, you can modify the train_and_save_models function in main.py and ensure your new model is saved as model_xgb.pkl (or update the loading path).Dataset: Replace garments_worker_productivity.csv in the content/ directory with your own dataset if you want to predict based on different data. Ensure your data columns match the expected input features.ContactFor any questions or feedback, please contact Shubham Prajapati at 220703shubham@gmail.com.
