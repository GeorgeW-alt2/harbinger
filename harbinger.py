import os
import random
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import tkinter as tk
from tkinter import filedialog
folder_path = filedialog.askdirectory(title="Select a folder containing CSV files")

prediction_year = int(input("Prediction year: "))

# Function to choose a random CSV file from a directory, searching recursively
def choose_random_file_recursive():
    root = tk.Tk()
    root.withdraw()  # Hide the root window

    
    if folder_path:
        # Recursively search for all CSV files in the directory and subdirectories
        csv_files = []
        for root, dirs, files in os.walk(folder_path):
            csv_files.extend([os.path.join(root, f) for f in files if f.endswith(".csv")])

        if csv_files:
            # Filter files that are over 0KB
            valid_files = [f for f in csv_files if os.path.getsize(f) > 0]
            
            if valid_files:
                # Randomly select a valid CSV file
                random_file = random.choice(valid_files)
                #print(f"Selected random file: {random_file} (Size: {os.path.getsize(random_file)} bytes)")
                return random_file
            else:
                #print("No valid CSV files (over 0KB) found in the selected folder and subfolders.")
                return None
        else:
            #print("No CSV files found in the selected folder and subfolders.")
            return None
    else:
        #print("No folder selected.")
        return None

# Function to fetch dataset from selected file
def fetch_dataset(file_path):
    #print(f"Fetching dataset from {file_path}...")
    
    # Fetch dataset as DataFrame
    df = pd.read_csv(file_path)
    
    #print("\nDataset Columns:", df.columns)
    
    return df

# Function to let user choose columns for features and target
def choose_columns(df):
    #print("\nAvailable columns in the dataset:")
    #print(df.columns.tolist())  # Convert to list for readability
    
    feature_col = "Year"
    target_col = random.choice(df.columns.tolist()[1:])
    
    if feature_col in df.columns and target_col in df.columns:
        return feature_col, target_col
    else:
        #print("Invalid column names provided.")
        return None, None

file_path = []

while True:
    
    file_path = choose_random_file_recursive()
    try:
        if file_path:

            # Fetch dataset
            df = fetch_dataset(file_path)

            # Let user choose columns
            feature_col, target_col = choose_columns(df)

            if feature_col and target_col:
                df = df[[feature_col, target_col]]

                # Convert feature to numeric if necessary and sort by feature
                df[feature_col] = pd.to_numeric(df[feature_col], errors='coerce')
                df = df.sort_values(by=feature_col)

                # Drop rows with missing values
                df.dropna(inplace=True)

                # Preprocess Data
                X = df[[feature_col]]  # Feature
                y = df[target_col]     # Target value

                # Split the data into training and testing sets
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                # Initialize the Linear Regression Model (placed outside the block to ensure proper access)
                regression_model = LinearRegression()
                
                # Train the Linear Regression Model
                regression_model.fit(X_train, y_train)

                # Make Predictions
                y_pred = regression_model.predict(X_test)

                # Evaluate the Model
                #print('Mean Squared Error:', mean_squared_error(y_test, y_pred))
                #print('R^2 Score:', r2_score(y_test, y_pred))

                # Display the value for the Last Year Available
                last_value = df[df[feature_col] == df[feature_col].max()][target_col].values[0]

                # Make Future Predictions
                year = prediction_year
                future_year = pd.DataFrame([[year]], columns=[feature_col])
                future_pred = regression_model.predict(future_year)

                # Determine if the future prediction is higher or lower
                comparison = "higher" if future_pred[0] > last_value else "lower"

                print(f"{target_col} in year {year} went {comparison}.")
                
            else:
                print("Could not proceed due to invalid column names.")
        else:
            print("No file was selected.")
    except:
        False