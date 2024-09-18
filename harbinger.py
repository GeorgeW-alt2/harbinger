import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import tkinter as tk
from tkinter import filedialog

# Function to let user choose a file
def choose_file():
    root = tk.Tk()
    root.withdraw()  # Hide the root window

    file_path = filedialog.askopenfilename(
        filetypes=[("CSV files", "*.csv")],
        title="Select a CSV file"
    )

    if file_path:
        print(f"Selected file: {file_path}")
        return file_path
    else:
        print("No file selected.")
        return None

# Function to fetch dataset from selected file
def fetch_dataset(file_path):
    print(f"Fetching dataset from {file_path}...")
    
    # Fetch dataset as DataFrame
    df = pd.read_csv(file_path)
    
    print("\nDataset Columns:", df.columns)
    
    return df

# Function to let user choose columns for features and target
def choose_columns(df):
    print("\nAvailable columns in the dataset:")
    print(df.columns.tolist())  # Convert to list for readability
    
    feature_col = input("Enter the name of the column to use as the feature (e.g., 'year'): ")
    target_col = input("Enter the name of the column to use as the target (e.g., 'co2'): ")
    
    if feature_col in df.columns and target_col in df.columns:
        return feature_col, target_col
    else:
        print("Invalid column names provided.")
        return None, None

file_path = []
while True:
    # Let user choose file
    if input("Select CSV file? [y/n]:").lower() =="y":
        
        file_path = choose_file()
    
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

            # Train the Model
            model = LinearRegression()
            model.fit(X_train, y_train)

            # Make Predictions
            y_pred = model.predict(X_test)

            # Evaluate the Model
            print('Mean Squared Error:', mean_squared_error(y_test, y_pred))
            print('R^2 Score:', r2_score(y_test, y_pred))

            # Display the value for the Last Year Available
            last_value = df[df[feature_col] == df[feature_col].max()][target_col].values[0]
            print(f"{target_col} value for the last available {feature_col} ({df[feature_col].max()}): {last_value}")

            # Make Future Predictions
            year = int(input("Year for prediction: "))
            future_year = pd.DataFrame([[year]], columns=[feature_col])
            future_pred = model.predict(future_year)
            print(f"Predicted {target_col} value for year: {future_pred[0]}")
        else:
            print("Could not proceed due to invalid column names.")
    else:
        print("No file was selected.")