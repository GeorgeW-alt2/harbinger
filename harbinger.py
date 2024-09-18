import google.generativeai as genai
genai.configure(api_key="")

import os
import random
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import tkinter as tk
from tkinter import filedialog

# Prompt user to select a folder containing CSV files
folder_path = filedialog.askdirectory(title="Select a folder containing CSV files")

# Ask for prediction year
prediction_year = int(input("Prediction year: "))

# Specify the log file path where the messages will be saved
log_file_path = 'log.txt'

def append_message_to_file(filename, message):
    """
    Appends a message to a text file.
    
    Parameters:
    filename (str): The name of the file to which the message will be appended.
    message (str): The message to append to the file.
    """
    with open(filename, 'a') as file:
        file.write(message + '\n\n')

# Function to choose a random CSV file from a directory, searching recursively
def choose_random_file_recursive():
    if folder_path:
        csv_files = []
        for root, dirs, files in os.walk(folder_path):
            csv_files.extend([os.path.join(root, f) for f in files if f.endswith(".csv")])
        
        if csv_files:
            valid_files = [f for f in csv_files if os.path.getsize(f) > 0]
            if valid_files:
                return random.choice(valid_files)
    return None

# Function to fetch dataset from selected file
def fetch_dataset(file_path):
    return pd.read_csv(file_path)

# Function to choose columns for features and target
def choose_columns(df):
    feature_col = "Year"
    target_col = random.choice(df.columns.tolist()[1:])
    
    if feature_col in df.columns and target_col in df.columns:
        return feature_col, target_col
    else:
        return None, None

# Main loop
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

                # Convert feature to numeric and sort by feature
                df[feature_col] = pd.to_numeric(df[feature_col], errors='coerce')
                df = df.sort_values(by=feature_col)
                df.dropna(inplace=True)

                # Prepare data for training
                X = df[[feature_col]]  # Feature
                y = df[target_col]     # Target value
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                # Train the Linear Regression Model
                regression_model = LinearRegression()
                regression_model.fit(X_train, y_train)

                # Make Predictions and evaluate
                y_pred = regression_model.predict(X_test)
                last_value = df[df[feature_col] == df[feature_col].max()][target_col].values[0]

                # Predict for the future year
                future_year = pd.DataFrame([[prediction_year]], columns=[feature_col])
                future_pred = regression_model.predict(future_year)
                comparison = "increasing" if future_pred[0] > last_value else "decreasing"

                # Generate comment using Generative AI
                prompt = f"write a comment imitating an adult talking about {target_col} in year {prediction_year} is {comparison}."
                model = genai.GenerativeModel("gemini-1.5-flash")
                response = model.generate_content(prompt)
                generated_text = response.text

                # Print and save the generated comment
                print("USER:", prompt)
                print("AI:", generated_text)
                print()
                append_message_to_file(log_file_path, "USER:"+ prompt + "\nAI:" + generated_text)  # Save to log file
                
            else:
                False
                #print("Could not proceed due to invalid column names.")
        else:
            False
            #print("No file was selected.")
    except Exception as e:
        False
        #print(f"An error occurred: {e}")
