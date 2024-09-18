import requests
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Function to fetch and display available country codes from the World Bank API
def fetch_country_codes():
    print("Fetching all country codes from the World Bank API...")
    
    # Endpoint to fetch all countries
    endpoint = 'https://api.worldbank.org/v2/country?format=json&per_page=500'
    
    response = requests.get(endpoint)
    data = response.json()

    # The data is in the second part; it contains the countries
    countries = pd.DataFrame(data[1])
    
    # Select relevant columns: id and name (country code and country name)
    countries = countries[['id', 'name']]
    
    # Display countries in a labeled list
    print("\nAvailable Countries:\n")
    for index, row in countries.iterrows():
        print(f"{index+1}. {row['id']} - {row['name']}")
    
    return countries

# Function to fetch and display available indicators from the World Bank API
def fetch_indicators():
    print("Fetching all indicators from the World Bank API...")
    
    # Endpoint to fetch all indicators
    endpoint = 'https://api.worldbank.org/v2/indicator?format=json&per_page=99999'
    
    response = requests.get(endpoint)
    data = response.json()

    # The data is in two parts; the second part contains the indicators
    indicators = pd.DataFrame(data[1])
    
    # Select relevant columns: id and name (indicator code and description)
    indicators = indicators[['id', 'name']]
    
    # Display indicators in a labeled list
    print("\nAvailable Indicators:\n")
    for index, row in indicators.iterrows():
        print(f"{index+1}. {row['id']} - {row['name']}")
    
    return indicators
while True:
    while True:

        # Fetch and display the available indicators
        indicators = fetch_indicators()

        # Select an indicator based on user input from the available list
        selected_index = int(input("\nEnter the number corresponding to your desired indicator: ")) - 1
        selected_indicator = indicators.iloc[selected_index]['id']

        print(f"\nYou selected: {selected_indicator} - {indicators.iloc[selected_index]['name']}")
        input("Press enter to continue.")
        # Fetch and display the available country codes
        countries = fetch_country_codes()

        # Select a country based on user input from the available list
        selected_country_index = int(input("\nEnter the number corresponding to your desired country: ")) - 1
        selected_country = countries.iloc[selected_country_index]['id']
        print(f"\nYou selected: {selected_country} - {countries.iloc[selected_country_index]['name']}")
        input("Press enter to continue.")
        # Step 1: Fetch Data from the World Bank API
        country_code = selected_country  # Selected country code
        indicator_code = selected_indicator  # Selected indicator code
        endpoint = f'https://api.worldbank.org/v2/country/{country_code}/indicator/{indicator_code}?format=json&per_page=100'

        response = requests.get(endpoint)
        data = response.json()

        # Step 2: Inspect the data structure
        if len(data) < 2 or not isinstance(data[1], list):
            print(f"No data available for the selected country: {country_code} and indicator: {indicator_code}")
            input("Press enter to continue.")
            break
        
        df = pd.DataFrame(data[1])
        print("\nRaw Data Columns:", df.columns)

        # Step 3: Check and convert relevant columns
        if 'date' in df.columns and 'value' in df.columns:
            df = df[['date', 'value']]

            # Convert year to numeric and sort by year
            df['date'] = pd.to_numeric(df['date'])
            df = df.sort_values(by='date')

            # Drop rows with missing values
            df.dropna(inplace=True)
        else:
            print("Required columns ('date', 'value') not found in the data.")
            print("Available columns in the data:", df.columns)
            input("Press enter to continue.")
            break

        # Step 4: Preprocess Data
        X = df[['date']]  # Year as the only feature
        y = df['value']   # Death rate (or other target value)

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Step 5: Train the Model
        model = LinearRegression()
        model.fit(X_train, y_train)

        # Step 6: Make Predictions
        y_pred = model.predict(X_test)

        # Step 7: Evaluate the Model
        print('Mean Squared Error:', mean_squared_error(y_test, y_pred))
        print('R^2 Score:', r2_score(y_test, y_pred))

        # Step 8: Display the value for the Last Year Available
        last_year = df['date'].max()
        last_year_value = df[df['date'] == last_year]['value'].values[0]
        print(f"Value for the last available year ({last_year}): {last_year_value}")

        # Step 9: Make Future Predictions
        future_year = pd.DataFrame([[2025]], columns=['date'])
        future_pred = model.predict(future_year)
        print(f"Predicted value for 2025: {future_pred[0]}")
        input("Press enter to continue.")