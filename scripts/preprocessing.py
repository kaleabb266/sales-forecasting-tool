import pandas as pd
import logging
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler

#count the missing values and return the percentage of the messing values 
def count_missing_values(df):
    # logging.info("Counting the missing values")
    missing_count = df.isnull().sum()
    missing_percentage = (missing_count / len(df)) * 100
    message = f"Missing Values:\n{missing_count}\nMissing Percentage:\n{missing_percentage}"
    # logging.info(message)
    return missing_count , missing_percentage


# Function to handle missing values
def handle_missing_values(df):
    # Fill missing CompetitionDistance with a large value (no nearby competition)
    # logging.info("Fill missing CompetitionDistance with a large value (no nearby competition)")
    df['CompetitionDistance'].fillna(df['CompetitionDistance'].max() + 1, inplace=True)
    
    # Fill missing CompetitionOpenSinceMonth and CompetitionOpenSinceYear with median values
    # logging.info("Fill missing competitionOpenSinceMonth and CompetitionOpenScinceYear with median vlaues")
    df['CompetitionOpenSinceMonth'].fillna(df['CompetitionOpenSinceMonth'].median(), inplace=True)
    df['CompetitionOpenSinceYear'].fillna(df['CompetitionOpenSinceYear'].median(), inplace=True)
    
    # Fill missing Promo2SinceWeek and Promo2SinceYear (no Promo2 participation)
    # logging.info("Fill missing Promo2SinceWeek and Promo2SinceYear (no promo2 participation)")
    df['Promo2SinceWeek'].fillna(0, inplace=True)
    df['Promo2SinceYear'].fillna(0, inplace=True)
    
    return df

# Function to create new features based on competition and promo information
def feature_engineering(df):
    # Create a feature 'CompetitionOpenSince' as a measure of competition age in months
    # logging.info("Crate a feature 'CompetitionOpenSince' as a measure of competition age in months")
    df['CompetitionOpenSince'] = (df['Date'].dt.year - df['CompetitionOpenSinceYear']) * 12 + \
                                  (df['Date'].dt.month - df['CompetitionOpenSinceMonth'])
    df['CompetitionOpenSince'] = df['CompetitionOpenSince'].apply(lambda x: max(x, 0))  # Handle negative values

    # Create a feature for Promo2 duration
    # logging.info("Creating a feature for Promo2 duration")
    df['Promo2Duration'] = (df['Date'].dt.year - df['Promo2SinceYear']) * 52 + \
                           (df['Date'].dt.isocalendar().week - df['Promo2SinceWeek'])
    df['Promo2Duration'] = df['Promo2Duration'].apply(lambda x: max(x, 0))  # Handle negative values

    return df

# Function to encode categorical variables
def encode_categorical(df):
    # Label encode categorical columns
    label_encoders = {}
    categorical_columns = ['StateHoliday', 'StoreType', 'Assortment', 'PromoInterval']
    # logging.info("Encoding Columns 'StateHoliday', 'StoreType', 'Assortment', 'PromoInterval'")
    for col in categorical_columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))  # Handle NA values by converting to string
        label_encoders[col] = le
    
    return df, label_encoders



# Function to scale numerical features
def scale_numerical(df, numerical_columns):
    # logging.info("Scaleing the Numurical features")
    scaler = StandardScaler()
    df[numerical_columns] = scaler.fit_transform(df[numerical_columns])
    return df, scaler

import pandas as pd
import numpy as np

# Function to handle outliers based on IQR method
def handle_outliers(df, columns, method='cap'):
    """
    Handles outliers using the IQR method.
    
    Parameters:
        df (pd.DataFrame): The dataframe containing the data.
        columns (list): List of columns to check for outliers.
        method (str): The method to handle outliers: 'remove', 'cap', or 'impute'.
                      'remove' - removes outliers,
                      'cap' - caps outliers to the IQR boundaries,
                      'impute' - replaces outliers with the median.
    
    Returns:
        pd.DataFrame: The dataframe with outliers handled.
    """
    for col in columns:
        # Calculate Q1 (25th percentile) and Q3 (75th percentile) for the column
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        
        # Define outlier boundaries
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # Detect outliers
        outliers_lower = df[col] < lower_bound
        outliers_upper = df[col] > upper_bound
        
        if method == 'remove':
            # Remove outliers
            # logging.info("Removing Outliers")
            df = df[~(outliers_lower | outliers_upper)]
        
        elif method == 'cap':
            # Cap outliers to the lower and upper bounds
            # logging.info("Capping Outliers between lower bound and upper bound")
            df.loc[outliers_lower, col] = lower_bound
            df.loc[outliers_upper, col] = upper_bound
        
        elif method == 'impute':
            # Replace outliers with the median of the column
            # logging.info("Replacing Outliers with whe median of the column")
            median_value = df[col].median()
            df.loc[outliers_lower | outliers_upper, col] = median_value
        
    return df



# Function to extract features from the 'Date' column
def extract_date_features(df):
    # logging.info("Extracting Date Features from 'Date' to 'Year', 'Month', 'Day', 'WeekOfYear', 'DayOfYear', 'IsWeekend'")
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['Day'] = df['Date'].dt.day
    df['WeekOfYear'] = df['Date'].dt.isocalendar().week
    df['DayOfYear'] = df['Date'].dt.dayofyear
    df['IsWeekend'] = df['DayOfWeek'].apply(lambda x: 1 if x in [6, 7] else 0)  # Saturday = 6, Sunday = 7
    df['IsWeekday'] = df['DayOfWeek'].apply(lambda x: 1 if x in [1, 2, 3, 4, 5] else 0)  # Monday to Friday
    
    # Add feature for beginning, mid, and end of the month
    # logging.info("Extracting month position (Beginning, Mid, End of month)")
    df['MonthPosition'] = df['Day'].apply(lambda x: 'Beginning' if x <= 10 else ('Mid' if 10 < x <= 20 else 'End'))
    
    return df

# Function to calculate days to next holiday and days after a holiday
def calculate_holiday_features(df, holiday_column='StateHoliday'):
    """
    Optimized function to calculate number of days to the next holiday and number of days after the previous holiday.

    Args:
    df (pd.DataFrame): The input DataFrame containing the date and holiday information.
    holiday_column (str): The column name indicating holidays.

    Returns:
    pd.DataFrame: DataFrame with new features for days to next holiday and days after previous holiday.
    """
    # Ensure data is sorted by date
    df = df.sort_values('Date').reset_index(drop=True)

    # Identify rows with holidays
    holiday_dates = df.loc[df[holiday_column] != 0, 'Date'].values

    # Pre-calculate holiday-related features
    days_to_holiday = np.full(len(df), np.inf)
    days_after_holiday = np.full(len(df), np.inf)

    # If there are holidays, calculate the days to and from them
    if len(holiday_dates) > 0:
        # Calculate days to the next holiday
        for i in range(len(df)):
            future_holidays = holiday_dates[holiday_dates > df['Date'].iloc[i]]
            if len(future_holidays) > 0:
                days_to_holiday[i] = (future_holidays[0] - df['Date'].iloc[i]).days

        # Calculate days after the previous holiday
        for i in range(len(df)):
            past_holidays = holiday_dates[holiday_dates < df['Date'].iloc[i]]
            if len(past_holidays) > 0:
                days_after_holiday[i] = (df['Date'].iloc[i] - past_holidays[-1]).days

    # Add the results back to the dataframe
    df['DaysToHoliday'] = days_to_holiday
    df['DaysAfterHoliday'] = days_after_holiday

    return df


# Master preprocessing function
def preprocess_data(df):
    # logging.info("Preprocessing is starting")
    # Convert 'Date' column to datetime
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Handle missing values
    df = handle_missing_values(df)
    
    # Feature engineering
    # logging.info("the place it is not working at development ###################################################")
    df = feature_engineering(df)
    
    # Extract date-based features
    df = extract_date_features(df)

    # Calculate holiday-related features
    df = calculate_holiday_features(df, holiday_column='StateHoliday')

    # Encode categorical variables
    df, label_encoders = encode_categorical(df)

    # Handle outliers in 'Sales', 'Customers', and 'CompetitionDistance'
    outlier_columns = ['Sales', 'Customers', 'CompetitionDistance']
    df = handle_outliers(df, outlier_columns, method='cap')

    # Select numerical columns for scaling (excluding target variable 'Sales')
    numerical_columns = ['Customers', 'CompetitionDistance', 'CompetitionOpenSince', 'Promo2Duration']
    
    # Scale numerical columns
    df, scaler = scale_numerical(df, numerical_columns)
    
    return df, label_encoders, scaler

