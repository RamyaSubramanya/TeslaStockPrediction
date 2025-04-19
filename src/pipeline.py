import pandas as pd

#read the data 
def load_and_prepare(file_path):
    """
    Load CSV data, convert date column, set index, and extract month/year.
    Returns time-series indexed DataFrames.
    """
    data = pd.read_csv(file_path)
 
    #format date column from object to date format
    data['Date'] = pd.to_datetime(data['Date'], format='%Y-%m-%d', errors='coerce')

    #copy of main data 
    time_series_data = data.copy()

    #set date column as index 
    time_series_data.set_index('Date', inplace=True)

    return time_series_data

def split_data(time_series_data, split_ratio=0.7):
    """
    Split time-series data into training and testing sets based on a split ratio.
    
    Parameters: 
    time_series_data (pd.DataFrame): The time-series data indexed by Date.
    split_ratio (float): The ratio of training data (default is 0.7).

    Returns:
    train (pd.DataFrame): The training data.
    test (pd.DataFrame): The testing data.
    """
    training_size = int(len(time_series_data)*split_ratio)
    #split the data into train, test
    train = time_series_data[:training_size]
    test = time_series_data[training_size:]
    return train, test