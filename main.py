import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

from pipeline import load_and_prepare, split_data
from modelling import train_model, plot_actual_vs_predicted

# Step 1: Load data
df = load_and_prepare("data/tsla_2025.csv")
print("Data has been loaded")

# Step 2: Train/Test split
train, test = split_data(df)
print("Data has been split into train and test")

# Step 3: Train ARIMA
predictions, mae, rmse, mape = train_model(train, test, order=(2, 0, 0))

# Step 4: Plot results
plot_actual_vs_predicted(test, predictions)
