import sys
import os

# Add the root directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.pipeline import load_and_prepare, split_data
from src.modelling import train_model
import pandas as pd

def test_model():
    time_series_data = load_and_prepare("data/tsla_2025.csv")
    train, test = split_data(time_series_data)
    predictions, mae, rmse, mape = train_model(train, test, order=(2, 0, 0))

    # Step 3: Simple functional checks
    assert len(predictions) == len(test), "Predictions and test set should have same length"

    assert isinstance(mae, float), "MAE should be a float"
    assert isinstance(rmse, float), "RMSE should be a float"
    assert isinstance(mape, float), "MAPE should be a float"
    print("✅ All basic model checks passed!")
