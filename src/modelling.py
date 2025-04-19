import math
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error


def train_model(train, test, order=(2,0,0)):
    """
    Arima model to predict volume (stock quantity)
    """
        
    #build the model and fit on train data
    model = ARIMA(train['Volume'], order=order)
    model_fit = model.fit()
    
    #print the results
    print("ARIMA models results:")
    print()
    print(model_fit.summary())

    #define the start and end index in order to make predictions 
    start = len(train)
    end = len(train+test)-1

    #make predictions on test data 
    predictions = model_fit.predict(start=start, end=end)
    predictions = pd.Series(predictions, index=test.index)
    predictions.to_csv("Predictions.csv", index=True)

    #error metric mae, rmse, mape
    mae = mean_absolute_error(test['Volume'], predictions)
    rmse = math.sqrt(mean_squared_error(test['Volume'], predictions))
    mape = round(mean_absolute_percentage_error(test['Volume'], predictions),2)*100
    print(f'MAE is {mae} and RMSE is {rmse}, we can say the predictions are off by {mape}%')
    print()

    return predictions, mae, rmse, mape

def plot_actual_vs_predicted(test, predictions):
    """
    #compare actual vs predicted
    """
    
    comparison = pd.concat([test['Volume'].reset_index(drop=True), predictions.reset_index(drop=True)], axis=1)
    comparison.columns = ['Actual', 'Predicted']
    comparison.to_csv("Actual vs Predicted.csv", index=True)

    print("\nActual vs Predicted plot:")
    comparison.plot(title="Actual vs Predicted Volume", figsize=(10, 5))
    plt.xlabel("Time Index")
    plt.ylabel("Volume")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    
