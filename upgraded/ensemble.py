import math

import keras
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler

file_path = '..\\dataset\\weather_kyiv.csv'
fnn_model_path = '..\\deep_nn_model.keras'
lstm_model_path = '..\\lstm.keras'
window_size = 9

df = pd.read_csv(file_path, sep=',', parse_dates=['date'])
df_interpolated = df.interpolate(method='linear')
df_interpolated['date'] = pd.to_datetime(df_interpolated['date'])
df_interpolated.set_index('date', inplace=True)
complete_date_range = pd.date_range(start=df_interpolated.index.min(), end=df_interpolated.index.max(), freq='D')
df_filled = df_interpolated.reindex(complete_date_range)
df_filled = df_filled.ffill()
df_filled.reset_index(inplace=True)

if math.isnan(df_filled['TAVG'][0]):
    df_filled['TAVG'][0] = df_filled['TAVG'][1]

test_index = round(df_filled.values.shape[0] * 0.8)

train_data = df_filled['TAVG'][0:test_index].values

X_train = []
y_train = []

for group_index in range(int(len(train_data)-window_size)):
    X_train.append(train_data[group_index:group_index+window_size])
    y_train.append([train_data[group_index+window_size]])

scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

X_train = scaler_X.fit_transform(np.array(X_train)).reshape(-1, window_size)
y_train = scaler_y.fit_transform(np.array(y_train).reshape(-1, 1))

fnn = keras.models.load_model(fnn_model_path)
lstm = keras.models.load_model(lstm_model_path)

test_size = 365
column_name = 'TAVG'
def nn_rolling_predictions(data):
    predictions = []

    for start in range(len(data)-window_size):
        fnn_pred = fnn.predict(scaler_X.transform(np.array([data[column_name][start:start+window_size].values])).reshape(-1, window_size))[0, 0]
        lstm_pred = lstm.predict(scaler_X.transform(np.array([data[column_name][start:start + window_size].values])).reshape(-1, window_size))[0, window_size-1, 0]
        ensemble_predictions = fnn_pred * 0.8 + lstm_pred * 0.2
        pred = scaler_y.inverse_transform([[ensemble_predictions]])[0,0]
        predictions.append(pred)
    return predictions

forecast_steps = test_size-window_size

predictions = nn_rolling_predictions(df_filled[test_index:test_index+test_size])

actual_values = df_filled[test_index:test_index+test_size][column_name].iloc[-forecast_steps:]
mse = mean_squared_error(actual_values, predictions)
print(f"Mean Squared Error (MSE): {mse}")

plt.plot(df_filled[test_index:test_index+test_size].index, df_filled[test_index:test_index+test_size][column_name], label='Actual')
plt.plot(actual_values.index, predictions, label='Predicted', color='red')
plt.xlabel('Date')
plt.ylabel('Temperature')
plt.title('Neural Network Rolling Predictions')
plt.legend()
plt.show()
