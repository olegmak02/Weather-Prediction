import math
import os

import keras
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM

file_path = '.\\dataset\\weather_kyiv.csv'
model_path = 'lstm.keras'
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

if not os.path.exists(model_path):
    model = Sequential()
    model.add(LSTM(units=50, activation='relu', return_sequences=True, input_shape=(window_size, 1)))
    model.add(LSTM(units=50, activation='relu', return_sequences=True))
    model.add(LSTM(16, activation='relu', return_sequences=True))
    model.add(LSTM(16, activation='relu', return_sequences=True))
    model.add(LSTM(16, activation='relu', return_sequences=True))
    model.add(LSTM(8, activation='relu', return_sequences=True))
    model.add(LSTM(5, activation='relu', return_sequences=True))
    model.add(Dense(1, activation='linear'))
    model.compile(optimizer='adam', loss='mean_squared_error')

    model.fit(X_train, y_train, epochs=15, batch_size=10, verbose=1)
    model.save(model_path)
else:
    model = keras.models.load_model(model_path)

test_size = 365
column_name = 'TAVG'
def nn_rolling_predictions(data):
    predictions = []

    for start in range(len(data)-window_size):
        d = scaler_X.transform(np.array([data[column_name][start:start + window_size].values])).reshape(-1, window_size)
        pred = model.predict(d)
        pred = scaler_y.inverse_transform([[pred[0,window_size-1,0]]])[0, 0]
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