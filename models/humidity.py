import os

import keras
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras.src.layers import GRU
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential

file_path = '../dataset/kyiv-weather.csv'
model_path = 'humid.keras'
window_size = 9
column_name = 'relative_humidity_2m (%)'

df = pd.read_csv(file_path)

df = df[column_name][30207:]

df_daily = []
for i in range(0, len(df), 24):
        window = df[i:i + window_size]
        average = sum(window) / len(window)
        df_daily.append(average)
df = df_daily

df = df[9::24][column_name] / 100.0
test_index = round(df.values.shape[0] * 0.8)
test_size = df.values.shape[0] - test_index

train_data = df[0:test_index].values

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
    model.add(GRU(units=20, activation='relu', return_sequences=True, input_shape=(window_size, 1)))
    model.add(GRU(8, activation='relu', return_sequences=True))
    model.add(GRU(5, activation='relu', return_sequences=True))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='mean_squared_error')

    model.fit(X_train, y_train, epochs=60, batch_size=3, verbose=1)
    model.save(model_path)
else:
    model = keras.models.load_model(model_path)

def nn_rolling_predictions(data):
    predictions = []

    for start in range(len(data)-window_size):
        d = scaler_X.transform(np.array([data[start:start + window_size].values])).reshape(-1, window_size)
        #print('d=')
        #print(data[start:start + window_size].values)
        pred = model.predict(d)
        pred = scaler_y.inverse_transform([[pred[0,window_size-1,0]]])[0, 0]
        #print(pred)
        predictions.append(pred)
    return predictions

forecast_steps = test_size-window_size

predictions = nn_rolling_predictions(df[test_index:test_index+test_size])
actual_values = df[test_index:test_index+test_size].iloc[-forecast_steps:]
mse = mean_squared_error(actual_values, predictions)
print(f"Mean Squared Error (MSE): {mse}")

plt.plot(df[test_index:test_index+test_size].index, df[test_index:test_index+test_size], label='Actual')
plt.plot(actual_values.index, predictions, label='Predicted', color='red')
plt.xlabel('Date')
plt.ylabel('Temperature')
plt.title('Neural Network Rolling Predictions')
plt.legend()
plt.show()
