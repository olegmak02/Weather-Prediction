from datetime import datetime, timedelta

import keras
import requests
from sklearn.preprocessing import MinMaxScaler
import numpy as np

fnn_model_path = '../simple_models/deep_nn_model.keras'
lstm_model_path = '../simple_models/lstm.keras'
window_size = 9
file_path = '..\\dataset\\kyiv-weather.csv'
column_name = 'cloud_cover (%)'

api_key = "WKVQGBK9C7UF2YE785WRUQ2UZ"
location = "Zaporizhzhia, Ukraine"
url = f"https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/zaporizhzhia/2000-03-22/2000-04-30?unitGroup=metric&key=WKVQGBK9C7UF2YE785WRUQ2UZ&contentType=json"

response = requests.get(url)

temperatures = []

if response.status_code == 200:
    data = response.json()
    print(data)
    print('\n\n\n\n\n')
    for entry in data['days']:
        temperatures.append(entry['temp'])
else:
    print(f"Error: {response.status_code}, {response.text}")

print(temperatures)
temperatures_last_9 = temperatures[-9:]
print('last 9 temps\n\n\n')
print(temperatures_last_9)





scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

scaler_X.fit_transform(np.array([temperatures_last_9])).reshape(-1, 9)
scaler_y.fit_transform(np.array([temperatures_last_9]).reshape(-1,1))

fnn = keras.models.load_model(fnn_model_path)
lstm = keras.models.load_model(lstm_model_path)

end_date = datetime.utcnow()
start_date = end_date - timedelta(days=10)

fnn_pred = fnn.predict(
    scaler_X.transform(np.array([temperatures_last_9[0:9]])).reshape(-1, window_size))[0, 0]
lstm_pred = lstm.predict(
    scaler_X.transform(np.array([temperatures_last_9[0:9]])).reshape(-1, window_size))[0, 0]
ensemble_prediction = fnn_pred * 0.8 + lstm_pred * 0.2
print(ensemble_prediction)
pred = scaler_y.inverse_transform([ensemble_prediction])
print('Predicted value is ')
print(pred)