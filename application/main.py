from datetime import datetime, timedelta

import keras
import requests

url = f'https://api.open-meteo.com/v1/forecast?latitude=47.8517&longitude=35.1171&hourly=temperature_2m,relative_humidity_2m,dew_point_2m,precipitation,snow_depth,cloud_cover,wind_speed_10m&timezone=Europe%2FBerlin&past_days=9&forecast_days=1'

models_path = '..\\models\\'
window_size = 9
response = requests.get(url)
current_date = datetime.now()
nine_days_ago = current_date - timedelta(days=window_size-1)
formatted_date = nine_days_ago.strftime("%Y-%m-%d") + 'T00:00'

temperature_model = keras.models.load_model(models_path + '')
humidity_model = keras.models.load_model(models_path + 'humid.keras')
dew_model = keras.models.load_model(models_path + 'dew.keras')
precipitation_model = keras.models.load_model(models_path + 'precipitation.keras')
snow_model = keras.models.load_model(models_path + 'sn.keras')
cloudcover_model = keras.models.load_model(models_path + 'cloud.keras')
wind_model = keras.models.load_model(models_path + 'wind.keras')

temperature_avg = []
temperature_max = []
temperature_min = []
humidity = []
dew = []
precipitation = []
snow = []
cloudcover = []
wind = []

if response.status_code == 200:
    data = response.json()
    print(data)
    data = data['hourly']
    start_index = data.index(formatted_date)

    for index in range(window_size):
        temperature_day = data['temperature_2m'][start_index:start_index+index*24]
        temperature_avg.append(sum(temperature_day)/len(temperature_day))
        temperature_min.append(min(temperature_day))
        temperature_max.append(max(temperature_day))

        humidity_day = data['relative_humidity_2m'][start_index:start_index + index * 24]
        humidity.append(sum(humidity_day)/len(humidity_day)/100)

        dew_day = data['dew_point_2m'][start_index:start_index + index * 24]
        dew.append(sum(dew_day)/len(dew_day))

        precipitation_day = data['precipitation'][start_index:start_index + index * 24]
        for group_index in range(int(len(precipitation_day) - window_size)):
            set = train_data[group_index:group_index + window_size].copy()
            acc = 0
            for ind in range(len(set)):
                acc = acc + set[ind]
                set[ind] = acc
            X_train.append(set)
        precipitation.append()

        snow_day = data['snow_depth'][start_index:start_index + index * 24]
        snow.append(sum(snow_day)/len(snow_day))

        cloudcover_day = data['cloud_cover'][start_index:start_index + index * 24]
        cloudcover.append(sum(cloudcover_day)/len(cloudcover_day)/100)

        wind_day = data['wind_speed_10m'][start_index:start_index + index * 24]
        wind.append(sum(wind_day)/len(wind_day))

else:
    print(f"Error: {response.status_code}, {response.text}")