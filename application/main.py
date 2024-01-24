import tkinter as tk
from datetime import datetime, timedelta

import joblib
import keras
import requests

models_path = '..\\models\\'
window_size = 9

temperature_model1 = keras.models.load_model('..\\simple_models\\lstm.keras')
temperature_model2 = keras.models.load_model('..\\simple_models\\deep_nn_model.keras')
humidity_model = keras.models.load_model(models_path + 'humid.keras')
dew_model = keras.models.load_model(models_path + 'dew.keras')
precipitation_model = keras.models.load_model(models_path + 'precipitation.keras')
snow_model = keras.models.load_model(models_path + 'sn.keras')
cloudcover_model = keras.models.load_model(models_path + 'cloud.keras')
wind_model = keras.models.load_model(models_path + 'wind.keras')

temperature_scaler = joblib.load('..\\scalers\\temperature.pkl')
humidity_scaler = joblib.load('..\\scalers\\humidity.pkl')
dew_scaler = joblib.load('..\\scalers\\dew.pkl')
precipitation_scaler = joblib.load('..\\scalers\\precip.pkl')
snow_scaler = joblib.load('..\\scalers\\snow.pkl')
cloudcover_scaler = joblib.load('..\\scalers\\cloudcover.pkl')
wind_scaler = joblib.load('..\\scalers\\wind.pkl')

def on_click():
    latitude = float(entry_lat.get())
    longitude = float(entry_long.get())
    url = f'https://api.open-meteo.com/v1/forecast?latitude={latitude}&longitude={longitude}&hourly=temperature_2m,relative_humidity_2m,dew_point_2m,precipitation,snow_depth,cloud_cover,wind_speed_10m&timezone=Europe%2FBerlin&past_days=9&forecast_days=1'
    response = requests.get(url)
    current_date = datetime.now()
    nine_days_ago = current_date - timedelta(days=window_size - 1)
    formatted_date = nine_days_ago.strftime("%Y-%m-%d") + 'T00:00'

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
        data = data['hourly']
        start_index = data['time'].index(formatted_date)

        for index in range(window_size):
            temperature_day = data['temperature_2m'][start_index + index * 24:start_index + (index + 1) * 24]
            temperature_avg.append(sum(temperature_day) / len(temperature_day))
            temperature_min.append(min(temperature_day))
            temperature_max.append(max(temperature_day))

            humidity_day = data['relative_humidity_2m'][start_index + index * 24:start_index + (index + 1) * 24]
            humidity.append(sum(humidity_day) / len(humidity_day) / 100)

            dew_day = data['dew_point_2m'][start_index + index * 24:start_index + (index + 1) * 24]
            dew.append(sum(dew_day) / len(dew_day))

            precipitation_day = data['precipitation'][start_index + index * 24:start_index + (index + 1) * 24]
            precipitation.append(sum(precipitation_day))

            snow_day = data['snow_depth'][start_index + index * 24:start_index + (index + 1) * 24]
            snow.append(sum(snow_day) / len(snow_day))

            cloudcover_day = data['cloud_cover'][start_index + index * 24:start_index + (index + 1) * 24]
            cloudcover.append(sum(cloudcover_day) / len(cloudcover_day) / 100)

            wind_day = data['wind_speed_10m'][start_index + index * 24:start_index + (index + 1) * 24]
            wind.append(sum(wind_day) / len(wind_day))

        acc = 0
        for index in range(int(len(precipitation))):
            acc = acc + precipitation[index]
            precipitation[index] = acc
        precipitation_acc = acc
    else:
        print(f"Error: {response.status_code}, {response.text}")

    temperature_avg = temperature_scaler.transform([temperature_avg])
    temperature_max = temperature_scaler.transform([temperature_max])
    temperature_min = temperature_scaler.transform([temperature_min])
    humidity = temperature_scaler.transform([humidity]).reshape(-1, window_size)
    dew = dew_scaler.transform([dew])
    precipitation = precipitation_scaler.transform([precipitation])
    snow = snow_scaler.transform([snow])
    cloudcover = cloudcover_scaler.transform([cloudcover])
    wind = wind_scaler.transform([wind])

    temperature_avg_pred = 0
    temperature_max_pred = 0
    temperature_min_pred = 0
    humidity_pred = 0
    dew_pred = 0
    precipitation_pred = 0
    snow_pred = 0
    cloudcover_pred = 0
    wind_pred = 0

    pred1 = temperature_model1.predict(temperature_avg).reshape(-1, window_size)
    pred2 = temperature_model2.predict(temperature_avg)[0, 0]
    pred = pred1 * 0.2 + pred2 * 0.8
    temperature_avg_pred = temperature_scaler.inverse_transform(pred)[0, window_size-1]

    pred1 = temperature_model1.predict(temperature_min).reshape(-1, window_size)
    pred2 = temperature_model2.predict(temperature_min)[0, 0]
    pred = pred1 * 0.2 + pred2 * 0.8
    temperature_min_pred = temperature_scaler.inverse_transform(pred)[0, window_size-1]

    pred1 = temperature_model1.predict(temperature_max).reshape(-1, window_size)
    pred2 = temperature_model2.predict(temperature_max)[0, 0]
    pred = pred1 * 0.2 + pred2 * 0.8
    temperature_max_pred = temperature_scaler.inverse_transform(pred)[0, window_size-1]

    pred = humidity_model.predict(humidity)
    pred = pred.reshape(-1, window_size)
    humidity_pred = humidity_scaler.inverse_transform(pred)[0, window_size-1]

    pred = dew_model.predict(dew).reshape(-1, window_size)
    pred = pred.reshape(-1, window_size)
    dew_pred = dew_scaler.inverse_transform(pred)[0, window_size-1]

    pred = precipitation_model.predict(precipitation).reshape(-1, window_size)
    pred = pred.reshape(-1, window_size)
    precipitation_pred = precipitation_scaler.inverse_transform(pred)[0, window_size - 1] - precipitation_acc
    precipitation_pred = max(0, precipitation_pred)

    pred = snow_model.predict(snow).reshape(-1, window_size)
    pred = pred.reshape(-1, window_size)
    snow_pred = snow_scaler.inverse_transform(pred)[0, window_size - 1]

    pred = cloudcover_model.predict(cloudcover)
    pred = pred.reshape(-1, window_size)
    cloudcover_pred = cloudcover_scaler.inverse_transform(pred)[0, window_size-1]

    pred = wind_model.predict(wind).reshape(-1, window_size)
    pred = pred.reshape(-1, window_size)
    wind_pred = wind_scaler.inverse_transform(pred)[0, window_size - 1]


    label1 = tk.Label(window, text="Середня температура за день (°C): {:.2f}".format(temperature_avg_pred))
    label2 = tk.Label(window, text="Максимальна температура за день (°C): {:.2f}".format(temperature_max_pred))
    label3 = tk.Label(window, text="Мінімальна температура за день (°C): {:.2f}".format(temperature_min_pred))
    label4 = tk.Label(window, text="Відностна вологість (%): {:.2f}".format(humidity_pred))
    label5 = tk.Label(window, text="Точка роси (°C): {:.2f}".format(dew_pred))
    label6 = tk.Label(window, text="Кількість опадів за день (мм): {:.2f}".format(precipitation_pred))
    label7 = tk.Label(window, text="Товщина снігу (мм): {:.2f}".format(snow_pred))
    label8 = tk.Label(window, text="Хмарність (%): {:.2f}".format(cloudcover_pred))
    label9 = tk.Label(window, text="Швидкість вітру (км/год): {:.2f}".format(wind_pred))

    label_lat.pack_forget()
    entry_lat.pack_forget()
    label_long.pack_forget()
    entry_long.pack_forget()
    submit_button.pack_forget()

    label1.pack()
    label2.pack()
    label3.pack()
    label4.pack()
    label5.pack()
    label6.pack()
    label7.pack()
    label8.pack()
    label9.pack()

    def on_click_again():
        label_title = tk.Label(window, text="Вкажіть широту та довготу місця для отримання прогнозу погоду на завтра")

        label1.pack_forget()
        label2.pack_forget()
        label3.pack_forget()
        label4.pack_forget()
        label5.pack_forget()
        label6.pack_forget()
        label7.pack_forget()
        label8.pack_forget()
        label9.pack_forget()
        again_button.pack_forget()

        label_lat.pack()
        entry_lat.pack()
        label_long.pack()
        entry_long.pack()
        submit_button.pack()

    again_button = tk.Button(window, text="Обрати інше місце", command=on_click_again)
    again_button.pack()


window = tk.Tk()
window.title("Input Form")
label_title = tk.Label(window, text="Вкажіть широту та довготу місця для отримання прогнозу погоду на завтра")

label_lat = tk.Label(window, text="Широта")
label_lat.pack()

entry_lat = tk.Entry(window)
entry_lat.pack()

label_long = tk.Label(window, text="Довгота")
label_long.pack()

entry_long = tk.Entry(window)
entry_long.pack()

submit_button = tk.Button(window, text="Отримати прогноз", command=on_click)
submit_button.pack()

window.mainloop()
