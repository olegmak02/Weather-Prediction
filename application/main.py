import tkinter as tk
from datetime import datetime, timedelta

import joblib
import keras
import requests

models_path = '..\\models\\'
window_size = 9

temperature_model1 = keras.models.load_model('../research/simple_models\\lstm.keras')
temperature_model2 = keras.models.load_model('../research/simple_models\\deep_nn_model.keras')
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

def get_meteo_data(latitude, longitude):
    url = f'https://api.open-meteo.com/v1/forecast?latitude={latitude}&longitude={longitude}&hourly=temperature_2m,relative_humidity_2m,dew_point_2m,precipitation,snow_depth,cloud_cover,wind_speed_10m&timezone=Europe%2FBerlin&past_days=9&forecast_days=1'
    return requests.get(url)

temperature_avg = []
temperature_max = []
temperature_min = []
humidity = []
dew = []
precipitation = []
snow = []
cloudcover = []
wind = []
precipitation_acc = 0


temperature_avg_scaled = []
temperature_max_scaled = []
temperature_min_scaled = []
humidity_scaled = []
dew_scaled = []
precipitation_scaled = []
snow_scaled = []
cloudcover_scaled = []
wind_scaled = []

temperature_avg_pred = 0
temperature_max_pred = 0
temperature_min_pred = 0
humidity_pred = 0
dew_pred = 0
precipitation_pred = 0
snow_pred = 0
cloudcover_pred = 0
wind_pred = 0

def data_preparation(data):
    current_date = datetime.now()
    nine_days_ago = current_date - timedelta(days=window_size - 1)
    formatted_date = nine_days_ago.strftime("%Y-%m-%d") + 'T00:00'
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

def scale_data():
    global temperature_avg_scaled
    global temperature_max_scaled
    global temperature_min_scaled
    global humidity_scaled
    global dew_scaled
    global precipitation_scaled
    global snow_scaled
    global cloudcover_scaled
    global wind_scaled

    temperature_avg_scaled = temperature_scaler.transform([temperature_avg])
    temperature_max_scaled = temperature_scaler.transform([temperature_max])
    temperature_min_scaled = temperature_scaler.transform([temperature_min])
    humidity_scaled = temperature_scaler.transform([humidity]).reshape(-1, window_size)
    dew_scaled = dew_scaler.transform([dew])
    precipitation_scaled = precipitation_scaler.transform([precipitation])
    snow_scaled = snow_scaler.transform([snow])
    cloudcover_scaled = cloudcover_scaler.transform([cloudcover])
    wind_scaled = wind_scaler.transform([wind])

def predict():
    global temperature_avg_pred
    global temperature_max_pred
    global temperature_min_pred
    global humidity_pred
    global dew_pred
    global precipitation_pred
    global snow_pred
    global cloudcover_pred
    global wind_pred

    pred1 = temperature_model1.predict(temperature_avg_scaled).reshape(-1, window_size)
    pred2 = temperature_model2.predict(temperature_avg_scaled)[0, 0]
    pred = pred1 * 0.2 + pred2 * 0.8
    temperature_avg_pred = temperature_scaler.inverse_transform(pred)[0, window_size - 1]

    pred1 = temperature_model1.predict(temperature_min_scaled).reshape(-1, window_size)
    pred2 = temperature_model2.predict(temperature_min_scaled)[0, 0]
    pred = pred1 * 0.2 + pred2 * 0.8
    temperature_min_pred = temperature_scaler.inverse_transform(pred)[0, window_size - 1]

    pred1 = temperature_model1.predict(temperature_max_scaled).reshape(-1, window_size)
    pred2 = temperature_model2.predict(temperature_max_scaled)[0, 0]
    pred = pred1 * 0.2 + pred2 * 0.8
    temperature_max_pred = temperature_scaler.inverse_transform(pred)[0, window_size - 1]

    pred = humidity_model.predict(humidity_scaled)
    pred = pred.reshape(-1, window_size)
    humidity_pred = humidity_scaler.inverse_transform(pred)[0, window_size - 1]

    pred = dew_model.predict(dew_scaled).reshape(-1, window_size)
    pred = pred.reshape(-1, window_size)
    dew_pred = dew_scaler.inverse_transform(pred)[0, window_size - 1]

    pred = precipitation_model.predict(precipitation_scaled).reshape(-1, window_size)
    pred = pred.reshape(-1, window_size)
    precipitation_pred = precipitation_scaler.inverse_transform(pred)[0, window_size - 1] - precipitation_acc
    precipitation_pred = max(0, precipitation_pred)

    pred = snow_model.predict(snow_scaled).reshape(-1, window_size)
    pred = pred.reshape(-1, window_size)
    snow_pred = snow_scaler.inverse_transform(pred)[0, window_size - 1]

    pred = cloudcover_model.predict(cloudcover_scaled)
    pred = pred.reshape(-1, window_size)
    cloudcover_pred = cloudcover_scaler.inverse_transform(pred)[0, window_size - 1]

    pred = wind_model.predict(wind_scaled).reshape(-1, window_size)
    pred = pred.reshape(-1, window_size)
    wind_pred = wind_scaler.inverse_transform(pred)[0, window_size - 1]

def on_click():
    try:
        latitude = float(entry_lat.get())
        longitude = float(entry_long.get())
        response = get_meteo_data(latitude, longitude)
    except ValueError:
        label_error.config(text="Введені значення не є числами")
        return

    if latitude > 180 or latitude < -180 or longitude > 180 or longitude < -180:
        label_error.config(text="Введені значення не лежать в діапазоні від -180 до 180")
        return

    if response.status_code == 200:
        data = response.json()
        data_preparation(data)
        label_error.config(text="")
    else:
        label_error.config(text=f"Невдалось отримати дані з метеоресурсу. Помилка: {response.status_code}, {response.text}")
        return

    scale_data()
    predict()

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
    label_error.pack_forget()
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
        tk.Label(window, text="Вкажіть широту та довготу місця для отримання прогнозу погоду на завтра")
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
        label_error.pack()
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

label_error = tk.Label(window, text="")
label_error.pack()

submit_button = tk.Button(window, text="Отримати прогноз", command=on_click)
submit_button.pack()

window.mainloop()
