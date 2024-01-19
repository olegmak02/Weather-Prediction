from datetime import datetime, timedelta
import requests

import requests
from datetime import datetime, timedelta

# Replace "YOUR_API_KEY" with your actual Visual Crossing Weather API key
api_key = "WKVQGBK9C7UF2YE785WRUQ2UZ"

# Replace with the location you're interested in (e.g., city and country)
location = "Zaporizhzhia, Ukraine"

# Calculate the start and end date for the historical data (last 10 days)
end_date = datetime.utcnow()
start_date = end_date - timedelta(days=10)

# Format dates in the required format (YYYY-MM-DD)
start_date_str = start_date.strftime("%Y-%m-%d")
end_date_str = end_date.strftime("%Y-%m-%d")

# Visual Crossing Weather API URL
url = f"https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/zaporizhzhia/2023-12-07/2023-12-17?unitGroup=metric&key=WKVQGBK9C7UF2YE785WRUQ2UZ&contentType=json"

# Make a GET request to the Visual Crossing Weather API
response = requests.get(url)

# Check if the request was successful (status code 200)
if response.status_code == 200:
    # Parse the JSON response
    data = response.json()
    print(data['days'][0])

    # Access the historical weather data
    for entry in data['days']:
        date = entry['datetime']
        temperature_max = entry['temp2max']
        temperature_min = entry['temp2min']
        precipitation = entry['precip']

        print(f"Date: {date}, Max Temperature: {temperature_max}, Min Temperature: {temperature_min}, Precipitation: {precipitation}")
else:
    print(f"Error: {response.status_code}, {response.text}")