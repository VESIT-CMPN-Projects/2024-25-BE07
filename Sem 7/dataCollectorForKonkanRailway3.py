import requests
from bs4 import BeautifulSoup
import time
import json
import os
from datetime import datetime
import ssl
from urllib3.poolmanager import PoolManager
from requests.adapters import HTTPAdapter

# Custom SSL adapter to allow legacy renegotiation
class SSLAdapter(HTTPAdapter):
    def __init__(self, ssl_context=None, **kwargs):
        self.ssl_context = ssl_context
        super().__init__(**kwargs)

    def init_poolmanager(self, *args, **kwargs):
        kwargs['ssl_context'] = self.ssl_context
        return super().init_poolmanager(*args, **kwargs)

# Create an SSL context with legacy renegotiation enabled
ssl_context = ssl.create_default_context()
ssl_context.options |= ssl.OP_LEGACY_SERVER_CONNECT  # Enable legacy SSL renegotiation

# Create a session with the custom SSL adapter
session = requests.Session()
adapter = SSLAdapter(ssl_context=ssl_context)
session.mount("https://", adapter)

# URL to scrape
url = 'https://konkanrailway.com/VisualTrain/otrktp0100Table.jsp'

# Headers
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Referer": "https://konkanrailway.com/"
}

# Function to scrape data
def scrape_data():
    try:
        response = session.get(url, headers=headers)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')

        # Locate timestamp
        timestamp_row = soup.find('tr', {'align': 'right'})
        if timestamp_row:
            timestamp_text = timestamp_row.find('font').text.strip()
            timestamp = timestamp_text.replace('Last Updated time : ', '').strip()
            timestamp = datetime.strptime(timestamp, '%d/%m/%Y %H:%M:%S').strftime('%Y-%m-%d %H:%M:%S')
        else:
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            print(f"Warning: Timestamp not found. Using current time: {timestamp}")

        # Find table with train data
        table = soup.find('table', {'id': 'empNoHelpList'})
        if table is None:
            print("Error: Could not find train data table.")
            return

        rows = table.find_all('tr')[3:]  # Skip headers
        data = {timestamp: []}

        for row in rows:
            cols = row.find_all('td')
            if len(cols) == 6:
                data[timestamp].append({
                    'Train No': cols[0].text.strip(),
                    'Train Name': cols[1].text.strip(),
                    'Status': cols[2].text.strip(),
                    'Station': cols[3].text.strip(),
                    'Time': cols[4].text.strip(),
                    'Delay': cols[5].text.strip()
                })

        # Save to JSON
        save_to_json(data)
    except Exception as e:
        print(f"Error: {e}")

# Function to save data to JSON file
def save_to_json(data):
    filename = 'train_status2.json'
    if os.path.exists(filename):
        with open(filename, 'r') as file:
            existing_data = json.load(file)
    else:
        existing_data = {}

    existing_data.update(data)
    with open(filename, 'w') as file:
        json.dump(existing_data, file, indent=4)

# Main loop to scrape data every 10 minutes
while True:
    scrape_data()
    time.sleep(600)
