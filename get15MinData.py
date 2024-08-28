from binance.client import Client
import pandas as pd
import datetime
import time

api_key = os.environ.get('API_KEY')
api_secret = os.environ.get('SECRET_KEY')

client = Client(api_key, api_secret)

symbol = 'ETHUSDT'
interval = '15m'

# Substitua 'start_str' com a data de início desejada
start_str = "3 year ago UTC"

# Lista para armazenar os dados coletados
all_klines = []

while True:
    klines = client.get_historical_klines(symbol, interval, start_str, limit=1000)
    if not klines:
        break
    all_klines.extend(klines)
    start_str = klines[-1][0] + 1  # Atualiza o start_str para o timestamp do último kline + 1 millisecond
    time.sleep(2)  # Pausa para evitar atingir o limite de rate da API

# Preparando os dados para o DataFrame
data_for_df = []
for kline in all_klines:
    date = datetime.datetime.fromtimestamp(kline[0] / 1000)
    open_price = float(kline[1])
    high_price = float(kline[2])
    low_price = float(kline[3])
    close_price = float(kline[4])
    volume = float(kline[5])
    quote_asset_volume = float(kline[7])  # Volume do ativo base (USDT)
    
    # Adicionando ao dataset
    data_for_df.append([date, open_price, high_price, low_price, close_price, volume, quote_asset_volume])

# Criando o DataFrame
df = pd.DataFrame(data_for_df, columns=['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Volume Base Asset'])

# Salvando o DataFrame como CSV
df.to_csv('ETHUSDT_Binance_futures_15_min.csv', index=False)