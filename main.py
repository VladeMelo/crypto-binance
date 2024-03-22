import pandas as pd
import requests
import joblib
import hmac
import hashlib
import time
import websocket
import json
import math
import os
import numpy as np
import datetime

# PRICES

def get_last_info(symbol, limit):
    # Endpoint para obter klines/candlesticks de futuros
    endpoint = '/fapi/v1/klines'

    # Parâmetros para a requisição: par de moedas, intervalo de tempo e limite de dados
    params = {
        'symbol': symbol,
        'interval': '15m',
        'limit': limit  # Número de candlesticks retornados
    }

    # Cabeçalhos da requisição
    headers = {
        'X-MBX-APIKEY': API_KEY
    }

    # Executa a requisição
    response = requests.get(BASE_URL + endpoint, params=params, headers=headers)
    
    return response.json()

# INDICATORS

def ATR(df, periods=14):
    df['High_Low'] = df['High'] - df['Low']
    df['High_Close'] = abs(df['High'] - df['Close'].shift())
    df['Low_Close'] = abs(df['Low'] - df['Close'].shift())
    df['TR'] = df[['High_Low', 'High_Close', 'Low_Close']].max(axis=1)

    return df['TR'].rolling(window=periods).mean()

def RSI(df, periods=15):
  close_percentual_variation = df['Close'] / df['Close'].shift(1) - 1

  descent_delta = close_percentual_variation.clip(upper=0) * -1
  rise_delta = close_percentual_variation.clip(lower=0)

  descent_delta_average = descent_delta.rolling(window=periods).mean()
  rise_delta_average = rise_delta.rolling(window=periods).mean()

  rs = rise_delta_average / descent_delta_average

  rsi = 100 - 100 / (1 + rs)

  return rsi

def Williams(df, periods):
  highest_high = df['High'].rolling(window=periods).max()
  lowest_low = df['Low'].rolling(window=periods).min()
  williams = ((highest_high - df['Close']) / (highest_high - lowest_low)) * -100

  return williams

def StochasticOscillator(df, periods_k=14, periods_d=3):
  highest_high = df['High'].rolling(window=periods_k).max()
  lowest_low = df['Low'].rolling(window=periods_k).min()
  k = ((df['Close'] - lowest_low) / (highest_high - lowest_low)) * 100

  d = k.rolling(window=periods_d).mean()

  return k, d

def MoneyFlowIndex(df, period=14):
  typical_price = (df['High'] + df['Low'] + df['Close']) / 3

  money_flow = typical_price * df['Volume_USDT']

  positive_flow = money_flow.where(typical_price > typical_price.shift(), 0)
  negative_flow = money_flow.where(typical_price < typical_price.shift(), 0)

  money_flow_Ratio = (positive_flow.rolling(window=period).sum() / 
                            negative_flow.rolling(window=period).sum())

  mfi = 100 - (100 / (1 + money_flow_Ratio))

  return mfi

def calculate_EMA(df, span):
  return df.ewm(span=span, adjust=False).mean()

def relative_difference(series1, series2):
  return (series1 - series2) / series2

def EMA5_20(df):
    return relative_difference(calculate_EMA(df['Close'], 5), calculate_EMA(df['Close'], 20))

def ADX(df, periods=14):
    df['DMplus'] = np.where((df['High'] - df['High'].shift(1)) > (df['Low'].shift(1) - df['Low']), 
                            df['High'] - df['High'].shift(1), 0)
    df['DMminus'] = np.where((df['Low'].shift(1) - df['Low']) > (df['High'] - df['High'].shift(1)), 
                             df['Low'].shift(1) - df['Low'], 0)
    atr = ATR(df, periods)
    adx = (abs(df['DMplus'] - df['DMminus']) / (df['DMplus'] + df['DMminus'])).rolling(window=periods).mean()
    
    return (adx / atr) * 100

def CMO(df, periods=14):
    diff = df['Close'] - df['Close'].shift(1)
    up = diff.where(diff > 0, 0)
    down = abs(diff.where(diff < 0, 0))
    sum_up = up.rolling(window=periods).sum()
    sum_down = down.rolling(window=periods).sum()

    return ((sum_up - sum_down) / (sum_up + sum_down)) * 100

# BUY

def get_timestamp():
    return int(time.time() * 1000)

def create_signature(query_string):
    return hmac.new(SECRET_KEY.encode(), query_string.encode(), hashlib.sha256).hexdigest()

def headers():
    return {'X-MBX-APIKEY': API_KEY}

def get_balance(asset='USDT'):
    url = f"{BASE_URL}/fapi/v2/balance"

    params = {'timestamp': get_timestamp()}

    query_string = '&'.join([f"{k}={v}" for k, v in params.items()])
    signature = create_signature(query_string)

    response = requests.get(url + '?' + query_string + '&signature=' + signature, headers=headers())
    balances = response.json()

    usdt_balance = next((item for item in balances if item['asset'] == asset), {}).get('balance', 0)

    return float(usdt_balance)

def get_market_price(symbol):
    url = f"{BASE_URL}/fapi/v1/ticker/price"

    params = {'symbol': symbol}
    
    response = requests.get(url, params=params)
    price_data = response.json()

    return float(price_data['price'])

def check_open_positions(symbol):
    url = f"{BASE_URL}/fapi/v2/positionRisk"

    params = {'timestamp': get_timestamp(), 'symbol': symbol}

    query_string = '&'.join([f"{k}={v}" for k, v in params.items()])
    signature = create_signature(query_string)

    response = requests.get(url + '?' + query_string + '&signature=' + signature, headers=headers())
    positions = response.json()

    # Considera posição aberta se a quantidade é maior que zero
    for position in positions:
        positionAmt = float(position['positionAmt'])

        if positionAmt != 0.0:
            positionType = 'Long' if positionAmt > 0 else 'Short'

            return (positionType, abs(positionAmt))
    
    return (None, 0)

def get_tick_size(symbol):
    url = f"{BASE_URL}/fapi/v1/exchangeInfo"

    response = requests.get(url)

    data = response.json()

    for pair in data['symbols']:
        if pair['symbol'] == symbol:
            for filter in pair['filters']:
                if filter['filterType'] == 'PRICE_FILTER':
                    return float(filter['tickSize'])
    return None

def adjust_value(value, tick_size):
    precision = int(-math.log10(tick_size))
    scale = 10 ** precision
    return math.floor(value * scale) / scale

def set_leverage(symbol, leverage):
    url = f"{BASE_URL}/fapi/v1/leverage"

    params = {
        'symbol': symbol,
        'leverage': leverage,
        'timestamp': get_timestamp()
    }

    query_string = '&'.join([f"{k}={v}" for k, v in params.items()])
    signature = create_signature(query_string)

    final_url = f"{url}?{query_string}&signature={signature}"

    response = requests.post(final_url, headers=headers())

    return response.json()

def get_current_leverage(symbol):
    url = f"{BASE_URL}/fapi/v2/positionRisk"

    params = {
        'symbol': symbol,
        'timestamp': get_timestamp()
    }

    query_string = '&'.join([f"{k}={v}" for k, v in params.items()])
    signature = create_signature(query_string)

    final_url = f"{url}?{query_string}&signature={signature}"

    response = requests.get(final_url, headers=headers())

    # Assuming the response includes a list of positions and you need the one matching your symbol
    positions = response.json()
    for position in positions:
        if position['symbol'] == symbol:
            return int(position['leverage'])

def place_order(symbol, side, type, quantity):
    url = f"{BASE_URL}/fapi/v1/order"

    params = {
        'symbol': symbol,
        'side': side,
        'type': type,
        'quantity': quantity,
        'timestamp': get_timestamp()
    }

    query_string = '&'.join([f"{k}={v}" for k, v in params.items()])
    signature = create_signature(query_string)

    response = requests.post(url + '?' + query_string + '&signature=' + signature, headers=headers())

    return response.json()

def get_crypto_quantity(symbol, leverage):
    tick_size = get_tick_size(symbol)  # Obtém o tickSize para o par
    crypto_price = get_market_price(symbol)  # Obtem o preço atual de mercado
    usdt_balance = get_balance()  # Obtém o saldo disponível
    crypto_quantity = (usdt_balance / crypto_price) * leverage   # Calcula a quantidade baseada no saldo e preço
    crypto_quantity = adjust_value(crypto_quantity * 0.97, tick_size)  # Ajusta a quantidade baseada no tickSize

    return crypto_quantity

# LOGIC
    
def run_logic():
    symbol = 'ETHUSDT'

    model = joblib.load('model.pkl')

    info = get_last_info(symbol, 4 * 48)

    open_time = list(map(lambda x: float(x[0]), info))
    open = list(map(lambda x: float(x[1]), info))
    high = list(map(lambda x: float(x[2]), info))
    low = list(map(lambda x: float(x[3]), info))
    close = list(map(lambda x: float(x[4]), info))
    volume_crypto = list(map(lambda x: float(x[5]), info))
    volume_usdt = list(map(lambda x: float(x[7]), info))
    trade_count = list(map(lambda x: x[8], info))

    data = {
        'Open_Time': open_time,
        'Open': open,
        'Close': close,
        'Low': low,
        'High': high,
        'Volume_Crypto': volume_crypto,
        'Volume_USDT': volume_usdt,
        'Trade_Count': trade_count
    }

    df = pd.DataFrame(data)

    df['Open_Time'] = df['Open_Time'].apply(lambda x: datetime.datetime.utcfromtimestamp(x / 1000).strftime('%H:%M'))

    df['Expected Return'] = df['Close'].shift(-1) / df['Close'] - 1
    df['Volatility'] = df['Expected Return'].shift(1).rolling(window=24).std() * np.sqrt(4 * 24 * 7)

    df['ATR'] = ATR(df.copy())
    df['RSI'] = RSI(df)
    df['Williams_%R'] = Williams(df, periods=15)
    df['%K'], df['%D'] = StochasticOscillator(df)
    df['MFI'] = MoneyFlowIndex(df)
    df['EMA5-20'] = EMA5_20(df)
    df['AO14'] = Williams(df, periods=14)
    df['ADX'] = ADX(df.copy())
    df['CMO'] = CMO(df)

    columns_in_order = ['Volume_Crypto', 'Volume_USDT', 'ATR', 'Williams_%R', '%K', '%D', 'AO14', 'ADX', 'CMO']

    df_reorganized = df[columns_in_order].tail(1)

    signal = model.predict(df_reorganized)[0]

    print('Sinal: {}'.format(signal))

    position_status, position_amount = check_open_positions(symbol)

    if signal == 1:
        row = df[df['Open_Time'] == '06:00'].iloc[1, :] # pega o penultimo 6 horas
        lower_band_1std = row["Close"] - row["Volatility"] * row["Close"]

        if position_status != 'Long': # estava fora do mercado
            if row['Close'] < lower_band_1std:
                leverage = 30
            else:
                leverage = 3

            set_leverage(symbol, leverage)

            crypto_quantity = get_crypto_quantity(symbol, leverage)

            place_order(symbol, 'BUY', 'MARKET', crypto_quantity)
        else: # ja esta dentro, sendo long ou long std
            position = 'Long' if get_current_leverage(symbol) == 3 else 'Long STD'

            if position == 'Long' and row['Close'] < lower_band_1std:
                place_order(symbol, 'SELL', 'MARKET', position_amount)

                time.sleep(10)

                leverage = 30
                set_leverage(symbol, leverage)

                crypto_quantity = get_crypto_quantity(symbol, leverage)

                place_order(symbol, 'BUY', 'MARKET', crypto_quantity)
            elif position == 'Long STD' and row['Close'] >= lower_band_1std:
                place_order(symbol, 'SELL', 'MARKET', position_amount)

                time.sleep(10)

                leverage = 3
                set_leverage(symbol, leverage)

                crypto_quantity = get_crypto_quantity(symbol, leverage)

                place_order(symbol, 'BUY', 'MARKET', crypto_quantity)
                
    elif signal == -1:
        if position_status == 'Long':
            place_order(symbol, 'SELL', 'MARKET', position_amount)

            time.sleep(10)

# SOCKET
    
def on_message(ws, message):
  data = json.loads(message)
  # Verifique se a mensagem é uma atualização de candlestick
  # print(data)
  if data['e'] == 'kline':
    kline = data['k']
    is_candle_closed = kline['x']

    if is_candle_closed:
      print("Vela de 15 min fechada.")

      run_logic()

def on_error(ws, error):
    print(error)

def on_close(ws):
    print("### closed ###")

def on_open(ws):
    print("WebSocket aberto.")
    # Subscreve para atualizações de candlestick de 1 hora para ADAUSDT
    subscribe_message = {
        "method": "SUBSCRIBE",
        "params": [
            "ethusdt@kline_15m"
        ],
        "id": 1
    }
    ws.send(json.dumps(subscribe_message))

if __name__ == "__main__":
    API_KEY = os.environ.get('API_KEY')
    SECRET_KEY = os.environ.get('SECRET_KEY')
    
    BASE_URL = 'https://fapi.binance.com'

    websocket_url = "wss://stream.binance.com:9443/ws/ethusdt@kline_15m"

    ws = websocket.WebSocketApp(websocket_url,
                                on_message=on_message,
                                on_error=on_error,
                                on_close=on_close)
    ws.on_open = on_open
    ws.run_forever()

    print('Começou o programa!')