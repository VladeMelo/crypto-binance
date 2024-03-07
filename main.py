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

# PRICES

def get_last_info(symbol, limit):
    # Endpoint para obter klines/candlesticks de futuros
    endpoint = '/fapi/v1/klines'

    # Parâmetros para a requisição: par de moedas, intervalo de tempo e limite de dados
    params = {
        'symbol': symbol,
        'interval': '1h',
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

def RSI(df, periods=15):
  close_percentual_variation = df['Close'] / df['Close'].shift(1) - 1

  descent_delta = close_percentual_variation.clip(upper=0) * -1
  rise_delta = close_percentual_variation.clip(lower=0)

  descent_delta_average = descent_delta.rolling(window=periods).mean()
  rise_delta_average = rise_delta.rolling(window=periods).mean()

  rs = rise_delta_average / descent_delta_average

  rsi = 100 - 100 / (1 + rs)

  return rsi

def MACD(df, short_window=12, long_window=26):
  ema_fast = df['Close'].ewm(span=short_window, adjust=False).mean()
  ema_slow = df['Close'].ewm(span=long_window, adjust=False).mean()

  macd = ema_fast - ema_slow

  return macd

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

  return money_flow, negative_flow, mfi

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
        if float(position['positionAmt']) != 0.0:
            return True
    
    return False

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
    precision = int(round(-math.log(tick_size, 10), 0))

    return round(value, precision)

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

# SELL

def get_open_position_amount(symbol):
  url = f"{BASE_URL}/fapi/v2/positionRisk"

  params = {'timestamp': get_timestamp(), 'symbol': symbol}

  query_string = '&'.join([f"{k}={v}" for k, v in params.items()])
  signature = create_signature(query_string)

  response = requests.get(url + '?' + query_string + '&signature=' + signature, headers=headers())
  positions = response.json()

  for position in positions:
    if position['symbol'] == symbol:
      positionAmt = float(position['positionAmt'])

      return positionAmt

  return 0

def sell_position(symbol):
  position_amount = get_open_position_amount(symbol)
  
  if position_amount > 0:
    print(f"Vendendo toda a posicao de {symbol}, Quantidade: {position_amount}")

    # Coloque uma ordem de venda ao mercado para toda a posição
    place_order(symbol, 'SELL', 'MARKET', abs(position_amount))
  else:
    print(f"Nao existe posicao aberta para {symbol} para vender.")

# LOGIC
    
def run_logic():
    symbol = 'ADAUSDT'

    model = joblib.load('model.pkl')

    info = get_last_info(symbol, 26)

    open = list(map(lambda x: float(x[1]), info))
    high = list(map(lambda x: float(x[2]), info))
    low = list(map(lambda x: float(x[3]), info))
    close = list(map(lambda x: float(x[4]), info))
    volume_crypto = list(map(lambda x: float(x[5]), info))
    volume_usdt = list(map(lambda x: float(x[7]), info))
    trade_count = list(map(lambda x: x[8], info))

    data = {
        'Open': open,
        'Close': close,
        'Low': low,
        'High': high,
        'Volume_Crypto': volume_crypto,
        'Volume_USDT': volume_usdt,
        'Trade_Count': trade_count
    }

    df = pd.DataFrame(data)

    df['RSI'] = RSI(df)
    df['MACD'] = MACD(df)
    df['Williams_%R'] = Williams(df, periods=15)
    df['%K'], df['%D'] = StochasticOscillator(df)
    df['Money_Flow'], df['Negative_Flow'], df['MFI'] = MoneyFlowIndex(df)

    columns_in_order = ['Volume_Crypto', 'Volume_USDT', 'Trade_Count', 'RSI', 'MACD', 'Williams_%R','%K', '%D', 'Money_Flow', 'Negative_Flow', 'MFI']

    df_reorganized = df[columns_in_order].tail(1)

    signal = (model.predict_proba(df_reorganized)[:, 1] >= 0.4).astype(int)[0]
    global last_signal

    print('Ultimo sinal: {} / Sinal atual: {}'.format(last_signal, signal))

    if last_signal != signal:
        if signal == 1:
            if not check_open_positions(symbol):
                print("Ordem de compra iniciada")

                leverage = 2  # Define a alavancagem desejada
                set_leverage(symbol, leverage)  # Define a alavancagem para o par 

                tick_size = get_tick_size(symbol)  # Obtém o tickSize para o par
                ada_price = get_market_price(symbol)  # Obtem o preço atual de mercado
                usdt_balance = get_balance()  # Obtém o saldo disponível
                ada_quantity = (usdt_balance / ada_price) * leverage - 1   # Calcula a quantidade baseada no saldo e preço
                ada_quantity = adjust_value(ada_quantity, tick_size)  # Ajusta a quantidade baseada no tickSize

                print('ADA Price:', ada_price)
                print('USDT Balance:', usdt_balance)
                print('ADA To Buy:', int(ada_quantity))

                # Faz a ordem de compra
                order_response = place_order(symbol, 'BUY', 'MARKET', int(ada_quantity))
                print("Ordem de compra enviada:", order_response)
            else:
                print("Voce ja tem uma posicao aberta.")
        else:
            print("Vender posicao")
            sell_position(symbol)

    last_signal = signal

# SOCKET
    
def on_message(ws, message):
  data = json.loads(message)
  # Verifique se a mensagem é uma atualização de candlestick
  # print(data)
  if data['e'] == 'kline':
    kline = data['k']
    is_candle_closed = kline['x']

    if is_candle_closed:
      print("Vela de 1 hora fechada.")

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
            "adausdt@kline_1h"
        ],
        "id": 1
    }
    ws.send(json.dumps(subscribe_message))

if __name__ == "__main__":
    API_KEY = os.environ.get('API_KEY')
    SECRET_KEY = os.environ.get('SECRET_KEY')
    
    BASE_URL = 'https://fapi.binance.com'

    websocket_url = "wss://stream.binance.com:9443/ws/adausdt@kline_1h"

    last_signal = 0

    ws = websocket.WebSocketApp(websocket_url,
                                on_message=on_message,
                                on_error=on_error,
                                on_close=on_close)
    ws.on_open = on_open
    ws.run_forever()
