import os
import json
import time
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
import requests
import streamlit as st
import streamlit.components.v1 as components
import yfinance as yf
from arch import arch_model

st.set_page_config(page_title='PSU Banks Live Hybrid Dashboard', layout='wide')

LOG_PATH = os.environ.get('APP_LOG_PATH', '/tmp/app_events.jsonl')

PSU_BANKS = {
    'SBIN.NS': 'State Bank of India',
    'PNB.NS': 'Punjab National Bank',
    'BANKBARODA.NS': 'Bank of Baroda',
    'CANBK.NS': 'Canara Bank',
    'UNIONBANK.NS': 'Union Bank of India',
    'INDIANB.NS': 'Indian Bank',
    'BANKINDIA.NS': 'Bank of India',
    'MAHABANK.NS': 'Bank of Maharashtra',
    'CENTRALBK.NS': 'Central Bank of India',
    'UCOBANK.NS': 'UCO Bank',
    'PSB.NS': 'Punjab & Sind Bank',
    'IOB.NS': 'Indian Overseas Bank'
}

TITLE_POSITIVE = ['beats', 'surge', 'rally', 'upgrade', 'profit', 'growth', 'strong', 'buyback', 'expands']
TITLE_NEGATIVE = ['falls', 'downgrade', 'loss', 'probe', 'fraud', 'weak', 'misses', 'cuts', 'decline']
NSE_HEADERS = {
    'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36',
    'Accept': 'application/json,text/plain,*/*',
    'Accept-Language': 'en-US,en;q=0.9',
    'Referer': 'https://www.nseindia.com/',
    'Connection': 'keep-alive'
}


def log_event(event_type, payload):
    try:
        rec = {
            'ts_utc': pd.Timestamp.utcnow().isoformat(),
            'event_type': str(event_type),
            'payload': payload
        }
        with open(LOG_PATH, 'a', encoding='utf-8') as file_handle:
            file_handle.write(json.dumps(rec) + '\n')
    except Exception:
        pass


@st.cache_resource
def load_models():
    started = time.time()
    models_local = joblib.load('hybrid_models (1).joblib')
    elapsed_ms = int((time.time() - started) * 1000)
    log_event('models_loaded', {'n_models': len(models_local), 'load_ms': elapsed_ms})
    return models_local


@st.cache_data(ttl=120)
def fetch_bulk_histories(symbols, period, interval, refresh_key):
    if len(symbols) == 0:
        return {}

    ticker_string = ' '.join(symbols)
    downloaded = yf.download(
        ticker_string,
        period=period,
        interval=interval,
        auto_adjust=False,
        progress=False,
        group_by='ticker',
        threads=False
    )

    result = {}
    if downloaded is None or len(downloaded) == 0:
        for symbol in symbols:
            result[symbol] = pd.DataFrame()
        return result

    if isinstance(downloaded.columns, pd.MultiIndex):
        for symbol in symbols:
            if symbol not in downloaded.columns.get_level_values(0):
                result[symbol] = pd.DataFrame()
                continue
            frame = downloaded[symbol].copy().reset_index()
            if 'Date' not in frame.columns:
                if 'Datetime' in frame.columns:
                    frame.rename(columns={'Datetime': 'Date'}, inplace=True)
                else:
                    frame.rename(columns={frame.columns[0]: 'Date'}, inplace=True)
            keep_cols = [column for column in ['Date', 'Open', 'High', 'Low', 'Close', 'Volume'] if column in frame.columns]
            frame = frame[keep_cols]
            frame.dropna(subset=['Close'], inplace=True)
            frame.sort_values('Date', inplace=True)
            frame.reset_index(drop=True, inplace=True)
            result[symbol] = frame
    else:
        frame = downloaded.copy().reset_index()
        if 'Date' not in frame.columns:
            if 'Datetime' in frame.columns:
                frame.rename(columns={'Datetime': 'Date'}, inplace=True)
            else:
                frame.rename(columns={frame.columns[0]: 'Date'}, inplace=True)
        keep_cols = [column for column in ['Date', 'Open', 'High', 'Low', 'Close', 'Volume'] if column in frame.columns]
        frame = frame[keep_cols]
        frame.dropna(subset=['Close'], inplace=True)
        frame.sort_values('Date', inplace=True)
        frame.reset_index(drop=True, inplace=True)
        only_symbol = symbols[0]
        result[only_symbol] = frame
        for symbol in symbols[1:]:
            result[symbol] = pd.DataFrame()

    return result


def normalize_ohlcv_frame(frame):
    if frame is None or len(frame) == 0:
        return pd.DataFrame()

    normalized = frame.copy()
    if 'Date' not in normalized.columns:
        normalized = normalized.reset_index()
        if 'Date' not in normalized.columns:
            if 'Datetime' in normalized.columns:
                normalized.rename(columns={'Datetime': 'Date'}, inplace=True)
            else:
                normalized.rename(columns={normalized.columns[0]: 'Date'}, inplace=True)

    if 'Close' not in normalized.columns:
        return pd.DataFrame()

    if 'Open' not in normalized.columns:
        normalized['Open'] = normalized['Close']
    if 'High' not in normalized.columns:
        normalized['High'] = normalized['Close']
    if 'Low' not in normalized.columns:
        normalized['Low'] = normalized['Close']
    if 'Volume' not in normalized.columns:
        normalized['Volume'] = np.nan

    normalized = normalized[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']].copy()
    normalized.dropna(subset=['Close'], inplace=True)
    normalized.sort_values('Date', inplace=True)
    normalized.reset_index(drop=True, inplace=True)
    return normalized


@st.cache_data(ttl=45)
def fetch_nse_quotes(symbols, refresh_key):
    if len(symbols) == 0:
        return {}

    quote_result = {}
    session = requests.Session()
    try:
        session.get('https://www.nseindia.com', headers=NSE_HEADERS, timeout=8)
    except Exception:
        pass

    for symbol in symbols:
        nse_symbol = symbol.replace('.NS', '').upper().strip()
        try:
            response = session.get(
                f'https://www.nseindia.com/api/quote-equity?symbol={nse_symbol}',
                headers=NSE_HEADERS,
                timeout=8
            )
            if response.status_code != 200:
                quote_result[symbol] = {'price': np.nan, 'source': 'NSE', 'ok': False, 'error': f'HTTP {response.status_code}'}
                continue
            payload = response.json()
            price_info = payload.get('priceInfo', {}) if isinstance(payload, dict) else {}
            last_price = price_info.get('lastPrice', np.nan)
            if isinstance(last_price, str):
                last_price = last_price.replace(',', '').strip()
            quote_result[symbol] = {
                'price': float(last_price),
                'source': 'NSE',
                'ok': True,
                'error': ''
            }
        except Exception as error:
            quote_result[symbol] = {'price': np.nan, 'source': 'NSE', 'ok': False, 'error': str(error)}

    return quote_result


def apply_latest_price_to_history(history_frame, latest_price):
    if history_frame is None or len(history_frame) == 0 or pd.isna(latest_price):
        return history_frame

    frame = history_frame.copy()
    frame['Date'] = pd.to_datetime(frame['Date'], errors='coerce')
    frame.dropna(subset=['Date'], inplace=True)
    frame.sort_values('Date', inplace=True)
    frame.reset_index(drop=True, inplace=True)
    if len(frame) == 0:
        return frame

    today = pd.Timestamp.now(tz='Asia/Kolkata').normalize().tz_localize(None)
    prev_close = float(frame['Close'].iloc[-1])
    prev_volume = float(frame['Volume'].iloc[-1]) if pd.notna(frame['Volume'].iloc[-1]) else np.nan

    new_row = {
        'Date': today,
        'Open': prev_close,
        'High': max(prev_close, float(latest_price)),
        'Low': min(prev_close, float(latest_price)),
        'Close': float(latest_price),
        'Volume': prev_volume
    }

    if frame['Date'].iloc[-1].date() == today.date():
        for key, value in new_row.items():
            frame.at[frame.index[-1], key] = value
    else:
        frame = pd.concat([frame, pd.DataFrame([new_row])], ignore_index=True)

    frame.sort_values('Date', inplace=True)
    frame.reset_index(drop=True, inplace=True)
    return frame


def rebase_history_to_price(history_frame, anchor_price):
    if history_frame is None or len(history_frame) == 0 or pd.isna(anchor_price):
        return history_frame

    frame = history_frame.copy()
    if 'Close' not in frame.columns:
        return frame
    last_close = float(frame['Close'].iloc[-1]) if pd.notna(frame['Close'].iloc[-1]) else np.nan
    if pd.isna(last_close) or last_close <= 0:
        return frame

    scale_factor = float(anchor_price) / last_close
    for column in ['Open', 'High', 'Low', 'Close']:
        if column in frame.columns:
            frame[column] = frame[column].astype(float) * scale_factor
    return frame


def compute_rsi(close_series, period=14):
    delta = close_series.diff()
    gains = delta.clip(lower=0)
    losses = -delta.clip(upper=0)
    avg_gain = gains.rolling(period).mean()
    avg_loss = losses.rolling(period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def compute_rse(close_series, window=20):
    data = close_series.dropna()
    if len(data) < window:
        return np.nan
    subset = data.iloc[-window:].values.astype(float)
    x_axis = np.arange(len(subset), dtype=float)
    slope, intercept = np.polyfit(x_axis, subset, 1)
    fitted = slope * x_axis + intercept
    return float(np.sqrt(np.mean((subset - fitted) ** 2)))


def compute_support_resistance(dataframe, window=20):
    if len(dataframe) < window:
        return np.nan, np.nan
    support = float(dataframe['Low'].rolling(window).min().iloc[-1])
    resistance = float(dataframe['High'].rolling(window).max().iloc[-1])
    return support, resistance


def compute_garch_vol(log_returns):
    returns = log_returns.dropna() * 100
    if len(returns) < 90:
        return np.nan
    try:
        model = arch_model(returns, mean='Zero', vol='Garch', p=1, q=1, dist='normal', rescale=False)
        fit = model.fit(disp='off', show_warning=False)
        variance_next = float(fit.forecast(horizon=1, reindex=False).variance.values[-1, 0])
        return float((np.sqrt(variance_next) / 100.0) * np.sqrt(252))
    except Exception:
        return np.nan


def compute_garch_vol_from_state(state):
    try:
        omega = float(state['garch_params']['omega'])
        alpha = float(state['garch_params']['alpha'])
        beta = float(state['garch_params']['beta'])
        sigma2_next = omega + alpha * float(state['last_eps2']) + beta * float(state['last_sigma2'])
        if sigma2_next <= 0:
            return np.nan
        return float((np.sqrt(sigma2_next) / 100.0) * np.sqrt(252))
    except Exception:
        return np.nan


def build_entry_plan(signal, live_price, support, resistance):
    if pd.isna(live_price):
        return np.nan, np.nan, np.nan, 'No live price available'

    live = float(live_price)
    support_val = float(support) if pd.notna(support) else live * 0.98
    resistance_val = float(resistance) if pd.notna(resistance) else live * 1.02

    if signal == 'BUY':
        entry = min(live, support_val * 1.01)
        stop_loss = min(entry * 0.98, support_val * 0.99)
        target = max(entry * 1.04, resistance_val)
        note = 'Buy on dip near support'
    elif signal == 'SELL':
        entry = max(live, resistance_val * 0.99)
        stop_loss = max(entry * 1.02, resistance_val * 1.01)
        target = min(entry * 0.96, support_val)
        note = 'Sell near resistance / weakness'
    else:
        entry = np.nan
        stop_loss = np.nan
        target = np.nan
        note = 'Wait: enter only near support (long) or near resistance rejection (short)'

    return float(entry) if pd.notna(entry) else np.nan, float(stop_loss) if pd.notna(stop_loss) else np.nan, float(target) if pd.notna(target) else np.nan, note


def build_feature_row(dataframe, garch_vol_1d):
    frame = dataframe.copy()
    frame['log_ret'] = np.log(frame['Close']).diff()
    frame['rv_21'] = frame['log_ret'].rolling(21).std() * np.sqrt(252)
    frame['rv_63'] = frame['log_ret'].rolling(63).std() * np.sqrt(252)
    for lag in [1, 2, 5]:
        frame[f'log_ret_lag_{lag}'] = frame['log_ret'].shift(lag)
        frame[f'rv_21_lag_{lag}'] = frame['rv_21'].shift(lag)

    latest = frame.iloc[-1]
    return pd.DataFrame([
        {
            'garch_vol_1d': float(garch_vol_1d),
            'rv_21': float(latest['rv_21']) if pd.notna(latest['rv_21']) else np.nan,
            'rv_63': float(latest['rv_63']) if pd.notna(latest['rv_63']) else np.nan,
            'log_ret_lag_1': float(latest['log_ret_lag_1']) if pd.notna(latest['log_ret_lag_1']) else np.nan,
            'log_ret_lag_2': float(latest['log_ret_lag_2']) if pd.notna(latest['log_ret_lag_2']) else np.nan,
            'log_ret_lag_5': float(latest['log_ret_lag_5']) if pd.notna(latest['log_ret_lag_5']) else np.nan,
            'rv_21_lag_1': float(latest['rv_21_lag_1']) if pd.notna(latest['rv_21_lag_1']) else np.nan,
            'rv_21_lag_2': float(latest['rv_21_lag_2']) if pd.notna(latest['rv_21_lag_2']) else np.nan,
            'rv_21_lag_5': float(latest['rv_21_lag_5']) if pd.notna(latest['rv_21_lag_5']) else np.nan
        }
    ])


def choose_reference_model(models, symbol):
    if symbol in models:
        return models[symbol], symbol
    if 'SBIN.NS' in models:
        return models['SBIN.NS'], 'SBIN.NS'
    first_key = list(models.keys())[0]
    return models[first_key], first_key


def sentiment_from_title(title):
    lowered = (title or '').lower()
    positive_hits = sum(1 for word in TITLE_POSITIVE if word in lowered)
    negative_hits = sum(1 for word in TITLE_NEGATIVE if word in lowered)
    score = positive_hits - negative_hits
    if score > 0:
        return 'Positive', score
    if score < 0:
        return 'Negative', score
    return 'Neutral', score


def classify_action_impact(action_row):
    if action_row is None:
        return 'Neutral', 'No recent corporate action captured.'
    dividends = action_row.get('Dividends', 0.0)
    splits = action_row.get('Stock Splits', 0.0)
    if pd.notna(dividends) and float(dividends) > 0:
        return 'Positive', 'Dividend action can support near-term sentiment.'
    if pd.notna(splits) and float(splits) > 0:
        return 'Neutral', 'Stock split usually improves liquidity more than fundamentals.'
    return 'Neutral', 'No high-signal corporate action detected in the latest entry.'


@st.cache_data(ttl=900)
def get_news(symbol, refresh_key):
    try:
        items = yf.Ticker(symbol).news or []
    except Exception:
        return []
    normalized = []
    for item in items[:10]:
        content = item.get('content', {}) if isinstance(item, dict) else {}
        title = content.get('title') or item.get('title') or 'Untitled'
        link = content.get('canonicalUrl', {}).get('url') if isinstance(content.get('canonicalUrl'), dict) else item.get('link')
        source = content.get('provider', {}).get('displayName') if isinstance(content.get('provider'), dict) else item.get('publisher', 'Unknown')
        ts = content.get('pubDate') or item.get('providerPublishTime')
        normalized.append({'title': title, 'link': link, 'source': source, 'published': ts})
    return normalized


@st.cache_data(ttl=1800)
def get_corporate_actions(symbol, refresh_key):
    try:
        actions = yf.Ticker(symbol).actions
        if actions is None or len(actions) == 0:
            return pd.DataFrame()
        actions = actions.reset_index().tail(10)
        return actions
    except Exception:
        return pd.DataFrame()


def score_bank(models, symbol, name, live_history, nse_quote):
    state, reference_name = choose_reference_model(models, symbol)

    frame = normalize_ohlcv_frame(live_history)
    data_source = 'Yahoo'

    if len(frame) < 100:
        model_history = normalize_ohlcv_frame(state.get('history_df', pd.DataFrame()))
        if len(model_history) > 0:
            frame = model_history
            data_source = 'Model snapshot'

    nse_price = np.nan
    nse_ok = False
    nse_error = ''
    if isinstance(nse_quote, dict):
        nse_price = nse_quote.get('price', np.nan)
        nse_ok = bool(nse_quote.get('ok', False)) and pd.notna(nse_price)
        nse_error = str(nse_quote.get('error', ''))

    if nse_ok:
        if data_source == 'Model snapshot':
            frame = rebase_history_to_price(frame, nse_price)
        frame = apply_latest_price_to_history(frame, nse_price)
        data_source = 'NSE quote + ' + data_source

    if len(frame) < 100:
        reason_text = 'Insufficient history for indicators'
        if nse_error:
            reason_text += f' | NSE quote error: {nse_error}'
        return {
            'Symbol': symbol,
            'Bank': name,
            'Status': 'No data',
            'Price Source': data_source,
            'Live Price': float(nse_price) if pd.notna(nse_price) else np.nan,
            'P(up)': np.nan,
            'Signal': 'HOLD',
            'Vol Forecast': np.nan,
            'MA20': np.nan,
            'MA50': np.nan,
            'RSI14': np.nan,
            'RSE20': np.nan,
            'Support': np.nan,
            'Resistance': np.nan,
            'Entry Price': np.nan,
            'Stop Loss': np.nan,
            'Target': np.nan,
            'Entry Note': 'No signal: insufficient data',
            'Ref Model': reference_name,
            'Chart Data': pd.DataFrame(),
            'Reason': reason_text
        }

    frame['MA20'] = frame['Close'].rolling(20).mean()
    frame['MA50'] = frame['Close'].rolling(50).mean()
    frame['RSI14'] = compute_rsi(frame['Close'], period=14)
    frame['LOG_RET'] = np.log(frame['Close']).diff()

    support, resistance = compute_support_resistance(frame, window=20)
    rse_value = compute_rse(frame['Close'], window=20)
    garch_vol = compute_garch_vol(frame['LOG_RET'])
    if pd.isna(garch_vol):
        garch_vol = compute_garch_vol_from_state(state)
    feature_row = build_feature_row(frame, garch_vol_1d=garch_vol)

    p_up = np.nan
    vol_forecast = np.nan
    status = 'Scored'
    reason = ''

    try:
        required_columns = state['feature_cols']
        model_input = feature_row.reindex(columns=required_columns)
        if model_input.isnull().any().any():
            status = 'Feature NA'
            reason = 'Feature window not ready for this refresh cycle'
        else:
            p_up = float(state['xgb_clf'].predict_proba(model_input.values)[0, 1])
            vol_forecast = float(state['xgb_reg'].predict(model_input.values)[0])
    except Exception as error:
        status = 'Inference error'
        reason = str(error)

    signal = 'HOLD'
    if pd.notna(p_up):
        if p_up >= 0.55:
            signal = 'BUY'
        elif p_up <= 0.45:
            signal = 'SELL'

    latest_row = frame.iloc[-1]
    entry_price, stop_loss, target_price, entry_note = build_entry_plan(
        signal=signal,
        live_price=float(latest_row['Close']) if pd.notna(latest_row['Close']) else np.nan,
        support=support,
        resistance=resistance
    )
    return {
        'Symbol': symbol,
        'Bank': name,
        'Status': status,
        'Price Source': data_source,
        'Live Price': float(latest_row['Close']),
        'P(up)': float(p_up) if pd.notna(p_up) else np.nan,
        'Signal': signal,
        'Vol Forecast': float(vol_forecast) if pd.notna(vol_forecast) else np.nan,
        'MA20': float(latest_row['MA20']) if pd.notna(latest_row['MA20']) else np.nan,
        'MA50': float(latest_row['MA50']) if pd.notna(latest_row['MA50']) else np.nan,
        'RSI14': float(latest_row['RSI14']) if pd.notna(latest_row['RSI14']) else np.nan,
        'RSE20': float(rse_value) if pd.notna(rse_value) else np.nan,
        'Support': support,
        'Resistance': resistance,
        'Entry Price': entry_price,
        'Stop Loss': stop_loss,
        'Target': target_price,
        'Entry Note': entry_note,
        'Ref Model': reference_name,
        'Chart Data': frame[['Date', 'Close', 'MA20', 'MA50', 'Volume']].tail(180),
        'Reason': reason
    }


models = load_models()

st.title('Public Sector Banks Live Hybrid GARCH + XGBoost Dashboard')
st.caption('Primary history feed: Yahoo Finance. Live price fallback: NSE direct quote API when Yahoo is throttled.')

with st.sidebar:
    st.subheader('Dashboard Controls')
    auto_refresh = st.toggle('Enable auto refresh', value=True)
    refresh_seconds = st.slider('Refresh interval (seconds)', min_value=60, max_value=600, value=180, step=30)
    lookback_period = st.selectbox('History window', ['6mo', '1y', '2y', '5y'], index=1)
    selected_banks = st.multiselect(
        'PSU banks to monitor',
        options=list(PSU_BANKS.keys()),
        default=list(PSU_BANKS.keys())
    )
    manual_refresh = st.button('Refresh now')

if auto_refresh:
    components.html(
        f"""
        <script>
        setTimeout(function() {{
            window.parent.location.reload();
        }}, {int(refresh_seconds) * 1000});
        </script>
        """,
        height=0,
        width=0
    )

refresh_key = int(time.time() // max(refresh_seconds, 1))
if manual_refresh:
    refresh_key = int(time.time())

if not selected_banks:
    st.warning('Select at least one PSU bank in the sidebar.')
    st.stop()

started_all = time.time()
history_by_symbol = fetch_bulk_histories(selected_banks, period=lookback_period, interval='1d', refresh_key=refresh_key)
nse_quotes = fetch_nse_quotes(selected_banks, refresh_key=refresh_key)

if 'last_good_history' not in st.session_state:
    st.session_state['last_good_history'] = {}

rows = []
for bank_symbol in selected_banks:
    current_history = history_by_symbol.get(bank_symbol, pd.DataFrame())
    if current_history is not None and len(current_history) > 0:
        st.session_state['last_good_history'][bank_symbol] = current_history
    else:
        current_history = st.session_state['last_good_history'].get(bank_symbol, pd.DataFrame())
    rows.append(
        score_bank(
            models=models,
            symbol=bank_symbol,
            name=PSU_BANKS[bank_symbol],
            live_history=current_history,
            nse_quote=nse_quotes.get(bank_symbol, {})
        )
    )

results_df = pd.DataFrame([{key: value for key, value in row.items() if key != 'Chart Data'} for row in rows])

if len(results_df) == 0:
    st.error('No live results were generated in this cycle.')
    st.stop()

if int((results_df['Status'] == 'No data').sum()) >= max(1, len(results_df) - 1):
    st.warning('Live provider is currently throttling requests. The dashboard is showing the most recent cached snapshot where available. Increase refresh interval to reduce throttling.')

signal_counts = results_df['Signal'].value_counts(dropna=False).to_dict()
inference_ms = int((time.time() - started_all) * 1000)
log_event('live_cycle_completed', {'rows': int(len(results_df)), 'signal_counts': signal_counts, 'elapsed_ms': inference_ms})

col1, col2, col3, col4 = st.columns(4)
col1.metric('Banks monitored', int(len(results_df)))
col2.metric('BUY signals', int((results_df['Signal'] == 'BUY').sum()))
col3.metric('SELL signals', int((results_df['Signal'] == 'SELL').sum()))
col4.metric('Cycle time (ms)', inference_ms)

st.subheader('Live Hybrid Signals and Technical Levels')
st.dataframe(
    results_df[['Symbol', 'Bank', 'Status', 'Price Source', 'Live Price', 'P(up)', 'Signal', 'Vol Forecast', 'MA20', 'MA50', 'RSI14', 'RSE20', 'Support', 'Resistance', 'Entry Price', 'Stop Loss', 'Target', 'Entry Note', 'Ref Model', 'Reason']],
    use_container_width=True,
    hide_index=True
)

st.subheader('Live Price Charts (All Selected Banks)')
for row in rows:
    chart_data = row['Chart Data']
    with st.expander(f"{row['Symbol']} · {row['Bank']} · Signal: {row['Signal']}", expanded=False):
        if chart_data is None or len(chart_data) == 0:
            st.caption('No chart data available for this symbol right now.')
        else:
            st.line_chart(chart_data.set_index('Date')[['Close', 'MA20', 'MA50']], use_container_width=True)
            st.bar_chart(chart_data.set_index('Date')[['Volume']], use_container_width=True)
            support_text = f"{row['Support']:.2f}" if pd.notna(row['Support']) else 'n/a'
            resistance_text = f"{row['Resistance']:.2f}" if pd.notna(row['Resistance']) else 'n/a'
            rsi_text = f"{row['RSI14']:.2f}" if pd.notna(row['RSI14']) else 'n/a'
            rse_text = f"{row['RSE20']:.4f}" if pd.notna(row['RSE20']) else 'n/a'
            st.caption(
                f"Support: {support_text} | Resistance: {resistance_text} | "
                f"RSI14: {rsi_text} | RSE20: {rse_text}"
            )

st.subheader('Market News and Corporate Actions Impact')
focus_symbol = st.selectbox('Choose bank for deep-dive news and actions', options=selected_banks, index=0)

news_items = get_news(focus_symbol, refresh_key=refresh_key)
actions_df = get_corporate_actions(focus_symbol, refresh_key=refresh_key)

news_table = []
for item in news_items:
    sentiment_label, sentiment_score = sentiment_from_title(item.get('title', ''))
    published = item.get('published')
    if isinstance(published, (int, float)):
        published_text = datetime.fromtimestamp(published, tz=timezone.utc).strftime('%Y-%m-%d %H:%M UTC')
    else:
        published_text = str(published) if published else 'n/a'
    news_table.append({
        'Published': published_text,
        'Source': item.get('source', 'Unknown'),
        'Sentiment': sentiment_label,
        'Score': sentiment_score,
        'Title': item.get('title', ''),
        'Link': item.get('link', '')
    })

if len(news_table) > 0:
    st.dataframe(pd.DataFrame(news_table), use_container_width=True, hide_index=True)
else:
    st.caption('No live news entries returned for this symbol in the current cycle.')

st.markdown('**Recent corporate actions**')
if actions_df is not None and len(actions_df) > 0:
    st.dataframe(actions_df, use_container_width=True, hide_index=True)
    latest_action = actions_df.iloc[-1].to_dict()
    impact_label, impact_text = classify_action_impact(latest_action)
    st.info(f'Corporate action impact ({impact_label}): {impact_text}')
else:
    st.caption('No recent corporate actions returned from the feed.')

st.markdown('**Indian market context**')
market_symbol = '^NSEBANK'
market_news = get_news(market_symbol, refresh_key=refresh_key)
if len(market_news) > 0:
    market_rows = []
    for item in market_news[:8]:
        sentiment_label, sentiment_score = sentiment_from_title(item.get('title', ''))
        market_rows.append({
            'Sentiment': sentiment_label,
            'Score': sentiment_score,
            'Title': item.get('title', ''),
            'Source': item.get('source', 'Unknown'),
            'Link': item.get('link', '')
        })
    st.dataframe(pd.DataFrame(market_rows), use_container_width=True, hide_index=True)
else:
    st.caption('No Bank Nifty market-news items returned at this time.')

st.subheader('Operational Notes')
st.caption(f'Last refresh cycle: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}. Logs are written to {LOG_PATH}.')
st.caption('Signals are model-driven estimates and not investment advice. Validate execution costs, liquidity, and risk controls before acting.')
