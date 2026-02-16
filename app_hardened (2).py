import os
import json
import time
import joblib
import numpy as np
import pandas as pd
import streamlit as st

# Optional live data
try:
    import yfinance as yf
    HAS_YF = True
except Exception:
    HAS_YF = False

st.set_page_config(page_title='Hybrid GARCH + XGBoost Banking Decision Helper', layout='wide')

LOG_PATH = os.environ.get('APP_LOG_PATH', '/tmp/app_events.jsonl')


def log_event(event_type, payload):
    try:
        rec = {
            'ts_utc': pd.Timestamp.utcnow().isoformat(),
            'event_type': str(event_type),
            'payload': payload
        }
        with open(LOG_PATH, 'a', encoding='utf-8') as f:
            f.write(json.dumps(rec) + '
')
    except Exception:
        # Logging must never break app
        pass


@st.cache_resource
def load_models():
    t0 = time.time()
    models_local = joblib.load('hybrid_models.joblib')
    load_ms = int((time.time() - t0) * 1000)
    log_event('models_loaded', {'n_models': len(models_local), 'load_ms': load_ms})
    return models_local


def health_check(models_local):
    checks = []
    checks.append({'check': 'models_present', 'ok': isinstance(models_local, dict) and len(models_local) > 0})

    # Check each model has required keys
    required_keys = ['ticker','feature_cols','garch_params','xgb_reg','xgb_clf','history_df','last_sigma2','last_eps2']
    for k in models_local.keys():
        stt = models_local[k]
        ok = True
        missing = []
        for rk in required_keys:
            if rk not in stt:
                ok = False
                missing.append(rk)
        checks.append({'check': 'state_keys_' + str(k), 'ok': ok, 'missing': missing})

        # Quick inference sanity check with last row
        try:
            hist = stt['history_df'].copy().sort_values('Date').reset_index(drop=True)
            if len(hist) < 70:
                checks.append({'check': 'history_len_' + str(k), 'ok': False, 'detail': 'need ~70+ rows, got ' + str(len(hist))})
                continue

            # Build features using dataset last close only
            omega = float(stt['garch_params']['omega'])
            alpha = float(stt['garch_params']['alpha'])
            beta = float(stt['garch_params']['beta'])
            sigma2_next = omega + alpha * float(stt['last_eps2']) + beta * float(stt['last_sigma2'])
            garch_vol_1d = (np.sqrt(float(sigma2_next)) / 100.0) * np.sqrt(252)

            last = hist.iloc[-1]
            feat = {
                'garch_vol_1d': float(garch_vol_1d),
                'rv_21': float(last['rv_21']),
                'rv_63': float(last['rv_63']),
                'log_ret_lag_1': float(last['log_ret_lag_1']),
                'log_ret_lag_2': float(last['log_ret_lag_2']),
                'log_ret_lag_5': float(last['log_ret_lag_5']),
                'rv_21_lag_1': float(last['rv_21_lag_1']),
                'rv_21_lag_2': float(last['rv_21_lag_2']),
                'rv_21_lag_5': float(last['rv_21_lag_5'])
            }
            feat_df = pd.DataFrame([feat])[stt['feature_cols']]
            if feat_df.isnull().any().any():
                checks.append({'check': 'feature_nan_' + str(k), 'ok': False})
                continue

            _ = float(stt['xgb_reg'].predict(feat_df.values)[0])
            p = float(stt['xgb_clf'].predict_proba(feat_df.values)[0, 1])
            checks.append({'check': 'inference_' + str(k), 'ok': np.isfinite(p) and p >= 0.0 and p <= 1.0})
        except Exception as e:
            checks.append({'check': 'inference_' + str(k), 'ok': False, 'error': str(e)})

    return pd.DataFrame(checks)


models = load_models()

st.title('Hybrid GARCH + XGBoost Banking Decision Helper')

with st.expander('User guide (quick)', expanded=True):
    st.markdown(
        """
This app is a *decision helper*, not a trading system.

What you get
- *P(up tomorrow)*: the model\'s estimate that next day return is positive
- *Vol proxy forecast*: the model\'s estimate of next-day volatility proxy (annualized scale)
- A *simple stance* rule: `BUY` if P(up) >= 0.55, `SELL` if <= 0.45 else `HOLD`

How to use it
- Pick an instrument
- Choose a price source (dataset, live Yahoo, or manual)
- Treat anything near 0.50 as low conviction

Important caveats
- Direction prediction is noisy at daily horizon
- Live data can fail (symbol, network, market hours) and will fallback
- Always validate on your own period and include costs/slippage
"""
    )

with st.expander('Health checks', expanded=False):
    health_df = health_check(models)
    st.dataframe(health_df, use_container_width=True)
    if bool((~health_df['ok']).any()):
        st.warning('Some checks failed. The app may still run, but you should fix the failing parts before relying on it.')
    else:
        st.success('All checks passed.')

st.write('---')

ticker = st.selectbox('Instrument', list(models.keys()), index=0)
state = models[ticker]

hist = state['history_df'].copy().sort_values('Date').reset_index(drop=True)

st.subheader('History snapshot (from training dataset)')
st.dataframe(hist.tail(12), use_container_width=True)

st.subheader('Live price ingestion')

col_a, col_b, col_c = st.columns([1.2, 1.2, 2.0])
with col_a:
    live_mode = st.radio('Price source', ['Use dataset last close', 'Fetch live (Yahoo Finance)', 'Manual entry'], index=0)
with col_b:
    lookback_days = st.number_input('Live lookback days', min_value=30, max_value=3650, value=365, step=30)
with col_c:
    st.caption('Live fetch is best-effort. If it fails, the app falls back to dataset close.')

manual_close = None
if live_mode == 'Manual entry':
    manual_close = st.number_input('Enter latest Close price', min_value=0.0001, value=float(hist['Close'].iloc[-1]))

YF_MAP = {
    '^NSEBANK': '^NSEBANK',
    'ICICIBANK.NS': 'ICICIBANK.NS',
    'SBIN.NS': 'SBIN.NS',
    'HDFCBANK.NS': 'HDFCBANK.NS'
}


def fetch_latest_close(tkr):
    if not HAS_YF:
        return None, 'yfinance not installed'
    sym = YF_MAP.get(tkr, tkr)
    try:
        dl = yf.download(sym, period=str(int(lookback_days)) + 'd', interval='1d', auto_adjust=False, progress=False)
        if dl is None or len(dl) == 0:
            return None, 'no data returned'
        if 'Close' in dl.columns and pd.notna(dl['Close'].iloc[-1]):
            return float(dl['Close'].iloc[-1]), 'ok'
        if 'Adj Close' in dl.columns and pd.notna(dl['Adj Close'].iloc[-1]):
            return float(dl['Adj Close'].iloc[-1]), 'ok_adj_close'
        return None, 'close missing'
    except Exception as e:
        return None, str(e)


def build_features(hist_df, latest_close_override=None):
    df = hist_df.copy().sort_values('Date').reset_index(drop=True)

    if latest_close_override is not None:
        last_close = float(df['Close'].iloc[-1])
        new_log_ret = np.log(float(latest_close_override)) - np.log(last_close)
        new_row = df.iloc[-1].copy()
        new_row['Date'] = pd.Timestamp.today().normalize()
        new_row['Close'] = float(latest_close_override)
        new_row['log_ret'] = float(new_log_ret)
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

        df['rv_21'] = df['log_ret'].rolling(21).std() * np.sqrt(252)
        df['rv_63'] = df['log_ret'].rolling(63).std() * np.sqrt(252)
        for lag in [1, 2, 5]:
            df['log_ret_lag_' + str(lag)] = df['log_ret'].shift(lag)
            df['rv_21_lag_' + str(lag)] = df['rv_21'].shift(lag)

    omega = float(state['garch_params']['omega'])
    alpha = float(state['garch_params']['alpha'])
    beta = float(state['garch_params']['beta'])
    sigma2_next = omega + alpha * float(state['last_eps2']) + beta * float(state['last_sigma2'])
    garch_vol_1d = (np.sqrt(float(sigma2_next)) / 100.0) * np.sqrt(252)

    last = df.iloc[-1]
    feat = {
        'garch_vol_1d': float(garch_vol_1d),
        'rv_21': float(last['rv_21']),
        'rv_63': float(last['rv_63']),
        'log_ret_lag_1': float(last['log_ret_lag_1']),
        'log_ret_lag_2': float(last['log_ret_lag_2']),
        'log_ret_lag_5': float(last['log_ret_lag_5']),
        'rv_21_lag_1': float(last['rv_21_lag_1']),
        'rv_21_lag_2': float(last['rv_21_lag_2']),
        'rv_21_lag_5': float(last['rv_21_lag_5'])
    }

    return pd.DataFrame([feat])[state['feature_cols']]


latest_close = None
close_source_note = None

if live_mode == 'Use dataset last close':
    latest_close = float(hist['Close'].iloc[-1])
    close_source_note = 'dataset'

if live_mode == 'Fetch live (Yahoo Finance)':
    live_close, live_msg = fetch_latest_close(ticker)
    if live_close is None:
        latest_close = float(hist['Close'].iloc[-1])
        close_source_note = 'dataset_fallback (' + str(live_msg) + ')'
    else:
        latest_close = float(live_close)
        close_source_note = 'yahoo (' + str(live_msg) + ')'

if live_mode == 'Manual entry':
    latest_close = float(manual_close)
    close_source_note = 'manual'

log_event('prediction_requested', {'ticker': ticker, 'price_source': close_source_note, 'close_used': float(latest_close)})

st.write('Latest close used ' + str(round(latest_close, 4)) + ' from ' + str(close_source_note))

feat_df = build_features(hist, latest_close_override=None if live_mode == 'Use dataset last close' else latest_close)

if feat_df.isnull().any().any():
    st.warning('Not enough history to compute features (need ~63+ trading days).')
    log_event('prediction_failed', {'ticker': ticker, 'reason': 'feature_nan'})
else:
    t0 = time.time()
    vol_pred = float(state['xgb_reg'].predict(feat_df.values)[0])
    dir_proba = float(state['xgb_clf'].predict_proba(feat_df.values)[0, 1])
    infer_ms = int((time.time() - t0) * 1000)

    if dir_proba >= 0.55:
        stance = 'BUY'
    elif dir_proba <= 0.45:
        stance = 'SELL'
    else:
        stance = 'HOLD'

    st.subheader('Model outputs')
    c1, c2, c3, c4 = st.columns(4)
    c1.metric('P(up tomorrow)', str(round(dir_proba, 3)))
    c2.metric('Vol proxy forecast (annualized)', str(round(vol_pred, 3)))
    c3.metric('Suggested stance (simple rule)', stance)
    c4.metric('Inference time (ms)', str(infer_ms))

    st.subheader('Features used (last row)')
    st.dataframe(feat_df, use_container_width=True)

    log_event('prediction_succeeded', {
        'ticker': ticker,
        'p_up': float(dir_proba),
        'vol_pred': float(vol_pred),
        'stance': stance,
        'infer_ms': infer_ms
    })



with st.expander('Log viewer (last 50 lines)', expanded=False):
    try:
        if os.path.exists(LOG_PATH):
            with open(LOG_PATH, 'r', encoding='utf-8') as f:
                log_lines = f.readlines()[-50:]
            st.code(''.join(log_lines))
        else:
            st.caption('No log file found at ' + str(LOG_PATH) + ' yet. Make a prediction to generate logs.')
    except Exception as e:
        st.caption('Could not read logs: ' + str(e))

st.subheader('Ops notes')
st.caption('Logs are appended to ' + str(LOG_PATH) + ' as JSONL. Set env var APP_LOG_PATH to change location.')

st.subheader('Run locally')
st.code('pip install -r requirements_live.txt\nstreamlit run app_hardened.py')
