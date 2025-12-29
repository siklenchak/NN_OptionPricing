import streamlit as st
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model
from scipy.stats import norm

# ============================================
# 1. НАЛАШТУВАННЯ ТА ЗАВАНТАЖЕННЯ
# ============================================
st.set_page_config(page_title="Options Price Predictor", layout="wide")

st.title("💰 AI Options Pricing Calculator")
st.markdown("Порівняння цін: **LSTM (AI)** vs **Black-Scholes (Math)** vs **Historical Market Data**")

# Функція кешування, щоб не вантажити модель при кожному кліку
@st.cache_resource
def load_resources():
    model = load_model('best_atm_model.keras')
    scaler_X = joblib.load('scaler_X.pkl')
    scaler_y = joblib.load('scaler_y.pkl')
    df_history = pd.read_csv('historical_data.csv')
    df_history['QUOTE_DATE'] = pd.to_datetime(df_history['QUOTE_DATE'])
    return model, scaler_X, scaler_y, df_history

# Завантажуємо ресурси
try:
    model, scaler_X, scaler_y, df_history = load_resources()
    st.success("Система готова! Модель та дані завантажено.")
except Exception as e:
    st.error(f"Помилка завантаження файлів. Перевірте наявність .keras, .pkl та .csv файлів. Деталі: {e}")
    st.stop()

# ============================================
# 2. БОКОВА ПАНЕЛЬ (ВВЕДЕННЯ ДАНИХ)
# ============================================
st.sidebar.header("Параметри Опціону")

# Вводимо параметри
S = st.sidebar.number_input("Ціна Активу (Underlying Price)", value=150.0, step=1.0)
K = st.sidebar.number_input("Страйк (Strike Price)", value=150.0, step=1.0)
dte = st.sidebar.number_input("Днів до експірації (DTE)", value=30, step=1, min_value=1)
volatility = st.sidebar.number_input("Волатильність (Sigma)", value=0.25, step=0.01, format="%.2f")
risk_free_rate = st.sidebar.number_input("Безризикова ставка (Risk-Free Rate)", value=0.05, step=0.01)

# Розрахункові змінні
T = dte / 365.0 # Час у роках

# ============================================
# 3. ЛОГІКА РОЗРАХУНКІВ
# ============================================

if st.button("🚀 Розрахувати ціну", type="primary"):
    
    # Спочатку шукаємо історичний аналог, щоб мати базу для порівняння
    # --- В. ПОШУК ІСТОРИЧНОГО АНАЛОГА ---
    target_moneyness = S / K
    history_subset = df_history[
        (df_history['DTE'] >= dte - 5) & 
        (df_history['DTE'] <= dte + 5)
    ].copy()
    
    hist_price = None # Початкове значення
    best_match_date = None

    if len(history_subset) > 0:
        history_subset['moneyness'] = history_subset['UNDERLYING_LAST'] / history_subset['STRIKE']
        history_subset['similarity_score'] = abs(history_subset['moneyness'] - target_moneyness)
        best_match = history_subset.sort_values('similarity_score').iloc[0]
        hist_price = best_match['C_LAST']
        best_match_date = best_match['QUOTE_DATE'].date()

    # --- А. BLACK-SCHOLES ---
    def calculate_bs(S, K, T, r, sigma):
        if T <= 0 or sigma <= 0: return max(S - K, 0.0)
        d1 = (np.log(S/K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

    bs_price = calculate_bs(S, K, T, risk_free_rate, volatility)

    # --- Б. LSTM НЕЙРОМЕРЕЖА ---
    moneyness_val = S / K
    input_data = pd.DataFrame({
        'UNDERLYING_LAST': [S],
        'T_years': [T],
        'vol_garch': [volatility],
        'Moneyness': [moneyness_val]
    })
    
    try:
        X_scaled = scaler_X.transform(input_data)
        X_reshaped = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))
        lstm_pred_scaled = model.predict(X_reshaped, verbose=0)
        lstm_price = scaler_y.inverse_transform(lstm_pred_scaled)[0][0]
        lstm_price = max(lstm_price, 0.0)
    except Exception as e:
        st.error(f"Помилка LSTM: {e}")
        lstm_price = 0.0

    # ============================================
    # ВІДОБРАЖЕННЯ РЕЗУЛЬТАТІВ З ВІДСОТКАМИ
    # ============================================
    col1, col2, col3 = st.columns(3)

    # Логіка для Delta (різниці)
    # Якщо є історія, порівнюємо з нею. Якщо немає - показуємо просто ціну.
    
    # 1. LSTM
    lstm_delta = None
    if hist_price:
        diff_pct = ((lstm_price - hist_price) / hist_price) * 100
        # Форматуємо: наприклад "+5.2% vs History"
        lstm_delta = f"{diff_pct:+.2f}% vs History"
    
    with col1:
        st.metric(label="🤖 LSTM AI Prediction", value=f"${lstm_price:.2f}", delta=lstm_delta)

    # 2. Black-Scholes
    bs_delta = None
    if hist_price:
        diff_pct = ((bs_price - hist_price) / hist_price) * 100
        bs_delta = f"{diff_pct:+.2f}% vs History"

    with col2:
        st.metric(label="📊 Black-Scholes Formula", value=f"${bs_price:.2f}", delta=bs_delta)

    # 3. History
    with col3:
        if hist_price:
            st.metric(label="📜 Historical Match", value=f"${hist_price:.2f}", delta=f"Date: {best_match_date}", delta_color="off")
            
            # === НОВЕ: Розгортаємо деталі ===
            with st.expander("ℹ️ Повні параметри збігу"):
                # Формуємо красивий словник або таблицю з даними цього рядка
                # best_match - це pandas Series, беремо з неї потрібне
                
                details_data = {
                    'Дата (Quote Date)': str(best_match['QUOTE_DATE'].date()),
                    'Ціна Активу (S)': f"${best_match['UNDERLYING_LAST']:.2f}",
                    'Страйк (K)': f"${best_match['STRIKE']:.2f}",
                    'DTE (Днів)': int(best_match['DTE']),
                    'Волатильність (GARCH)': f"{best_match['vol_garch']:.4f}",
                    'Moneyness (S/K)': f"{best_match['moneyness']:.4f}",
                    'Реальна ціна (Market)': f"${best_match['C_LAST']:.2f}"
                }
                
                # Виводимо як таблицю або JSON
                st.table(pd.DataFrame(list(details_data.items()), columns=['Параметр', 'Значення']))
                
        else:
            st.warning("Історичних аналогів не знайдено")

    # ============================================
    # 4. ТАБЛИЦЯ ПОРІВНЯННЯ (COMPARISON TABLE)
    # ============================================
    if hist_price:
        st.divider()
        st.subheader("📉 Детальне порівняння похибок")
        
        # Створюємо таблицю
        comp_data = {
            'Model': ['Historical (Real)', 'LSTM (AI)', 'Black-Scholes'],
            'Price ($)': [hist_price, lstm_price, bs_price],
            'Difference ($)': [0.0, lstm_price - hist_price, bs_price - hist_price],
            'Error (%)': [0.0, ((lstm_price - hist_price)/hist_price)*100, ((bs_price - hist_price)/hist_price)*100]
        }
        
        df_comp = pd.DataFrame(comp_data)
        
        # Округлення для краси
        df_comp['Price ($)'] = df_comp['Price ($)'].map('${:,.2f}'.format)
        df_comp['Difference ($)'] = df_comp['Difference ($)'].map('{:+,.2f}'.format)
        df_comp['Error (%)'] = df_comp['Error (%)'].map('{:+.2f}%'.format)

        # Виводимо стилізовану таблицю
        st.table(df_comp.set_index('Model'))
        
        # Розрахунок помилок (абсолютне значення для порівняння)
        lstm_err = abs((lstm_price - hist_price)/hist_price)
        bs_err = abs((bs_price - hist_price)/hist_price)
        
        # Формуємо рядок з деталями історичного матчу
        match_details = f"порівняно з реальною угодою від **{best_match_date}** (Ціна: **${hist_price:.2f}**)"
        
        # Виводимо висновок + деталі матчу
        if lstm_err < bs_err:
            st.success(f"✅ **Висновок:** Нейромережа виявилася точнішою! Її похибка **{lstm_err:.1%}**, проти **{bs_err:.1%}** у формули.\n\nResult based on: {match_details}")
        else:
            st.info(f"ℹ️ **Висновок:** Класична формула тут спрацювала краще. Похибка LSTM: **{lstm_err:.1%}**.\n\nResult based on: {match_details}")