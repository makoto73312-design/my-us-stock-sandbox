import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# --- 1. 網頁核心外觀配置 ---
st.set_page_config(page_title="美股雷達", page_icon="🔮", layout="wide")
st.title("🔮 量化投資沙盒 V05.2 (動態倉位追蹤版)")
st.markdown("已實裝 **華爾街 Regime-Switching 引擎**、**智能區塊底色** 與 🌟**V05.2 動態部位追蹤、真實進場成本與未實現損益監控**")

# --- 2. 側邊欄控制台 ---
st.sidebar.header("⚙️ 全自動大掃描設定")

# 🚨🚨🚨 請記得將你的【美股 Google 試算表】共用連結貼在下方的引號裡面 🚨🚨🚨
GSHEET_URL = "https://docs.google.com/spreadsheets/d/1YuF63YTtUfzGQ70Wu1Bc_xSF9VxBAfFQGYHdP6-WUFk/edit?usp=drive_link"

@st.cache_data(ttl=60)
def get_tickers_from_sheet(url):
    try:
        if "docs.google.com" not in url:
            return "NVDA, AAPL, TSLA, MSFT, AMD"
        csv_url = url.split("/edit")[0] + "/export?format=csv"
        df = pd.read_csv(csv_url, header=None)
        tickers = df.iloc[:, 0].dropna().astype(str).str.strip().str.upper().tolist()
        valid_tickers = [t for t in tickers if not any(c >= '\u4e00' and c <= '\u9fff' for c in t) and len(t) > 0]
        if not valid_tickers:
            return "NVDA, AAPL, TSLA, MSFT, AMD"
        return ", ".join(valid_tickers)
    except Exception as e:
        return "NVDA, AAPL, TSLA, MSFT, AMD"

default_tickers = get_tickers_from_sheet(GSHEET_URL)

tickers_input = st.sidebar.text_area(
    "📡 當前雲端同步清單 (新增股票請至試算表修改)", 
    default_tickers, 
    height=100
)
ticker_list = [t.strip().upper() for t in tickers_input.split(',') if t.strip()]

backtest_days = st.sidebar.slider("歷史回測天數設定", min_value=100, max_value=500, value=300, step=50)

market_posture = st.sidebar.selectbox(
    "⚖️ 當前市場防禦姿態 (環境切換開關)",
    ["🛡️ 標準平衡型", "🚀 大膽進攻型", "🥶 極度謹慎型"],
    index=0
)

# --- 3. 輔助功能 ---
def draw_progress_bar(score, active_char):
    filled_count = int(round(score / 10))
    filled_count = max(0, min(10, filled_count))
    empty_count = 10 - filled_count
    if filled_count == 0 and score > 0: 
        filled_count = 1
        empty_count = 9
    return f"[{active_char * filled_count}{'░' * empty_count}]"

# --- 4. 技術指標核心 ---
def calculate_indicators(df):
    high_low_diff = (df['High'] - df['Low']).replace(0, 0.001) 
    mf_multiplier = ((df['Close'] - df['Low']) - (df['High'] - df['Close'])) / high_low_diff
    df['主力籌碼'] = (df['Volume'] * mf_multiplier / 1000000).round(2)
    
    df['MA5'] = df['Close'].rolling(5).mean()
    df['MA10'] = df['Close'].rolling(10).mean()
    df['MA14'] = df['Close'].rolling(14).mean()
    df['MA20'] = df['Close'].rolling(20).mean()
    df['MA21'] = df['Close'].rolling(21).mean()
    df['MA30'] = df['Close'].rolling(30).mean()
    df['50MA'] = df['Close'].rolling(50).mean()
    df['200MA'] = df['Close'].rolling(200).mean()
    df['ROC14'] = df['Close'].pct_change(14)
    
    delta = df['Close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    ema_gain = gain.ewm(com=13, adjust=False).mean()
    ema_loss = loss.ewm(com=13, adjust=False).mean()
    rs = ema_gain / ema_loss.replace(0, 0.001)
    df['RSI_14'] = 100 - (100 / (1 + rs))
    
    df['Vol_MA20'] = df['Volume'].rolling(20).mean()
    
    df['主力籌碼_Q80'] = df['主力籌碼'].rolling(50).quantile(0.8)
    df['主力籌碼_Q90'] = df['主力籌碼'].rolling(50).quantile(0.9)
    df['主力籌碼_Q95'] = df['主力籌碼'].rolling(50).quantile(0.95)
    
    ema12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = ema12 - ema26
    df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Hist'] = df['MACD'] - df['Signal']
    
    macd_shrink = [0] * len(df)
    hist = df['MACD_Hist'].values
    for i in range(1, len(df)):
        if hist[i] < 0 and hist[i] > hist[i-1]:
            macd_shrink[i] = macd_shrink[i-1] + 1
        else:
            macd_shrink[i] = 0
    df['MACD_Shrink'] = macd_shrink
    return df

# --- 5. 歷史回測引擎 (🌟 V05.2 倉位與進場成本追蹤) ---
def run_backtest_engine(df, strategy_name, days, posture):
    valid_df = df.dropna(subset=['200MA', 'ROC14', 'MACD_Hist', 'RSI_14', 'Vol_MA20', '主力籌碼_Q80', '主力籌碼_Q90', '主力籌碼_Q95']).tail(days).copy()
    if len(valid_df) < 5:
        # 🟢 V05.2 修正：補足 14 個回傳值，完美對齊
        return "⚠️ 數據不足", 0, 0, 0, 0, "❌ 不推薦", "🛑 數據不足", "-", "-", "-", [], [], [], valid_df
    
    if "🚀 大膽進攻型" in posture: rsi_max, vol_mult, dip_pct, rsi_min, chip_col = 75, 1.05, -0.10, 35, '主力籌碼_Q80'
    elif "🥶 極度謹慎型" in posture: rsi_max, vol_mult, dip_pct, rsi_min, chip_col = 65, 1.50, -0.15, 25, '主力籌碼_Q95'
    else: rsi_max, vol_mult, dip_pct, rsi_min, chip_col = 70, 1.20, -0.10, 30, '主力籌碼_Q90'

    if "A:" in strategy_name: s_ma, d_ma, stop_loss_pct = valid_df['MA5'], valid_df['MA14'], 0.05
    elif "B:" in strategy_name: s_ma, d_ma, stop_loss_pct = valid_df['MA14'], valid_df['MA21'], 0.075
    elif "C:" in strategy_name: s_ma, d_ma, stop_loss_pct = valid_df['MA10'], valid_df['MA30'], 0.10
    elif "D:" in strategy_name: s_ma, d_ma, stop_loss_pct = valid_df['MA20'], valid_df['200MA'], 0.05
    else: s_ma, d_ma, stop_loss_pct = valid_df['MA5'], valid_df['MA20'], 0.06

    max_ma, min_ma = valid_df[['MA5', 'MA14', '50MA']].max(axis=1), valid_df[['MA5', 'MA14', '50MA']].min(axis=1)
    is_entangled_series = ((max_ma - min_ma) / valid_df['50MA'].replace(0, 0.001)) < 0.025
    is_entangled_arr = is_entangled_series.values

    closes, highs, lows = valid_df['Close'].values, valid_df['High'].values, valid_df['Low'].values
    s_mas, d_mas, m200s, r14s, rsis = s_ma.values, d_ma.values, valid_df['200MA'].values, valid_df['ROC14'].values, valid_df['RSI_14'].values
    vols, vol_m20s = valid_df['Volume'].values, valid_df['Vol_MA20'].values
    m_shrinks, m_hists, m_flows, chip_threshs = valid_df['MACD_Shrink'].values, valid_df['MACD_Hist'].values, valid_df['主力籌碼'].values, valid_df[chip_col].values

    has_position = False
    entry_price, highest_price_since_entry = 0, 0
    total_trades, win_trades = 0, 0
    total_return, total_gross_profit, total_gross_loss = 0.0, 0.0, 0.0
    trade_logs, plot_buys, plot_sells = [], [], []

    for i in range(len(valid_df)):
        date_str = valid_df.index[i].strftime('%Y-%m-%d')
        close_p, high_p, low_p = closes[i], highs[i], lows[i]
        sma_p, dma_p, m200_p, r14_p, rsi_p = s_mas[i], d_mas[i], m200s[i], r14s[i], rsis[i]
        vol_p, vol_m20_p = vols[i], vol_m20s[i]
        m_shrink_p, m_hist_p, m_flow_p, chip_thresh_p = m_shrinks[i], m_hists[i], m_flows[i], chip_threshs[i]
        m_hist_y = m_hists[i-1] if i > 0 else 0
        is_entangled = is_entangled_arr[i]

        if not has_position:
            is_buy = False
            if "A:" in strategy_name:
                if (m_shrink_p >= 1 or (m_hist_p > m_hist_y and m_hist_p > 0)) and r14_p > 0 and rsi_p < rsi_max: is_buy = True
            elif "B:" in strategy_name or "C:" in strategy_name:
                if (not is_entangled) and close_p > sma_p and sma_p > dma_p and vol_p > vol_m20_p * vol_mult: is_buy = True
            elif "D:" in strategy_name:
                if m200_p > 0 and (close_p - m200_p)/m200_p <= dip_pct and m_shrink_p >= 1 and rsi_p < rsi_min: is_buy = True
            elif "E:" in strategy_name:
                if m_flow_p > chip_thresh_p and m_flow_p > 0: is_buy = True
            
            if is_buy:
                has_position, entry_price, highest_price_since_entry = True, close_p, close_p
                total_trades += 1
                trade_logs.append({"交易日期": date_str, "動作狀態": "🟢 買入進場 (BUY)", "執行價格": f"${close_p:.2f}", "單筆報酬": "-"})
                plot_buys.append((valid_df.index[i], close_p))
        else:
            highest_price_since_entry = max(highest_price_since_entry, high_p)
            is_exit, exit_price = False, close_p

            if "D:" not in strategy_name:
                if low_p <= highest_price_since_entry * (1 - stop_loss_pct):
                    is_exit, exit_price = True, highest_price_since_entry * (1 - stop_loss_pct)
                elif ("B:" in strategy_name or "C:" in strategy_name) and is_entangled:
                    is_exit, exit_price = True, close_p
            else:
                if high_p >= m200_p: is_exit, exit_price = True, m200_p
                elif low_p <= entry_price * 0.95: is_exit, exit_price = True, entry_price * 0.95

            if is_exit:
                trade_return = (exit_price - entry_price) / entry_price
                total_return += trade_return
                if trade_return > 0: win_trades += 1; total_gross_profit += trade_return
                else: total_gross_loss += abs(trade_return)
                has_position = False
                trade_logs.append({"交易日期": date_str, "動作狀態": "🔴 賣出出場 (SELL)", "執行價格": f"${exit_price:.2f}", "單筆報酬": f"{trade_return*100:+.2f}%"})
                plot_sells.append((valid_df.index[i], exit_price))

    # --- 🌟 V05.2 核心：狀態追蹤與輸出 ---
    final_win_rate = win_trades / total_trades if total_trades > 0 else 0.0
    profit_factor = total_gross_profit / total_gross_loss if total_gross_loss > 0 else (99.9 if total_gross_profit > 0 else 0.0)
    pf_str = "無限" if profit_factor == 99.9 else f"{profit_factor:.2f}"

    stars = "❌ 不推薦"
    if total_return > 0 and total_trades > 0:
        if total_return >= 0.25 and final_win_rate >= 0.55: stars = "⭐⭐⭐⭐⭐"
        elif total_return >= 0.15 or final_win_rate >= 0.50: stars = "⭐⭐⭐⭐"
        else: stars = "⭐⭐"

    last_action = trade_logs[-1] if len(trade_logs) > 0 else None
    today_str = valid_df.index[-1].strftime('%Y-%m-%d')
    current_close = closes[-1]
    
    current_status = "💵 空手觀望 (CASH)"
    entry_price_str = "-"  # 🟢 V05.2 新增：建議進場價 / 真實持股成本
    sl_price_str = "-"
    pnl_str = "-"

    if has_position:
        # 如果最後一筆交易是今天買入
        if last_action and last_action["交易日期"] == today_str and "BUY" in last_action["動作狀態"]:
            current_status = "🚀 今日大膽建倉 (BUY)"
            unrealized_pnl = 0.0
            entry_price_str = f"${current_close:.2f}"
        else:
            current_status = "📦 獲利續抱中 (HOLD)"
            unrealized_pnl = (current_close - entry_price) / entry_price
            pnl_str = f"{unrealized_pnl*100:+.2f}%"
            entry_price_str = f"${entry_price:.2f}"  # 🌟 續抱中直接顯示當初真實買入成本！

        if "D:" not in strategy_name: 
            sl_price_str = f"${highest_price_since_entry * (1 - stop_loss_pct):.2f}"
        else: 
            sl_price_str = f"${max(entry_price * 0.95, m200s[-1]):.2f}"
    else:
        # 如果最後一筆交易是今天賣出
        if last_action and last_action["交易日期"] == today_str and "SELL" in last_action["動作狀態"]:
            current_status = "🔴 今日觸發防守賣出 (SELL)"
        else:
            current_status = "💵 空手觀望 (CASH)"

    # 🟢 回傳值完美對齊 (增加了 entry_price_str)
    return "📡 運算完畢", total_return, final_win_rate, total_trades, pf_str, stars, current_status, entry_price_str, sl_price_str, pnl_str, trade_logs, plot_buys, plot_sells, valid_df

# --- 6. Session State 記憶庫 ---
if "calculated" not in st.session_state:
    st.session_state.calculated = False
    st.session_state.final_df = None
    st.session_state.detail_db = {}
    st.session_state.last_posture = ""

if st.session_state.calculated and st.session_state.last_posture != market_posture:
    st.session_state.calculated = False

# --- 7. 網頁分頁與渲染 ---
tab_summary, tab_debug = st.tabs(["📊 倉位追蹤綜合矩陣", "🔍 深度數據與視覺化對照面板"])

with tab_summary:
    st.info(f"💡 當前系統姿態：**{market_posture}**。")
    if st.button("🚀 啟動全自動大腦跨標的回測引擎", use_container_width=True):
        with st.spinner(f"正在以 【{market_posture}】 模式刷新美股大數據..."):
            master_report, strategies = [], ["A: 激進動能型", "B: 穩健波段型", "C: 槓桿防守型", "D: 均值回歸抄底型", "E: 籌碼主力跟隨型"]
            for ticker in ticker_list:
                df_stock = yf.download(ticker, period="2y", progress=False)
                if df_stock.empty: continue
                df_stock.columns = [col[0] if isinstance(col, tuple) else col for col in df_stock.columns]
                df_stock = calculate_indicators(df_stock)
                
                # 🟢 【V05.1 安全防護補丁】：先剃除沒有收盤價的空白列（解決美股盤前或例假日造成的 NaN 問題）
                df_temp_clean = df_stock.dropna(subset=['Close'])
                current_close = float(df_temp_clean['Close'].iloc[-1]) if not df_temp_clean.empty else 0.0
                
                for strat in strategies:
                    # 🟢 V05.2 修改：解構時加入 entry_price_val 變數
                    radar, ret, win, trades, pf, stars, cur_status, entry_price_val, sl_price, pnl, t_logs, p_buys, p_sells, v_df = run_backtest_engine(df_stock, strat, backtest_days, market_posture)
                    st.session_state.detail_db[(ticker, strat)] = {"logs": pd.DataFrame(t_logs), "buys": p_buys, "sells": p_sells, "v_df": v_df}
                    master_report.append({
                        "股票代號": ticker, 
                        "當前市價": f"${current_close:.2f}", 
                        "策略手法": strat,
                        "倉位狀態": cur_status,
                        "建議進場價 (或持股成本)": entry_price_val, # 🟢 V05.2 新增：回歸顯示
                        "未實現損益": pnl,
                        "嚴格防守/停損價": sl_price,
                        "總報酬率": f"{ret * 100:+.2f}%", 
                        "歷史勝率": f"{win * 100:.1f}%",
                        "交易次數": trades, 
                        "獲利因子": pf, 
                        "推薦指數": stars
                    })
            st.session_state.final_df = pd.DataFrame(master_report)
            st.session_state.calculated, st.session_state.last_posture = True, market_posture
            st.success(f"📊 倉位與防守價位精準計算完成！")
            
    if st.session_state.calculated:
        def apply_block_shading(df):
            unique_tickers = df["股票代號"].unique()
            styles = pd.DataFrame('', index=df.index, columns=df.columns)
            for i, ticker in enumerate(unique_tickers):
                bg_color = 'background-color: rgba(128, 128, 128, 0.16)' if i % 2 == 0 else 'background-color: rgba(0, 0, 0, 0)'
                mask = df["股票代號"] == ticker
                styles.loc[mask, :] = bg_color
            return styles

        styled_df = st.session_state.final_df.style.apply(apply_block_shading, axis=None)
        st.dataframe(styled_df, use_container_width=True, hide_index=True)

with tab_debug:
    st.header("🛠️ 歷史交易明細與 K 線點位檢查器")
    if st.session_state.calculated:
        col_tk, col_st = st.columns(2)
        with col_tk: debug_ticker = st.selectbox("🎯 選擇想檢查的股票代號", ticker_list)
        with col_st: debug_strat = st.selectbox("🔮 選擇想拆解的決策策略", ["A: 激進動能型", "B: 穩健波段型", "C: 槓桿防守型", "D: 均值回歸抄底型", "E: 籌碼主力跟隨型"])
        db_key = (debug_ticker, debug_strat)
        if db_key in st.session_state.detail_db:
            data_pack = st.session_state.detail_db[db_key]
            logs_df, buys, sells, v_df = data_pack["logs"], data_pack["buys"], data_pack["sells"], data_pack["v_df"]
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=v_df.index, y=v_df['Close'], mode='lines', name='收盤價', line=dict(color='lightgrey', width=1.5)))
            if len(buys) > 0:
                fig.add_trace(go.Scatter(x=[b[0] for b in buys], y=[b[1] for b in buys], mode='markers', name='🟢 BUY (進場)', marker=dict(symbol='triangle-up', size=12, color='#00FF00'), hovertemplate='<b>進場買點</b><br>日期: %{x|%Y-%m-%d}<br>價格: $%{y:.2f}<extra></extra>'))
            if len(sells) > 0:
                fig.add_trace(go.Scatter(x=[s[0] for s in sells], y=[s[1] for s in sells], mode='markers', name='🔴 SELL (出場)', marker=dict(symbol='triangle-down', size=12, color='#FF0000'), hovertemplate='<b>風控出場點</b><br>日期: %{x|%Y-%m-%d}<br>價格: $%{y:.2f}<extra></extra>'))
            fig.update_layout(title=f"<b>{debug_ticker} - {debug_strat} 回測路徑圖</b>", xaxis_title="交易日期", yaxis_title="價格 ($)", hovermode="x unified", template="plotly_dark")
            st.plotly_chart(fig, use_container_width=True)
            if not logs_df.empty: st.dataframe(logs_df, use_container_width=True, hide_index=True)
