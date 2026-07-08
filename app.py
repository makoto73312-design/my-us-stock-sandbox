import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# --- 1. 網頁核心外觀配置 ---
st.set_page_config(page_title="環境感知量化沙盒 V04", page_icon="🔮", layout="wide")
st.title("🔮 量化投資沙盒完全體 (環境主控切換版)")
st.markdown("已成功實裝 **華爾街 Regime-Switching 引擎**，可由側邊欄自由切換大膽、平衡、謹慎防禦姿態")

# --- 2. 側邊欄控制台 (內建終極主控開關) ---
st.sidebar.header("⚙️ 全自動大掃描設定")
default_tickers = "NVDA, AAPL, TSLA, MSFT, AMD"
tickers_input = st.sidebar.text_input("輸入要分析的美股代號清單 (用逗號隔開)", default_tickers)
ticker_list = [t.strip().upper() for t in tickers_input.split(',')]

backtest_days = st.sidebar.slider("歷史回測天數設定", min_value=100, max_value=500, value=300, step=50)

# 🌟 核心新功能：大膽/謹慎環境切換閥
market_posture = st.sidebar.selectbox(
    "⚖️ 當前市場防禦姿態 (環境切換開關)",
    ["🛡️ 標準平衡型", "🚀 大膽進攻型", "🥶 極度謹慎型"],
    index=0,
    help="調整此檔位會自動重寫 5 大策略的內部風控與進場門檻。大膽型放寬濾網捕捉大牛市；謹慎型大幅收緊防禦熊市。"
)

# --- 3. 輔助功能：動能雷達進度條 ---
def draw_progress_bar(score, active_char):
    filled_count = int(round(score / 10))
    filled_count = max(0, min(10, filled_count))
    empty_count = 10 - filled_count
    if filled_count == 0 and score > 0: 
        filled_count = 1
        empty_count = 9
    return f"[{active_char * filled_count}{'░' * empty_count}]"

# --- 4. 技術指標核心計算大腦 (預先算出多檔位分級參數) ---
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
    
    # 🌟 為策略 E 的大膽/標準/謹慎動態籌碼計算三個分位數 (前20%、前10%、前5%)
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

# --- 5. 環境感知 5 大策略回測引擎 ---
def run_backtest_engine(df, strategy_name, days, posture):
    valid_df = df.dropna(subset=['200MA', 'ROC14', 'MACD_Hist', 'RSI_14', 'Vol_MA20', '主力籌碼_Q80', '主力籌碼_Q90', '主力籌碼_Q95']).tail(days).copy()
    if len(valid_df) < 5:
        return "⚠️ 數據不足", 0, 0, 0, 0, "❌ 不推薦", "🛑 觀望/無訊號", 0.05, [], [], [], valid_df
    
    # 🌟 核心環境改寫邏輯：根據下拉選單動態重寫風控閥值
    if "🚀 大膽進攻型" in posture:
        rsi_max = 75       # 策略 A：放寬水溫上限，容忍追高
        vol_mult = 1.05    # 策略 B/C：爆量濾網放寬，只要微幅放量就衝
        dip_pct = -0.10     # 策略 D：抄底維持跌破 10%
        rsi_min = 35       # 策略 D：恐慌氣囊放寬，不用極度絕望也可以抄
        chip_col = '主力籌碼_Q80' # 策略 E：籌碼只需達前 20% 大戶流入
    elif "🥶 極度謹慎型" in posture:
        rsi_max = 65       # 策略 A：水溫上限極度收緊，嚴防套牢
        vol_mult = 1.50    # 策略 B/C：必須爆量達 1.5 倍的超級大主力才跟
        dip_pct = -0.15     # 策略 D：抄底門檻拉高，必須血流成河跌破 15%
        rsi_min = 25       # 策略 D：RSI 必須低於 25 的極度恐慌冰點才伸手
        chip_col = '主力籌碼_Q95' # 策略 E：必須是前 5% 的瘋狂爆買大單
    else: # 🛡️ 標準平衡型
        rsi_max = 70
        vol_mult = 1.20
        dip_pct = -0.10
        rsi_min = 30
        chip_col = '主力籌碼_Q90'

    # 固定策略基礎設定
    if "A:" in strategy_name:
        s_ma, d_ma, stop_loss_pct = valid_df['MA5'], valid_df['MA14'], 0.05
    elif "B:" in strategy_name:
        s_ma, d_ma, stop_loss_pct = valid_df['MA14'], valid_df['MA21'], 0.075
    elif "C:" in strategy_name:
        s_ma, d_ma, stop_loss_pct = valid_df['MA10'], valid_df['MA30'], 0.10
    elif "D:" in strategy_name:
        s_ma, d_ma, stop_loss_pct = valid_df['MA20'], valid_df['200MA'], 0.05
    else: 
        s_ma, d_ma, stop_loss_pct = valid_df['MA5'], valid_df['MA20'], 0.06

    max_ma = valid_df[['MA5', 'MA14', '50MA']].max(axis=1)
    min_ma = valid_df[['MA5', 'MA14', '50MA']].min(axis=1)
    is_entangled_series = ((max_ma - min_ma) / valid_df['50MA'].replace(0, 0.001)) < 0.025

    closes = valid_df['Close'].values
    highs = valid_df['High'].values
    lows = valid_df['Low'].values
    s_mas = s_ma.values
    d_mas = d_ma.values
    m200s = valid_df['200MA'].values
    r14s = valid_df['ROC14'].values
    rsis = valid_df['RSI_14'].values
    vols = valid_df['Volume'].values
    vol_m20s = valid_df['Vol_MA20'].values
    m_hists = valid_df['MACD_Hist'].values
    m_shrinks = valid_df['MACD_Shrink'].values
    m_flows = valid_df['主力籌碼'].values
    chip_threshs = valid_df[chip_col].values
    is_entangled_arr = is_entangled_series.values

    has_position = False
    entry_price = 0
    highest_price_since_entry = 0
    total_trades, win_trades = 0, 0
    total_return, total_gross_profit, total_gross_loss = 0.0, 0.0, 0.0

    trade_logs = []
    plot_buys = []
    plot_sells = []

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
                if (m_shrink_p >= 1 or (m_hist_p > m_hist_y and m_hist_p > 0)) and r14_p > 0 and rsi_p < rsi_max: 
                    is_buy = True
            elif "B:" in strategy_name or "C:" in strategy_name:
                if (not is_entangled) and close_p > sma_p and sma_p > dma_p and vol_p > vol_m20_p * vol_mult: 
                    is_buy = True
            elif "D:" in strategy_name:
                if m200_p > 0 and (close_p - m200_p)/m200_p <= dip_pct and m_shrink_p >= 1 and rsi_p < rsi_min: 
                    is_buy = True
            elif "E:" in strategy_name:
                if m_flow_p > chip_thresh_p and m_flow_p > 0: 
                    is_buy = True
            
            if is_buy:
                has_position = True
                entry_price = close_p
                highest_price_since_entry = close_p
                total_trades += 1
                trade_logs.append({"交易日期": date_str, "動作狀態": "🟢 買入進場 (BUY)", "執行價格": f"${close_p:.2f}", "單筆報酬": "-"})
                plot_buys.append((valid_df.index[i], close_p))
        else:
            highest_price_since_entry = max(highest_price_since_entry, high_p)
            is_exit = False
            exit_price = close_p

            if "D:" not in strategy_name:
                if low_p <= highest_price_since_entry * (1 - stop_loss_pct):
                    is_exit = True
                    exit_price = highest_price_since_entry * (1 - stop_loss_pct)
                elif ("B:" in strategy_name or "C:" in strategy_name) and is_entangled:
                    is_exit = True
                    exit_price = close_p
            else:
                if high_p >= m200_p:
                    is_exit = True
                    exit_price = m200_p
                elif low_p <= entry_price * 0.95:
                    is_exit = True
                    exit_price = entry_price * 0.95

            if is_exit:
                trade_return = (exit_price - entry_price) / entry_price
                total_return += trade_return
                if trade_return > 0:
                    win_trades += 1
                    total_gross_profit += trade_return
                else:
                    total_gross_loss += abs(trade_return)
                has_position = False
                trade_logs.append({"交易日期": date_str, "動作狀態": "🔴 賣出出場 (SELL)", "執行價格": f"${exit_price:.2f}", "單筆報酬": f"{trade_return*100:+.2f}%"})
                plot_sells.append((valid_df.index[i], exit_price))

    final_win_rate = win_trades / total_trades if total_trades > 0 else 0.0
    profit_factor = total_gross_profit / total_gross_loss if total_gross_loss > 0 else (99.9 if total_gross_profit > 0 else 0.0)
    pf_str = "無限" if profit_factor == 99.9 else f"{profit_factor:.2f}"

    stars = "❌ 不推薦"
    if total_return > 0 and total_trades > 0:
        if total_return >= 0.25 and final_win_rate >= 0.55: stars = "⭐⭐⭐⭐⭐ (首選)"
        elif total_return >= 0.15 or final_win_rate >= 0.50: stars = "⭐⭐⭐⭐"
        else: stars = "⭐⭐"

    # 當日即時雷達
    latest = valid_df.iloc[-1]
    prev = valid_df.iloc[-2]
    close_T = float(latest['Close'])
    macdHist_T, macdHist_Y = float(latest['MACD_Hist']), float(prev['MACD_Hist'])
    shortMA_T, shortMA_Y = float(latest[s_ma.name]), float(prev[s_ma.name])
    ma200_T = float(latest['200MA'])
    masterFlow_T, masterFlow_Y = float(latest['主力籌碼']), float(prev['主力籌碼'])
    is_entangled_T = is_entangled_series.iloc[-1]

    radar_text = "💤 狀態不明"
    if "A:" in strategy_name:
        raw_slope = macdHist_T - macdHist_Y
        intensity_pct = (raw_slope / close_T) * 100
        power_score = min(100, int(round(abs(intensity_pct) * 333)))
        if macdHist_T > macdHist_Y and macdHist_T > 0: radar_text = f"🏎️ 狂暴加速 {draw_progress_bar(power_score, '🔥')} 油門 {power_score}%"
        elif macdHist_T < macdHist_Y and macdHist_T > 0: radar_text = f"🔄 多頭減速 {draw_progress_bar(power_score, '💥')} 失速 {power_score}%"
        elif macdHist_T > macdHist_Y and macdHist_T < 0: radar_text = f"🛡️ 下墜煞車 {draw_progress_bar(power_score, '🛡️')} 煞車 {power_score}%"
        else: radar_text = f"📉 跌勢加速 {draw_progress_bar(power_score, '💥')} 下墜 {power_score}%"
    elif "B:" in strategy_name or "C:" in strategy_name:
        ma_slope_pct = ((shortMA_T - shortMA_Y) / shortMA_Y) * 100
        power_score = min(100, int(round(abs(ma_slope_pct) * 200)))
        if is_entangled_T: radar_text = "💤 毫無波瀾 [░░░░░░░░░░] 盤整 0%"
        elif ma_slope_pct > 0 and close_T > shortMA_T: radar_text = f"🏎️ 均線昂揚 {draw_progress_bar(power_score, '🔥')} 油門 {power_score}%"
        elif ma_slope_pct <= 0 and close_T > shortMA_T: radar_text = f"🔄 弧度走平 {draw_progress_bar(power_score, '🔸')} 減速 {power_score}%"
        else: radar_text = f"📉 均線下彎 {draw_progress_bar(power_score, '💥')} 下墜 {power_score}%"
    elif "D:" in strategy_name:
        raw_slope = macdHist_T - macdHist_Y
        brake_pct = (raw_slope / close_T) * 100
        distance_pct = ((close_T - ma200_T) / ma200_T) * 100
        if macdHist_T > macdHist_Y: radar_text = f"🛑 綠柱縮腳 {draw_progress_bar(min(100, int(round(abs(brake_pct)*333))), '🛡️')} 煞車 {min(100, int(round(abs(brake_pct)*333)))}%"
        elif distance_pct <= -10: radar_text = f"⏳ 極度超跌 {draw_progress_bar(min(100, int(round(abs(distance_pct)*5))), '📉')} 恐慌 {min(100, int(round(abs(distance_pct)*5)))}%"
        else: radar_text = "❌ 未達抄底區 [░░░░░░░░░░] 穩定 0%"
    elif "E:" in strategy_name:
        chip_change = masterFlow_T - masterFlow_Y
        base = abs(masterFlow_Y) if masterFlow_Y != 0 else 1
        chip_grow_pct = (chip_change / base) * 100
        power_score = min(100, int(round(abs(chip_grow_pct))))
        if masterFlow_T > 0 and chip_change > 0: radar_text = f"💥 主力搶貨 {draw_progress_bar(power_score, '🔥')} 搶購 {power_score}%"
        elif masterFlow_T > 0 and chip_change <= 0: radar_text = f"⚠️ 主力續抱 {draw_progress_bar(power_score, '🔸')} 持倉 {power_score}%"
        else: radar_text = f"❌ 主力棄守 {draw_progress_bar(power_score, '💥')} 撤離 {power_score}%"

    # 即時決策（同步受到主控開關控制）
    latest_buy_signal = "🛑 觀望/無訊號"
    if "A:" in strategy_name and (latest['MACD_Shrink'] >= 1 or (macdHist_T > macdHist_Y and macdHist_T > 0)) and latest['ROC14'] > 0 and latest['RSI_14'] < rsi_max: 
        latest_buy_signal = "🚀 大膽建倉 (BUY)"
    elif ("B:" in strategy_name or "C:" in strategy_name) and (not is_entangled_T) and close_T > shortMA_T and shortMA_T > shortMA_Y and latest['Volume'] > latest['Vol_MA20'] * vol_mult: 
        latest_buy_signal = "🚀 大膽建倉 (BUY)"
    elif "D:" in strategy_name and ma200_T > 0 and (close_T - ma200_T)/ma200_T <= dip_pct and latest['MACD_Shrink'] >= 1 and latest['RSI_14'] < rsi_min: 
        latest_buy_signal = "🚀 大膽建倉 (BUY)"
    elif "E:" in strategy_name and masterFlow_T > latest[chip_col] and masterFlow_T > 0: 
        latest_buy_signal = "🚀 大膽建倉 (BUY)"

    return radar_text, total_return, final_win_rate, total_trades, pf_str, stars, latest_buy_signal, stop_loss_pct, trade_logs, plot_buys, plot_sells, valid_df

# --- 6. Session State 記憶庫 ---
if "calculated" not in st.session_state:
    st.session_state.calculated = False
    st.session_state.final_df = None
    st.session_state.detail_db = {}
    st.session_state.last_posture = ""

# 如果使用者在側邊欄切換了防禦姿態，強制後台清除記憶重新計算
if st.session_state.calculated and st.session_state.last_posture != market_posture:
    st.session_state.calculated = False

# --- 7. 網頁分頁系統 ---
tab_summary, tab_debug = st.tabs(["📊 綜合決策大分流矩陣", "🔍 深度數據與視覺化對照面板 (Debug 控制台)"])

# 【分頁一：綜合大盤掃描】
with tab_summary:
    st.info(f"💡 當前系統姿態：**{market_posture}**。點擊下方按鈕即可依此姿態刷新全體回測。")
    if st.button("🚀 啟動全自動大腦跨標的回測引擎", use_container_width=True):
        with st.spinner(f"正在以 【{market_posture}】 模式同步執行全球機構級大比對..."):
            master_report = []
            strategies = ["A: 激進動能型", "B: 穩健波段型", "C: 槓桿防守型", "D: 均值回歸抄底型", "E: 籌碼主力跟隨型"]
            
            for ticker in ticker_list:
                df_stock = yf.download(ticker, period="2y", progress=False)
                if df_stock.empty: continue
                df_stock.columns = [col[0] if isinstance(col, tuple) else col for col in df_stock.columns]
                df_stock = calculate_indicators(df_stock)
                
                current_close = float(df_stock['Close'].iloc[-1])
                
                for strat in strategies:
                    radar, ret, win, trades, pf, stars, buy_signal, sl_pct, t_logs, p_buys, p_sells, v_df = run_backtest_engine(df_stock, strat, backtest_days, market_posture)
                    suggested_sl = current_close * (1 - sl_pct)
                    suggested_tp = current_close * (1 + sl_pct * 2)
                    
                    st.session_state.detail_db[(ticker, strat)] = {
                        "logs": pd.DataFrame(t_logs), "buys": p_buys, "sells": p_sells, "v_df": v_df
                    }
                    
                    master_report.append({
                        "股票代號": ticker, "當前市價": f"${current_close:.2f}", "策略手法": strat,
                        "🔮 當前動能雷達 (💥加減速)": radar, "今日決策": buy_signal,
                        "建議進場價": f"${current_close:.2f}" if "BUY" in buy_signal else "⏳ 觀望中",
                        "嚴格停損價": f"${suggested_sl:.2f}" if "BUY" in buy_signal else "⏳ 觀望中",
                        "預計停利價": f"${suggested_tp:.2f}" if "BUY" in buy_signal else "⏳ 觀望中",
                        "總報酬率": f"{ret * 100:+.2f}%", "歷史勝率": f"{win * 100:.1f}%",
                        "交易次數": trades, "獲利因子": pf, "推薦指數": stars
                    })
                    
            st.session_state.final_df = pd.DataFrame(master_report)
            st.session_state.calculated = True
            st.session_state.last_posture = market_posture
            st.success(f"📊 交叉比對完成！已成功將全標的鎖定在 【{market_posture}】 風控軌道。")
            
    if st.session_state.calculated:
        st.dataframe(st.session_state.final_df, use_container_width=True, hide_index=True)

# 【分頁二：深度核心 Debug 面板】
with tab_debug:
    st.header("🛠️ 歷史交易明細與 K 線點位檢查器")
    if not st.session_state.calculated:
        st.info("💡 請先回到第一個分頁按下「啟動全自動大腦」，此處才會解鎖數據。")
    else:
        col_tk, col_st = st.columns(2)
        with col_tk:
            debug_ticker = st.selectbox("🎯 選擇想檢查的股票代號", ticker_list)
        with col_st:
            debug_strat = st.selectbox("🔮 選擇想拆解的決策策略", ["A: 激進動能型", "B: 穩健波段型", "C: 槓桿防守型", "D: 均值回歸抄底型", "E: 籌碼主力跟隨型"])
            
        db_key = (debug_ticker, debug_strat)
        
        if db_key in st.session_state.detail_db:
            data_pack = st.session_state.detail_db[db_key]
            logs_df = data_pack["logs"]
            buys = data_pack["buys"]
            sells = data_pack["sells"]
            v_df = data_pack["v_df"]
            
            st.subheader(f"📊 歷史回測訊號點位互動圖表 (當前模式: {st.session_state.last_posture})")
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=v_df.index, y=v_df['Close'], mode='lines', name='收盤價',
                line=dict(color='lightgrey', width=1.5)
            ))
            
            if len(buys) > 0:
                fig.add_trace(go.Scatter(
                    x=[b[0] for b in buys], y=[b[1] for b in buys],
                    mode='markers', name='🟢 BUY (進場)',
                    marker=dict(symbol='triangle-up', size=12, color='#00FF00'),
                    hovertemplate='<b>進場買點</b><br>日期: %{x|%Y-%m-%d}<br>價格: $%{y:.2f}<br>當日RSI: %{customdata[0]:.1f}<br>當日成交量: %{customdata[1]:.2s}<br>主力籌碼: %{customdata[2]:.2f}M<extra></extra>',
                    customdata=[[float(v_df.loc[b[0], 'RSI_14']), float(v_df.loc[b[0], 'Volume']), float(v_df.loc[b[0], '主力籌碼'])] for b in buys]
                ))
                
            if len(sells) > 0:
                fig.add_trace(go.Scatter(
                    x=[s[0] for s in sells], y=[s[1] for s in sells],
                    mode='markers', name='🔴 SELL (出場)',
                    marker=dict(symbol='triangle-down', size=12, color='#FF0000'),
                    hovertemplate='<b>風控出場點</b><br>日期: %{x|%Y-%m-%d}<br>價格: $%{y:.2f}<extra></extra>'
                ))
                
            fig.update_layout(
                title=f"<b>{debug_ticker} - {debug_strat} 回測路徑圖 (姿態：{st.session_state.last_posture})</b>",
                xaxis_title="交易日期", yaxis_title="價格 ($)", hovermode="x unified",
                template="plotly_dark", legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
            )
            st.plotly_chart(fig, use_container_width=True)
            
            st.subheader("📋 歷史交易明細對照日誌")
            if logs_df.empty:
                st.warning("⏳ 在當前姿態嚴密保護下，歷史上無任何失誤訊號觸發。")
            else:
                st.dataframe(logs_df, use_container_width=True, hide_index=True)
        else:
            st.warning("⚠️ 查無此組合資料，請重新執行掃描。")