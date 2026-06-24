# app.py
import akshare as ak
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from datetime import datetime, timedelta

st.set_page_config(page_title="A股均线粘合突破选股平台", layout="wide")

st.title("A股均线粘合突破选股平台 V1")
st.caption("均线粘合 → 放量突破 → 统计未来60个交易日最大涨幅是否超过50%。")

# =========================
# 参数区
# =========================

st.sidebar.header("策略参数")

years = st.sidebar.slider("历史回测年数", 1, 10, 5)
compression_threshold = st.sidebar.slider("均线粘合度阈值", 0.01, 0.10, 0.05, 0.005)
volume_multiplier = st.sidebar.slider("放量倍数", 1.0, 5.0, 1.8, 0.1)
turnover_threshold = st.sidebar.slider("最低换手率", 0.0, 20.0, 3.0, 0.5)
future_days = st.sidebar.slider("未来观察天数", 20, 120, 60, 5)
target_return = st.sidebar.slider("目标涨幅", 0.10, 1.50, 0.50, 0.05)
max_stocks = st.sidebar.slider("最多扫描股票数量", 10, 500, 100, 10)

run_scan = st.sidebar.button("开始扫描")


# =========================
# 工具函数
# =========================

def normalize_code(code):
    code = str(code).strip()
    code = code.replace("SZ:", "").replace("SH:", "")
    code = code.replace("sz", "").replace("sh", "")
    return code.zfill(6)


@st.cache_data(show_spinner=False)
def get_stock_list():
    try:
        df = ak.stock_info_a_code_name()
        df.columns = ["code", "name"]
        df["code"] = df["code"].astype(str).str.zfill(6)
        df = df[~df["name"].str.contains("ST|退|退市", na=False)]
        df = df.reset_index(drop=True)
        return df, "akshare"

    except Exception:
        fallback_stocks = [
            ["000725", "京东方A"],
            ["000333", "美的集团"],
            ["000651", "格力电器"],
            ["002415", "海康威视"],
            ["002594", "比亚迪"],
            ["300750", "宁德时代"],
            ["300059", "东方财富"],
            ["300124", "汇川技术"],
            ["600519", "贵州茅台"],
            ["600036", "招商银行"],
            ["601318", "中国平安"],
            ["601012", "隆基绿能"],
            ["601899", "紫金矿业"],
            ["603259", "药明康德"],
            ["688981", "中芯国际"],
        ]

        df = pd.DataFrame(fallback_stocks, columns=["code", "name"])
        return df, "fallback"


@st.cache_data(show_spinner=False)
def get_daily_data(code, start_date, end_date):
    try:
        code = normalize_code(code)

        df = ak.stock_zh_a_hist(
            symbol=code,
            period="daily",
            start_date=start_date,
            end_date=end_date,
            adjust="qfq"
        )

        if df is None or df.empty:
            return pd.DataFrame()

        df = df.rename(columns={
            "日期": "date",
            "开盘": "open",
            "收盘": "close",
            "最高": "high",
            "最低": "low",
            "成交量": "volume",
            "成交额": "amount",
            "振幅": "amplitude",
            "涨跌幅": "pct_change",
            "换手率": "turnover"
        })

        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)

        numeric_cols = [
            "open", "close", "high", "low", "volume", "amount",
            "amplitude", "pct_change", "turnover"
        ]

        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        df = df.dropna(subset=["close"])
        return df

    except Exception:
        return pd.DataFrame()


def calculate_indicators(df, future_days, target_return):
    df = df.copy()

    for ma in [5, 10, 20, 30, 60, 120]:
        df[f"MA{ma}"] = df["close"].rolling(ma).mean()

    ma_cols = ["MA5", "MA10", "MA20", "MA30", "MA60"]

    df["ma_max"] = df[ma_cols].max(axis=1)
    df["ma_min"] = df[ma_cols].min(axis=1)

    df["compression"] = (df["ma_max"] - df["ma_min"]) / df["close"]

    df["volume_ma20"] = df["volume"].rolling(20).mean()
    df["volume_ratio"] = df["volume"] / df["volume_ma20"]

    df["break_ma60"] = df["close"] > df["MA60"]
    df["break_20d_high"] = df["close"] >= df["high"].rolling(20).max()

    df["future_max_close"] = (
        df["close"]
        .shift(-1)
        .rolling(window=future_days)
        .max()
        .shift(-(future_days - 1))
    )

    df["future_return"] = df["future_max_close"] / df["close"] - 1
    df["success"] = df["future_return"] >= target_return

    return df


def classify_compression(x):
    if pd.isna(x):
        return "无数据"
    if x <= 0.03:
        return "S级超级粘合"
    elif x <= 0.05:
        return "A级强粘合"
    elif x <= 0.08:
        return "B级普通粘合"
    else:
        return "不粘合"


def find_events(df, code, name):
    df = calculate_indicators(df, future_days, target_return)

    condition = (
        (df["compression"] <= compression_threshold) &
        (df["volume_ratio"] >= volume_multiplier) &
        (df["turnover"] >= turnover_threshold) &
        (df["break_ma60"]) &
        (df["break_20d_high"])
    )

    events = df[condition].copy()

    if events.empty:
        return pd.DataFrame()

    events["code"] = code
    events["name"] = name
    events["compression_level"] = events["compression"].apply(classify_compression)

    cols = [
        "date", "code", "name",
        "close", "compression", "compression_level",
        "volume_ratio", "turnover", "amplitude",
        "pct_change", "future_return", "success"
    ]

    return events[cols]


def calculate_score(row):
    score = 0

    if row["compression"] <= 0.03:
        score += 30
    elif row["compression"] <= 0.05:
        score += 24
    elif row["compression"] <= 0.08:
        score += 15

    if row["volume_ratio"] >= 3:
        score += 25
    elif row["volume_ratio"] >= 2:
        score += 20
    elif row["volume_ratio"] >= 1.5:
        score += 12

    if 3 <= row["turnover"] <= 12:
        score += 20
    elif row["turnover"] > 12:
        score += 12
    elif row["turnover"] > 1:
        score += 8

    if 3 <= row["amplitude"] <= 10:
        score += 15
    elif row["amplitude"] > 10:
        score += 8

    if 2 <= row["pct_change"] <= 8:
        score += 10
    elif row["pct_change"] > 8:
        score += 6

    return score


# =========================
# 主程序
# =========================

end_date = datetime.today()
start_date = end_date - timedelta(days=365 * years)

start_str = start_date.strftime("%Y%m%d")
end_str = end_date.strftime("%Y%m%d")

stock_list, source = get_stock_list()

st.subheader("A股股票池")

if source == "akshare":
    st.success(f"股票池加载成功：{len(stock_list)} 只，已剔除 ST、退市股。")
else:
    st.warning("AkShare 股票列表接口连接失败，当前使用备用股票池。日线接口如果也失败，建议后续改成本地CSV上传版。")

sample_stock = st.text_input("单股分析代码，例如 000725", "000725")


# =========================
# 单股分析
# =========================

st.subheader("单股均线粘合分析")

if sample_stock:
    sample_stock = normalize_code(sample_stock)
    df_single = get_daily_data(sample_stock, start_str, end_str)

    if not df_single.empty:
        df_single = calculate_indicators(df_single, future_days, target_return)

        latest = df_single.iloc[-1]

        col1, col2, col3, col4 = st.columns(4)

        col1.metric("当前收盘价", round(latest["close"], 2))
        col2.metric("当前粘合度", f"{latest['compression']:.2%}")
        col3.metric("放量倍数", f"{latest['volume_ratio']:.2f}")
        col4.metric("换手率", f"{latest['turnover']:.2f}%")

        st.write("当前粘合等级：", classify_compression(latest["compression"]))

        fig = go.Figure()

        fig.add_trace(go.Candlestick(
            x=df_single["date"],
            open=df_single["open"],
            high=df_single["high"],
            low=df_single["low"],
            close=df_single["close"],
            name="K线"
        ))

        for ma in [5, 10, 20, 30, 60, 120]:
            fig.add_trace(go.Scatter(
                x=df_single["date"],
                y=df_single[f"MA{ma}"],
                mode="lines",
                name=f"MA{ma}"
            ))

        fig.update_layout(
            height=650,
            xaxis_rangeslider_visible=False,
            title=f"{sample_stock} 均线粘合走势"
        )

        st.plotly_chart(fig, use_container_width=True)

        recent = df_single.tail(120)[[
            "date", "close", "MA5", "MA10", "MA20", "MA30", "MA60",
            "compression", "volume_ratio", "turnover", "amplitude",
            "future_return", "success"
        ]]

        st.dataframe(recent.sort_values("date", ascending=False), use_container_width=True)

    else:
        st.warning("没有获取到该股票日线数据。可能是 AkShare 数据源在 Streamlit Cloud 上被阻挡。")


# =========================
# 全市场扫描
# =========================

st.subheader("全市场历史事件扫描")

if run_scan:
    results = []

    scan_list = stock_list.head(max_stocks).reset_index(drop=True)

    progress = st.progress(0)
    status = st.empty()

    for idx, row in scan_list.iterrows():
        code = row["code"]
        name = row["name"]

        status.text(f"正在扫描：{code} {name}")

        df = get_daily_data(code, start_str, end_str)

        if not df.empty and len(df) > 150:
            event_df = find_events(df, code, name)

            if not event_df.empty:
                results.append(event_df)

        progress.progress((idx + 1) / len(scan_list))

    progress.empty()
    status.empty()

    if results:
        final_df = pd.concat(results, ignore_index=True)

        final_df["score"] = final_df.apply(calculate_score, axis=1)
        final_df = final_df.sort_values(["score", "future_return"], ascending=False)

        st.success(f"扫描完成，共发现 {len(final_df)} 个历史事件。")

        total_events = len(final_df)
        success_events = final_df["success"].sum()
        success_rate = success_events / total_events if total_events > 0 else 0

        col1, col2, col3 = st.columns(3)

        col1.metric("事件总数", total_events)
        col2.metric("60日涨幅超过目标次数", int(success_events))
        col3.metric("成功率", f"{success_rate:.2%}")

        st.subheader("历史事件结果")
        st.dataframe(final_df, use_container_width=True)

        st.subheader("不同粘合等级成功率")

        group_df = final_df.groupby("compression_level").agg(
            事件数量=("success", "count"),
            成功数量=("success", "sum"),
            平均未来最大涨幅=("future_return", "mean"),
            平均评分=("score", "mean")
        ).reset_index()

        group_df["成功率"] = group_df["成功数量"] / group_df["事件数量"]

        st.dataframe(group_df, use_container_width=True)

        csv = final_df.to_csv(index=False).encode("utf-8-sig")

        st.download_button(
            label="下载历史事件CSV",
            data=csv,
            file_name="a_share_ma_compression_events.csv",
            mime="text/csv"
        )

    else:
        st.warning("没有扫描到符合条件的事件。可以放宽粘合度、放量倍数或换手率参数。")
