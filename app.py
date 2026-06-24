import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score

st.set_page_config(page_title="A股均线粘合选股平台 V4", layout="wide")

st.title("A股均线粘合突破选股平台 V4")
st.caption("CSV上传版：因子分析、分桶成功率、Logistic回归、Random Forest因子重要性、Top20候选股。")

st.sidebar.header("策略参数")

compression_threshold = st.sidebar.slider("均线粘合度阈值", 0.01, 0.15, 0.08, 0.005)
volume_reference = st.sidebar.slider("放量参考线", 0.5, 5.0, 1.2, 0.1)
turnover_reference = st.sidebar.slider("换手率参考线", 0.0, 20.0, 0.0, 0.5)
future_days = st.sidebar.slider("未来观察天数", 20, 120, 60, 5)
target_return = st.sidebar.slider("目标涨幅", 0.10, 1.50, 0.50, 0.05)

uploaded_file = st.file_uploader("上传A股历史日线CSV文件", type=["csv"])


def load_csv(file):
    try:
        df = pd.read_csv(file, encoding="utf-8-sig")
    except Exception:
        df = pd.read_csv(file, encoding="gbk")

    rename_map = {
        "日期": "date",
        "股票代码": "code",
        "代码": "code",
        "股票名称": "name",
        "名称": "name",
        "开盘": "open",
        "最高": "high",
        "最低": "low",
        "收盘": "close",
        "成交量": "volume",
        "成交额": "amount",
        "换手率": "turnover",
        "振幅": "amplitude",
        "涨跌幅": "pct_change",
        "Date": "date",
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Volume": "volume",
    }

    df = df.rename(columns=rename_map)

    required_cols = ["date", "code", "open", "high", "low", "close", "volume"]
    missing = [c for c in required_cols if c not in df.columns]

    if missing:
        st.error(f"CSV缺少必要列：{missing}")
        return pd.DataFrame()

    if "name" not in df.columns:
        df["name"] = df["code"].astype(str)

    if "turnover" not in df.columns:
        df["turnover"] = 0

    if "amplitude" not in df.columns:
        df["amplitude"] = (df["high"] - df["low"]) / df["close"] * 100

    if "pct_change" not in df.columns:
        df["pct_change"] = df.groupby("code")["close"].pct_change() * 100

    df["date"] = pd.to_datetime(df["date"])
    df["code"] = df["code"].astype(str).str.replace(".0", "", regex=False).str.zfill(6)

    num_cols = [
        "open", "high", "low", "close", "volume",
        "turnover", "amplitude", "pct_change"
    ]

    for col in num_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["date", "code", "close"])
    df = df.sort_values(["code", "date"]).reset_index(drop=True)

    return df


def calculate_indicators(df):
    df = df.copy()

    for ma in [5, 10, 20, 30, 60, 120]:
        df[f"MA{ma}"] = df.groupby("code")["close"].transform(
            lambda x: x.rolling(ma).mean()
        )

    ma_cols = ["MA5", "MA10", "MA20", "MA30", "MA60"]

    df["ma_max"] = df[ma_cols].max(axis=1)
    df["ma_min"] = df[ma_cols].min(axis=1)
    df["compression"] = (df["ma_max"] - df["ma_min"]) / df["close"]

    df["volume_ma20"] = df.groupby("code")["volume"].transform(
        lambda x: x.rolling(20).mean()
    )
    df["volume_ratio"] = df["volume"] / df["volume_ma20"]

    df["return_5d"] = df.groupby("code")["close"].pct_change(5)
    df["return_10d"] = df.groupby("code")["close"].pct_change(10)
    df["return_20d"] = df.groupby("code")["close"].pct_change(20)

    df["volatility_20d"] = df.groupby("code")["pct_change"].transform(
        lambda x: x.rolling(20).std()
    )

    df["price_vs_ma60"] = df["close"] / df["MA60"] - 1
    df["price_vs_ma120"] = df["close"] / df["MA120"] - 1

    df["break_ma60"] = df["close"] > df["MA60"]

    df["rolling_20_high"] = df.groupby("code")["high"].transform(
        lambda x: x.rolling(20).max()
    )
    df["rolling_60_high"] = df.groupby("code")["high"].transform(
        lambda x: x.rolling(60).max()
    )

    df["break_20d_high"] = df["close"] >= df["rolling_20_high"]
    df["break_60d_high"] = df["close"] >= df["rolling_60_high"]

    df["future_max_close"] = df.groupby("code")["close"].transform(
        lambda x: x.shift(-1).rolling(future_days).max().shift(-(future_days - 1))
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
        return "C级弱粘合"


def calculate_score(row):
    score = 0

    if row["compression"] <= 0.03:
        score += 30
    elif row["compression"] <= 0.05:
        score += 24
    elif row["compression"] <= 0.08:
        score += 18
    elif row["compression"] <= 0.12:
        score += 10

    if row["volume_ratio"] >= 3:
        score += 25
    elif row["volume_ratio"] >= 2:
        score += 20
    elif row["volume_ratio"] >= 1.2:
        score += 12

    if row["turnover"] == 0:
        score += 5
    elif 3 <= row["turnover"] <= 12:
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


def bucket_analysis(df, factor, bins):
    temp = df.copy()
    temp[f"{factor}_bucket"] = pd.cut(temp[factor], bins=bins)

    result = temp.groupby(f"{factor}_bucket", observed=False).agg(
        样本数量=("success", "count"),
        成功数量=("success", "sum"),
        平均未来涨幅=("future_return", "mean"),
        中位数未来涨幅=("future_return", "median")
    ).reset_index()

    result["成功率"] = result["成功数量"] / result["样本数量"]

    return result


if uploaded_file is None:
    st.warning("请先上传 CSV 文件。")
    st.stop()

raw_df = load_csv(uploaded_file)

if raw_df.empty:
    st.stop()

st.success(f"CSV加载成功：{len(raw_df)} 行，{raw_df['code'].nunique()} 只股票。")

df = calculate_indicators(raw_df)

condition = (
    (df["compression"] <= compression_threshold) &
    (df["break_ma60"]) &
    (df["break_20d_high"])
)

events = df[condition].copy()

if events.empty:
    st.warning("没有扫描到符合条件的历史事件，可以放宽左侧均线粘合度阈值。")
    st.stop()

events["compression_level"] = events["compression"].apply(classify_compression)
events["score"] = events.apply(calculate_score, axis=1)

events["是否高于放量参考线"] = events["volume_ratio"] >= volume_reference
events["是否高于换手率参考线"] = events["turnover"] >= turnover_reference

feature_cols = [
    "compression",
    "volume_ratio",
    "turnover",
    "amplitude",
    "pct_change",
    "return_5d",
    "return_10d",
    "return_20d",
    "volatility_20d",
    "price_vs_ma60",
    "price_vs_ma120"
]

events_model = events.dropna(subset=feature_cols + ["success", "future_return"]).copy()
events_model["success_int"] = events_model["success"].astype(int)

st.subheader("一、事件研究结果")

final_cols = [
    "date", "code", "name", "close",
    "compression", "compression_level",
    "volume_ratio", "是否高于放量参考线",
    "turnover", "是否高于换手率参考线",
    "amplitude", "pct_change", "return_20d", "volatility_20d",
    "future_return", "success", "score"
]

final_df = events[final_cols].sort_values(
    ["score", "future_return"],
    ascending=False
)

col1, col2, col3 = st.columns(3)
col1.metric("事件总数", len(final_df))
col2.metric("成功事件数", int(final_df["success"].sum()))
col3.metric("成功率", f"{final_df['success'].mean():.2%}")

st.dataframe(final_df, use_container_width=True)

st.subheader("二、单股K线分析")

stock_options = (
    df[["code", "name"]]
    .drop_duplicates()
    .assign(label=lambda x: x["code"] + " - " + x["name"].astype(str))
)

selected_label = st.selectbox("选择股票", stock_options["label"].tolist())
selected_code = selected_label.split(" - ")[0]

single_df = df[df["code"] == selected_code].copy()
latest = single_df.iloc[-1]

c1, c2, c3, c4 = st.columns(4)
c1.metric("最新收盘价", round(latest["close"], 2))
c2.metric("最新粘合度", f"{latest['compression']:.2%}")
c3.metric("最新放量倍数", f"{latest['volume_ratio']:.2f}")
c4.metric("最新20日波动率", f"{latest['volatility_20d']:.2f}")

fig = go.Figure()

fig.add_trace(go.Candlestick(
    x=single_df["date"],
    open=single_df["open"],
    high=single_df["high"],
    low=single_df["low"],
    close=single_df["close"],
    name="K线"
))

for ma in [5, 10, 20, 30, 60, 120]:
    fig.add_trace(go.Scatter(
        x=single_df["date"],
        y=single_df[f"MA{ma}"],
        mode="lines",
        name=f"MA{ma}"
    ))

fig.update_layout(height=650, xaxis_rangeslider_visible=False)
st.plotly_chart(fig, use_container_width=True)

st.subheader("三、因子相关性分析")

if len(events_model) >= 10:
    corr_df = (
        events_model[feature_cols + ["future_return"]]
        .corr(method="spearman")["future_return"]
        .drop("future_return")
        .sort_values(ascending=False)
        .reset_index()
    )

    corr_df.columns = ["因子", "Spearman相关性"]

    st.dataframe(corr_df, use_container_width=True)

    fig_corr = go.Figure()
    fig_corr.add_trace(go.Bar(
        x=corr_df["因子"],
        y=corr_df["Spearman相关性"]
    ))
    fig_corr.update_layout(
        height=450,
        title="因子与未来60日最大涨幅 Spearman相关性"
    )
    st.plotly_chart(fig_corr, use_container_width=True)

else:
    st.warning("样本量太少，暂不建议看相关性。至少需要10个事件。")

st.subheader("四、分桶成功率分析")

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "粘合度",
    "放量倍数",
    "换手率",
    "振幅",
    "20日动量"
])

with tab1:
    result = bucket_analysis(
        events_model,
        "compression",
        bins=[0, 0.03, 0.05, 0.08, 0.12, 1]
    )
    st.dataframe(result, use_container_width=True)

with tab2:
    result = bucket_analysis(
        events_model,
        "volume_ratio",
        bins=[0, 1, 1.2, 1.5, 2, 3, 5, 100]
    )
    st.dataframe(result, use_container_width=True)

with tab3:
    result = bucket_analysis(
        events_model,
        "turnover",
        bins=[-0.01, 0.01, 1, 3, 5, 8, 12, 20, 100]
    )
    st.dataframe(result, use_container_width=True)

with tab4:
    result = bucket_analysis(
        events_model,
        "amplitude",
        bins=[0, 3, 5, 8, 12, 20, 100]
    )
    st.dataframe(result, use_container_width=True)

with tab5:
    result = bucket_analysis(
        events_model,
        "return_20d",
        bins=[-1, -0.2, -0.1, 0, 0.1, 0.2, 1]
    )
    st.dataframe(result, use_container_width=True)

st.subheader("五、参考线分组分析")

col_a, col_b = st.columns(2)

with col_a:
    vol_ref_df = events_model.copy()
    vol_ref_df["放量参考分组"] = np.where(
        vol_ref_df["volume_ratio"] >= volume_reference,
        "高于放量参考线",
        "低于放量参考线"
    )

    vol_summary = vol_ref_df.groupby("放量参考分组").agg(
        样本数量=("success", "count"),
        成功数量=("success", "sum"),
        平均未来涨幅=("future_return", "mean"),
        中位数未来涨幅=("future_return", "median")
    ).reset_index()

    vol_summary["成功率"] = vol_summary["成功数量"] / vol_summary["样本数量"]

    st.write("放量参考线分析")
    st.dataframe(vol_summary, use_container_width=True)

with col_b:
    turnover_ref_df = events_model.copy()
    turnover_ref_df["换手率参考分组"] = np.where(
        turnover_ref_df["turnover"] >= turnover_reference,
        "高于换手率参考线",
        "低于换手率参考线"
    )

    turnover_summary = turnover_ref_df.groupby("换手率参考分组").agg(
        样本数量=("success", "count"),
        成功数量=("success", "sum"),
        平均未来涨幅=("future_return", "mean"),
        中位数未来涨幅=("future_return", "median")
    ).reset_index()

    turnover_summary["成功率"] = turnover_summary["成功数量"] / turnover_summary["样本数量"]

    st.write("换手率参考线分析")
    st.dataframe(turnover_summary, use_container_width=True)

st.subheader("六、建模分析")

if len(events_model) >= 30 and events_model["success_int"].nunique() == 2:
    X = events_model[feature_cols]
    y = events_model["success_int"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.30,
        random_state=42,
        stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    logit = LogisticRegression(max_iter=1000)
    logit.fit(X_train_scaled, y_train)

    logit_pred = logit.predict(X_test_scaled)
    logit_prob = logit.predict_proba(X_test_scaled)[:, 1]

    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=5,
        random_state=42,
        class_weight="balanced"
    )
    rf.fit(X_train, y_train)

    rf_pred = rf.predict(X_test)
    rf_prob = rf.predict_proba(X_test)[:, 1]

    col1, col2 = st.columns(2)

    with col1:
        st.write("Logistic Regression")
        st.metric("Accuracy", f"{accuracy_score(y_test, logit_pred):.2%}")
        st.metric("ROC-AUC", f"{roc_auc_score(y_test, logit_prob):.3f}")

        coef_df = pd.DataFrame({
            "因子": feature_cols,
            "系数": logit.coef_[0]
        }).sort_values("系数", ascending=False)

        st.dataframe(coef_df, use_container_width=True)

    with col2:
        st.write("Random Forest")
        st.metric("Accuracy", f"{accuracy_score(y_test, rf_pred):.2%}")
        st.metric("ROC-AUC", f"{roc_auc_score(y_test, rf_prob):.3f}")

        importance_df = pd.DataFrame({
            "因子": feature_cols,
            "重要性": rf.feature_importances_
        }).sort_values("重要性", ascending=False)

        st.dataframe(importance_df, use_container_width=True)

        fig_imp = go.Figure()
        fig_imp.add_trace(go.Bar(
            x=importance_df["因子"],
            y=importance_df["重要性"]
        ))
        fig_imp.update_layout(
            height=450,
            title="Random Forest 因子重要性"
        )
        st.plotly_chart(fig_imp, use_container_width=True)

else:
    st.warning("建模样本不足。至少需要30个事件，并且成功/失败样本都要有。")

st.subheader("七、Top 20 候选事件")

st.dataframe(final_df.head(20), use_container_width=True)

csv = final_df.to_csv(index=False).encode("utf-8-sig")

st.download_button(
    label="下载V4事件研究结果CSV",
    data=csv,
    file_name="a_share_ma_compression_events_v4.csv",
    mime="text/csv"
)
