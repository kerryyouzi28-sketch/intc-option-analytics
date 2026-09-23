import math
import os
import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import norm


def run_pipeline():
    ticker = "INTC"
    target_days = 30  # 評估 30 天後到期的選擇權

    print(f"開始抓取 {ticker} 最新市場數據...")
    stock = yf.Ticker(ticker)
    hist = stock.history(period="1y")

    if hist.empty:
        print("抓取數據失敗！")
        return

    # 取得最新收盤價與年化波動度
    current_price = float(hist["Close"].iloc[-1])
    daily_returns = np.log(hist["Close"] / hist["Close"].shift(1))
    annual_volatility = float(daily_returns.std() * np.sqrt(252))

    risk_free_rate = 0.045
    T = target_days / 365.0
    base_strike = round(current_price)

    results = []
    # 測試現價前後的履約價
    for offset in range(-10, 25, 2):
        strike = float(base_strike + offset)

        # Black-Scholes d2 勝率計算
        d2 = (
            np.log(current_price / strike)
            + (risk_free_rate - 0.5 * (annual_volatility**2)) * T
        ) / (annual_volatility * np.sqrt(T))

        win_rate = float(norm.cdf(d2) * 100)

        if strike < current_price:
            status = "價內 (ITM)"
        elif strike == base_strike:
            status = "價平 (ATM)"
        else:
            status = "價外 (OTM)"

        results.append(
            {
                "履約價 (Strike)": strike,
                "狀態": status,
                "到期進入價內勝率 (%)": round(win_rate, 2),
            }
        )

    df = pd.DataFrame(results)

    # 印出結果
    print(
        f"\n{ticker} 當前股價: ${current_price:.2f} | 歷史波動度: {annual_volatility*100:.2f}%\n"
    )
    print(df.to_string(index=False))

    # 自動儲存結果為 CSV 報表
    os.makedirs("reports", exist_ok=True)
    df.to_csv("reports/intc_win_rate.csv", index=False, encoding="utf-8-sig")
    print("\n分析報告已成功儲存至 reports/intc_win_rate.csv！")


if __name__ == "__main__":
    run_pipeline()
