from finrobot.functional.analyzer import ReportAnalysisUtils
import os

def analyze_stock(ticker, fyear="2025", tmp_dir="/tmp"):
    """
    Run a full stock analysis using ReportAnalysisUtils and return a dict of results.
    """
    os.makedirs(tmp_dir, exist_ok=True)
    results = {}
    # Income Statement
    income_path = os.path.join(tmp_dir, f"{ticker}_income.txt")
    results['income'] = ReportAnalysisUtils.analyze_income_stmt(ticker, fyear, income_path)
    # Balance Sheet
    balance_path = os.path.join(tmp_dir, f"{ticker}_balance.txt")
    results['balance'] = ReportAnalysisUtils.analyze_balance_sheet(ticker, fyear, balance_path)
    # Cash Flow
    cashflow_path = os.path.join(tmp_dir, f"{ticker}_cashflow.txt")
    results['cashflow'] = ReportAnalysisUtils.analyze_cash_flow(ticker, fyear, cashflow_path)
    # Segment Analysis
    segment_path = os.path.join(tmp_dir, f"{ticker}_segment.txt")
    results['segment'] = ReportAnalysisUtils.analyze_segment_stmt(ticker, fyear, segment_path)
    # Add more as needed
    return results
