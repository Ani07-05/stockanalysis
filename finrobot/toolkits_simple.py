from finrobot.data_source.yfinance_utils import YFinanceUtils

def get_tools():
    return [YFinanceUtils().get_stock_data]
