import numpy as np
import pandas as pd


def get_daily_volume(end_date, symbols, return_failed = False, period_months = 12, start_date = None):
    from yahooquery import Ticker
    from datetime import datetime
    from dateutil.relativedelta import relativedelta

    end_date = datetime.strptime(end_date, "%Y-%m-%d")
    if start_date == None:
        start_date = end_date - relativedelta(months = period_months)

    start_date = datetime.strptime(start_date, "%Y-%m-%d")

    tickers = Ticker(symbols, asynchronous=True)
    _volume = tickers.history(
        interval = "1d", 
        start = start_date,
        end = end_date 
    )

    # get daily prices
    error_tickers = []
    if type(_volume) == pd.DataFrame:
        volume = {key:_volume.loc[key, "volume"] for key in _volume.index.levels[0]}
    else:
        volume = {}
        for key, value in _volume.items():
            if type(value) == pd.DataFrame:
                volume.update({key:value["volume"]})
            else:
                error_tickers.append(key)

    volume_df = pd.concat(volume, axis=1, sort = True)

    return volume_df

def get_weekly_prices(end_date, symbols, return_failed = False, period_months = 12, start_date = None):
    from yahooquery import Ticker
    from datetime import datetime
    from dateutil.relativedelta import relativedelta

    end_date = datetime.strptime(end_date, "%Y-%m-%d")
    if start_date == None:
        start_date = end_date - relativedelta(months = period_months)

    start_date = datetime.strptime(start_date, "%Y-%m-%d")

    tickers = Ticker(symbols, asynchronous=True)
    _prices = tickers.history(
        # period = period,
        interval = "1d", 
        start = start_date,
        end = end_date 
    )

    # get daily prices
    error_tickers = []
    if type(_prices) == pd.DataFrame:
        prices = {key:_prices.loc[key, "adjclose"] for key in _prices.index.levels[0]}
    else:
        prices = {}
        for key, value in _prices.items():
            if type(value) == pd.DataFrame:
                prices.update({key:value["adjclose"]})
            else:
                error_tickers.append(key)

    prices_df = pd.concat(prices, axis=1, sort = True)

    # days and weeks
    days = pd.date_range(start = start_date, end = end_date, freq = "B", inclusive = "both")
    weeks = pd.date_range(start = start_date, end = end_date, freq = "W-WED", inclusive = "both")

    prices_df = prices_df.reindex(days).fillna(method = "ffill")

    weekly_prices_df = prices_df.loc[weeks, :]
    weekly_prices_df = weekly_prices_df.dropna(axis=1)
    
    if return_failed:
        return error_tickers, weekly_prices_df
    else:
        return weekly_prices_df


def get_weekly_returns(end_date, symbols, return_failed = False, period_months = 12, start_date = None):
    from datetime import datetime

    error_tickers, weekly_prices = get_weekly_prices(
        end_date = end_date, 
        symbols = symbols, 
        return_failed = True, 
        period_months = period_months, 
        start_date = start_date
    )
    
    # get returns
    returns = weekly_prices.pct_change()[1:].dropna(axis = 1)

    if return_failed:
        return error_tickers, returns
    else:
        return returns

def get_annualized_returns(returns, freq = "weekly"):
    from scipy.stats.mstats import gmean
    
    # calculate mean
    mean = gmean(1+returns) - 1
    
    # calculate the annulized return
    if freq == "daily":
        annualmean = (1 + mean)**252 - 1
    elif freq == "weekly":
        annualmean = (1 + mean)**52 - 1
    else:
        annualmean = None
        
    return annualmean

def get_annualized_std(returns, freq = "Weekly"):
    from scipy.stats.mstats import gmean
    
    # calculate weekly mean and cov
    cov = returns.cov()

    # calculate the annulized return and cov
    if freq == "daily":
        annualcov = cov*252
    elif freq == "weekly":
        annualcov = cov*52
    else:
        annualcov = None

    annualvar = np.diagonal(annualcov)
    annualstd = np.sqrt(annualvar)
    
    return annualstd


def get_pdi(returns):
    from sklearn import preprocessing
    from sklearn.decomposition import PCA
    
    st_returns = returns.apply(preprocessing.scale)
    
    pca = PCA().fit(st_returns)
    W = pca.explained_variance_ratio_

    value = 0
    for k in range(1, len(W) + 1):
        value = value + k * W[k - 1]
    pdi = 2 * value - 1
    
    return pdi

def get_wpdi(returns, weights):
    from sklearn import preprocessing
    from sklearn.decomposition import PCA
    
    w_returns = returns * weights
    
    st_returns = returns.apply(preprocessing.scale)
    stw_returns = st_returns * weights
    wst_returns = w_returns.apply(preprocessing.scale)
    
    # corr = ret.corr(method="spearman")
    # cov = ret.cov()
    
    ret = stw_returns
    pca = PCA().fit(ret)
    W = pca.explained_variance_ratio_

    value = 0
    for k in range(1, len(W) + 1):
        value = value + k * W[k - 1]
    pdi = 2 * value - 1
    
    return pdi


def sharpe_ratio(returns, returns_type):
    from scipy.stats.mstats import gmean
    
    # calculate weekly mean and cov
    mean = gmean(1+returns) - 1
    cov = returns.cov()

    # calculate the annulized return and cov
    if returns_type == "daily":
        annualmean = (1 + mean)**252 - 1
        annualcov = cov*252
    elif returns_type == "weekly":
        annualmean = (1 + mean)**52 - 1
        annualcov = cov*52
    else:
        annualmean = None
        annualcov = None


    annualvar = np.diagonal(annualcov)
    annualstd = np.sqrt(annualvar)

    sharp = annualmean/annualstd
    
    return sharp

def VaR(returns, alpha=0.95, lookback_weeks = None):

    if lookback_weeks == None:
        ret_window = returns
    else:
        # get the return window
        ret_window = returns.iloc[-lookback_weeks:]
    
    # Compute the correct percentile loss 
    value_at_risk = np.percentile(ret_window, 100 * (1-alpha), axis = 0)
    return value_at_risk

def CVaR(returns, alpha=0.95, lookback_weeks = None):
    from Code.functions import VaR
    
    # get return window 
    if lookback_weeks == None:
        ret_window = returns
    else:
        # get the return window
        ret_window = returns.iloc[-lookback_weeks:]
        
    # Call out to our existing function
    value_at_risk = VaR(returns, alpha, lookback_weeks=lookback_weeks)
    
    cvar = np.nanmean(ret_window[ret_window < value_at_risk], axis = 0)
    
    return cvar


def starr_ratio(returns, returns_type, alpha=0.95, lookback_weeks = None):
    from Code.functions import CVaR
    from Code.functions import get_annualized_returns
    
    annualmean = get_annualized_returns(returns, returns_type)
    
    cvar = CVaR(returns, alpha, lookback_weeks = lookback_weeks)
    starr = annualmean/(-cvar)
    return starr


def get_stats(returns):
    from Code.functions import sharpe_ratio, starr_ratio
    from Code.functions import get_annualized_returns, get_annualized_std
    from Code.functions import CVaR
    
    
    stats = {
        "Weekly mean Returns (%)": returns.mean()[0] * 100, 
        "Std of Weekly Returns (%)": returns.std()[0] * 100, 
        "Annualized Returns (%)": get_annualized_returns(returns, "weekly")[0] * 100,
        "Annualized Std (%)": get_annualized_std(returns, "weekly")[0] * 100,
        "Sharpe Ratio": sharpe_ratio(returns, "weekly")[0],
        "Starr Ratio": starr_ratio(returns, "weekly")[0],
        "CVaR (%)": CVaR(returns)[0] * 100

    }
    
    return stats

def get_subset_stat(returns):
    from Code.functions import get_annualized_returns, get_annualized_std, sharpe_ratio, starr_ratio
    corr = returns.corr("spearman").values

    subset_tab = {
        "n": len(returns.columns),
        "Avg. annualized returns (%)": get_annualized_returns(returns, "weekly").mean() * 100,
        "Avg. annualized std (%)":  get_annualized_std(returns, "weekly").mean() * 100,
        "Avg. Sharpe Ratio": sharpe_ratio(returns, "weekly").mean(),
        "Avg. Starr Ratio": starr_ratio(returns, "weekly").mean(),
        "PDI": get_pdi(returns),
        "Avg. pair-wise correlation": corr[np.triu_indices(n = corr.shape[0] ,k = 1)].mean(),
    }
    
    return subset_tab


def compare_subset_stat(returns, subsets):
    
    from Code.functions import get_subset_stat
    from Code.functions import get_annualized_returns, get_annualized_std, sharpe_ratio, starr_ratio
    
    stats = {}
    for size, subset in subsets.items():
        size_returns = returns[subset]
        stats[size] = get_subset_stat(size_returns)
        
    stats = pd.DataFrame(stats)
    stats.columns = [c + " Subset" for c in stats.columns]
    return stats