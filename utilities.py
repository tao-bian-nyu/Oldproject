import pandas as pd
import numpy as np
from multiprocessing import Pool
# import warnings
from cvxopt import matrix, solvers
from scipy.stats.mstats import gmean
# from datetime import datetime

solvers.options['show_progress'] = False
# warnings.filterwarnings("ignore")

def orthogonal(df1, df2):  # make df1 orthogonal to df2 row-wise; already de-meaned
    df10 = df1.sub(df1.mean(axis=1), axis=0)
    df20 = df2.copy()
    df20[df10.isna()] = np.nan
    df20 = df20.sub(df20.mean(axis=1), axis=0)
    beta = (df10*df20).sum(axis=1) / (df20**2).sum(axis=1)
    return df10 - df20.mul(beta, axis=0)

def gram_schmidt(df_list):
    res = [df_list[0]]
    for i in range(1, len(df_list)):
        df = df_list[i]
        for c in res:
            df = orthogonal(df, c)
        res.append(df)
    return res


def portfolio_model(factor, num_stks=100.0, method='EW', gamma=0.001, longshort='long', industry=None, **kwargs):
    '''
    factor ranking, 1=best
    Generate portfolio stock weights using factor
    Since factor produces continuous ranking for stocks, specify # of stocks to include in the portfolio.
    Generate weights of stocks, can use M-V optimization or equal weights or by size
    Since a single-factor model is used, MVO is performed at stock level, i.e., allocate weights amount the num_stks stocks selected by factor every day.
    '''
    # max_weight = 1.0/num_stks                                                        # max weights of stocks in equal-weighted case
    # rank_threshold = (factor.min(axis=1)+num_stks).values.reshape(-1,1)         # long stocks < rank_threshold_long
    # portfolio = np.where(factor.values<rank_threshold, max_weight, 0)

    num_stks = num_stks*1.0
    if type(num_stks)==float:
        max_weight = 1.0/num_stks
    else:
        num_stks = num_stks.shift(1).loc[factor.index]
        max_weight = 1.0/num_stks.values.reshape(-1, 1)

    rank_threshold = (factor.min(axis=1)+num_stks).values.reshape(-1, 1)
    rank_threshold = np.nan_to_num(rank_threshold, nan=0)
    portfolio = np.where(factor.values < rank_threshold, max_weight, 0)
    portfolio = portfolio/portfolio.sum(axis=1).reshape(-1,1) # in case num_stks > # of stks

    if method =='MVO':
        # use MVO to allocate weights, use num_days days c2c return to estimate
        # max_weight = kwargs['max_weight']
        num_days = 1 #60                                                                   # use past 3 months data to estimate return and covariance matrix
        daily_ret = kwargs['ret_c2c'].fillna(0)
        portfolio_df = pd.DataFrame(portfolio, columns=factor.columns, index=factor.index)
        for idx in portfolio_df.index:
            # print(idx)
            selected = portfolio_df.loc[idx]
            if type(num_stks)==float:
                max_weight = 1.4/num_stks
            else:
                max_weight = float(1.4/num_stks.loc[idx])
            cols = selected[selected>0].index                                           # stocks that are selected
            i = daily_ret.index.get_loc(idx)
            rets = daily_ret.iloc[i-num_days:i][cols]                                             # daily return in the past num_days days
            if industry is not None:                                                    # industry neutral through MVO
                ind = industry.loc[idx, cols].values
                ind = np.nan_to_num(ind, nan=0)
                ind = ind[~np.all(ind == 0, axis=1)]                                # remove industry that is not in portfolio
                # print(ind)
                if ind.shape[0] == 0:
                    ind = None
                    max_ind=1
                else:
                    max_ind=1.1/ind.shape[0]
            else:
                ind = None
                max_ind=1
            # rets_mean, rets_cov = rets.mean().values, _factor_cov(rets.values)              # estimate mean vector and cov matrix
            

            rets_mean = rets.mean().values
            n = len(rets_mean)
            rets_cov = np.eye(n)                                                    # use I in MVO, goal is only to neutralize industry here                                              

            if longshort=='long':
                portfolio_df.loc[idx, cols] = _mvo(rets_mean, rets_cov, max_weight, gamma=gamma, ind=ind, max_ind=max_ind)     # at least num_stks/1.4 stocks are selected
            else:
                portfolio_df.loc[idx, cols] = _mvo(-rets_mean, rets_cov, max_weight, gamma=gamma, ind=ind, max_ind=max_ind)     # at least num_stks/1.4 stocks are selected

        portfolio = portfolio_df.values

    return portfolio, rank_threshold


def _mvo(rets_mean, rets_cov, max_weight, gamma=0.001, ind=None, max_ind=1):  # M-V optimization
    '''
    mean-variance optimization
    min 0.5w^TPw - gamma * q^Tw
    subject to : \sum w_i=1, 0<w_i<max_weight
    since we use historical mean return to estimate future return is not accurate, use small gamma here 
    '''
    n = len(rets_mean)
    ones = np.ones([n, 1])
    I = np.eye(n)
    
    P, q = rets_cov, -rets_mean*gamma
    if ind is not None:
        ones_ind = np.ones([ind.shape[0], 1])
        h = np.concatenate([ones*max_weight, ones*0, ones_ind*max_ind])
        G = np.concatenate([I, -I, ind])
    else:
        h = np.concatenate([ones*max_weight, ones*0])
        G = np.concatenate([I, -I])
    A = ones.T
    b = np.array([1.0])
    w = solvers.qp(matrix(P), matrix(q),
                   matrix(G), matrix(h),
                   matrix(A), matrix(b))['x']
    return np.array(w).flatten()


def _factor_cov(mat):  # use factor model with PCA to estimate covariance matrix, avoid singularity
    N = mat.shape[0]
    m_std = np.nanstd(mat, axis=0)
    mat_norm = (mat-np.nanmean(mat, axis=0))/m_std
    mat_norm[np.isnan(mat_norm)] = 0
    vals_corr, vecs_corr = np.linalg.eig(mat_norm.T @ mat_norm/N)
    vals_corr = np.real(vals_corr)
    idx = vals_corr.argsort()[::-1]         # sort eigenvalues
    vals_corr = vals_corr[idx]
    vecs_corr = vecs_corr[:, idx]
    n = min(mat.shape)
    sigma = vals_corr[:n].sum()
    for i in range(n):
        # select eigenvalues that represent 90% of total volatility, the rest are noise
        if vals_corr[:i+1].sum() >= 0.9*sigma:
            break

    V = vecs_corr[:, :i+1]    # select principle components
    beta = np.linalg.inv(V.T @ V) @ V.T @ mat_norm.T

    corr = np.real((V@beta) @ (V@beta).T / N)
    diag_idx = np.diag_indices(corr.shape[0])
    corr[diag_idx] = 1.0
    cov = np.multiply(corr, (m_std.reshape(-1, 1) @  m_std.reshape(1, -1)))
    return cov


def _rebalance_turnover(position, weights, ranks, target, rank_threshold, max_turnover=1.0):
    '''
    rebalance portfolio based on ranks, current weights, and target weights. 
    sell stock if its ranks>rank_threshold; exit portfolio
    buy stock if its rank<=rank_threshold; enter portfolio
    target = target total position
    weights = target weights
    max_turnover: between 0 and 1, used for turnover control
    '''
    value_sell = (position[(position > 0) & (ranks > rank_threshold)]).sum()
    value_buy = weights[(ranks <= rank_threshold) & (position/target <= weights)].sum()*target
    value_buy -= position[(position > 0) & (ranks <= rank_threshold) & (position/target <= weights)].sum()
    # To ensure total position is stable, if value_sell > value_buy, sell less; if value_sell < value_buy, buy less
    turnover = min([value_buy, value_sell, max_turnover*target])

    ranks_sell = sorted(ranks[(position > 0) & (ranks > rank_threshold)])
    while np.nansum(position) > target-turnover+0.0001:  # to sell, start from lowest rank
        if len(ranks_sell) == 0:  # sold all
            break
        last_rank = ranks_sell.pop()
        v = position[ranks == last_rank].sum()
        if v < np.nansum(position) - target+turnover:  # sell all
            position[ranks == last_rank] = 0
        else:  # sell partial
            position[ranks == last_rank] -= position[ranks == last_rank] * max(0, (np.nansum(position)-target+turnover)/v)

    ranks_buy = sorted(ranks[(position/target < weights) & (ranks <= rank_threshold)])
    while np.nansum(position) < target-0.0001:  # to buy, start from highest rank
        if len(ranks_buy) == 0:  # bought all
            break
        top_rank = ranks_buy.pop(0)
        v = weights[ranks == top_rank].sum()*target - position[ranks == top_rank].sum()
        if v < target-np.nansum(position):  # buy to target weight
            position[ranks == top_rank] = weights[ranks == top_rank]*target
        else:  # buy partial
            position[ranks == top_rank] += (weights[ranks == top_rank]*target - position[ranks == top_rank]) * max(0, (target-np.nansum(position))/v)

    return position

def _rebalance_vanilla(position, weights, ranks, target, rank_threshold):
    '''
    rebalance portfolio based on ranks, no turnover control
    '''
    position[ranks >= rank_threshold] = 0
    position[ranks < rank_threshold] = weights[(ranks < rank_threshold)] / weights[(ranks < rank_threshold)].sum()*target
    return position


def backtesting(factor, target=1, net=0.01, weighting_method='EW', backtesting_method='vanilla', begin_date='2010-01-01', end_date='2018-01-01', max_turnover=1.0, longshort='long', **kwargs):
    '''
    generate dollar-valued positions for each stock daily 
    '''

    # use factor from yesterday to select stocks and trade today
    factor = factor.shift(1)
    factor = factor[(factor.index >= begin_date) & (factor.index < end_date)]       # specify begin and end dates
    c2c = kwargs['ret_c2c'].fillna(0).loc[factor.index].values + 1              # close / preclose

    # generate portfolio weights for each day, either equal-weight or MVO is used. MVO takes some time

    # num_stks = kwargs['num_stks']*1.0
    portfolio, rank_threshold = portfolio_model(factor=factor, method=weighting_method, longshort=longshort, **kwargs)

    # if type(num_stks)==float:
    #     max_weight = 1.0/num_stks
    # else:
    #     num_stks = num_stks.loc[factor.index]
    #     max_weight = 1.0/num_stks.values.reshape(-1, 1)

    # rank_threshold = (factor.min(axis=1)+num_stks).values.reshape(-1, 1)
    # rank_threshold = np.nan_to_num(rank_threshold, nan=0)
    # portfolio = np.where(factor.values < rank_threshold, max_weight, 0)

    # Portfolio initialization
    idx = 0
    pre2c_idx = c2c[idx, :]

    weights = portfolio[idx, :]
    position = weights/weights.sum() * target
    # trading shares are determined at preclose price; calculate position at close
    position = position * pre2c_idx
    position_hist = position

    for idx in range(1, factor.shape[0]):
        ranks = factor.values[idx, :]
        pre2c_idx = c2c[idx, :]

        threshold = rank_threshold[idx, 0]
        weights = portfolio[idx, :]

        position[ranks==19999] = 0 # liquidate untradable stocks
        position[ranks>9990] = 0 # liquidatge stocks not in factor
        # ensure the long/short position is within the range [target-net, target] during trading
        if np.nansum(position) > target:
            adjusted_target = target
        elif np.nansum(position) < target-net:
            adjusted_target = target-net        # allow position to be slightly smaller than target
        else:
            adjusted_target = np.nansum(position)
        
        if backtesting_method == 'vanilla':
            position = _rebalance_vanilla(position, weights, ranks, adjusted_target, threshold) * pre2c_idx
        else:
            position = _rebalance_turnover(position, weights, ranks, adjusted_target, threshold, max_turnover=max_turnover) * pre2c_idx
        
        position_hist = np.vstack((position_hist, position))      # EOD position
        # note that EOD position may not satisfy max_gmv and net conditions,
        # due to the shares to be traded are determined at preclose before open and the stock price changes from preclose to close during the day.

    position_hist = pd.DataFrame(position_hist, columns=factor.columns, index=factor.index)
    return position_hist


def _backtesting_wrapper(config):
    return backtesting(**config)


def batch_backtesting(config_list):  # fetch s&p500 price volume
    with Pool() as p:
        res = p.map(_backtesting_wrapper, config_list)
    return res


def pnl_calculation(positions, trading_cost=(-0.0001, -0.0001), **kwargs):
    '''
    calculate pnl based on daily positions at close
    trading_pnl = (pos - posL1 * close/preclose) * execprice/close * (close/execprice-1 - fee)
    holding_pnl = posL1 * (close/preclose-1)
    '''

    r_buy, r_sell = 0, 0
    r_keep = kwargs['ret_c2c'].loc[positions.index].fillna(0)

    positions_L1 = (positions.shift(1)).fillna(0)
    # shares_close = positions/close                      # shares at the end of day
    # shares_open = positions_L1/preclose               # shares at the start of day

    # long
    pos = np.where(positions.values > 0, positions.values, 0)
    pos_L1 = np.where(positions_L1.values > 0, positions_L1.values, 0)
    pos_L10 = pos_L1*(1+r_keep)

    pnl_cost_long = np.where(pos == pos_L10, r_keep * pos_L1,                                                                       # no rebalance
                             np.where(pos > pos_L10, r_keep * pos_L1 + (r_buy + trading_cost[0]) * (pos - pos_L10)/(r_buy+1),        # add position, buy more
                             np.where(pos < pos_L10, r_keep * pos_L1 + (r_sell - trading_cost[1]) * (pos - pos_L10)/(r_sell+1), 0)))  # reduce position, sell more

    pnl_long = np.where(pos == pos_L10, r_keep * pos_L1,
                        np.where(pos > pos_L10, r_keep * pos_L1 + r_buy * (pos - pos_L10)/(r_buy+1),
                                 np.where(pos < pos_L10, r_keep * pos_L1 + r_sell * (pos - pos_L10)/(r_sell+1), 0)))
    pnl_short = 0
    pnl_cost_short = 0

    # short
    pos = np.where(positions.values < 0, -positions.values, 0)
    pos_L1 = np.where(positions_L1.values < 0, -positions_L1.values, 0)
    pos_L10 = pos_L1*(1+r_keep)

    pnl_cost_short = np.where(pos == pos_L10, -r_keep * pos_L1,
                              np.where(pos > pos_L10, -r_keep * pos_L1 - (r_sell - trading_cost[1]) * (pos - pos_L10)/(r_buy+1),         # add short position, sell more
                                       np.where(pos < pos_L10, -r_keep * pos_L1 - (r_buy + trading_cost[0]) * (pos - pos_L10)/(r_sell+1), 0)))      # reduce short position, buy more

    pnl_short = np.where(pos == pos_L10, -r_keep * pos_L1,
                         np.where(pos > pos_L10, -r_keep * pos_L1 - r_sell * (pos - pos_L10)/(r_buy+1),
                                  np.where(pos < pos_L10, -r_keep * pos_L1 - r_buy * (pos - pos_L10)/(r_sell+1), 0)))

    pnl = pd.DataFrame(pnl_long+pnl_short, columns=positions.columns, index=positions.index)
    pnl_cost = pd.DataFrame(pnl_cost_long+pnl_cost_short, columns=positions.columns, index=positions.index)

    turnover = (abs(positions - positions_L1*(1+r_keep))).sum(1)/2 # one-side turnover

    return {
        'pnl': pnl,
        'pnl_cost': pnl_cost,
        'turnover': turnover
    }

def summarize(pnl, pnl_cost, turnover, positions, rate=0.002, N=252):
    # evaluate model performance
    pnl = pnl.sum(1)
    nav = pnl.cumsum()
    max_nav = nav.cummax()
    max_nav[max_nav<0] = 0
    ones = pd.Series(1, index=pnl_cost.index).cumsum()
    nav_geo = ((1+pnl).expanding().apply(gmean)-1)*ones

    pnl_cost = pnl_cost.sum(1)
    nav_cost = pnl_cost.cumsum()
    max_nav_cost = nav_cost.cummax()
    max_nav_cost[max_nav<0] = 0
    nav_cost_geo = ((1+pnl_cost).expanding().apply(gmean)-1)*ones

    drawdown_cost = nav_cost - max_nav_cost
    drawdown = nav - max_nav

    dd=pd.DataFrame()
    dd['drawdown_cost']=drawdown_cost
    dd['drawdown']=drawdown
    dd['days'] = 1
    dd['days'] = dd['days'].cumsum()
    drawdown_period = dd.drop(dd[dd['drawdown']==0].index)['days'].diff()
    drawdown_cost_period = dd.drop(dd[dd['drawdown_cost']==0].index)['days'].diff()

    yearly_stats = pd.DataFrame()
    yearly_stats['sharpe'] = sharpe_yearly(pnl, rate=rate, N=N)
    yearly_stats['sharpe_cost'] = sharpe_yearly(pnl_cost, rate=rate, N=N)
    yearly_stats['sortino'] = sortino_yearly(pnl, rate=rate, N=N)
    yearly_stats['sortino_cost'] = sortino_yearly(pnl_cost, rate=rate, N=N)
    yearly_stats['pnl'] = pnl.groupby([pnl.index.year]).mean()*N
    yearly_stats['pnl_cost'] = pnl_cost.groupby([pnl_cost.index.year]).mean()*N
    yearly_stats['volatility'] = pnl.groupby([pnl.index.year]).std()*np.sqrt(N)
    yearly_stats['volatility_cost'] = pnl_cost.groupby([pnl_cost.index.year]).std()*np.sqrt(N)
    pnlp = pnl[pnl>0]
    yearly_stats['pos_days'] = pnlp.groupby([pnlp.index.year]).count() / pnl.groupby([pnl.index.year]).count()
    pnlp = pnl_cost[pnl_cost>0]
    yearly_stats['pos_days_cost'] = pnlp.groupby([pnlp.index.year]).count() / pnl_cost.groupby([pnl_cost.index.year]).count()

    return {
        'summary': {
            'sharpe': sharpe(pnl, rate=rate, N=N),
            'sharpe_cost': sharpe(pnl_cost, rate=rate, N=N),
            'sortino': sortino(pnl, rate=rate, N=N),
            'sortino_cost': sortino(pnl_cost, rate=rate, N=N),
            'pnl': pnl.mean()*N,
            'pnl_cost': pnl_cost.mean()*N,
            'volatility': pnl.std()*np.sqrt(N),
            'volatility_cost': pnl_cost.std()*np.sqrt(N),
            'pos_days': pnl[pnl>0].count()/pnl.count(),
            'pos_days_cost': pnl_cost[pnl_cost>0].count()/pnl_cost.count()
        },
        
        'yearly_stats': yearly_stats,

        'turnover': turnover,
        'pnl': pnl,
        'pnl_cost': pnl_cost,
        'nav_cost': nav_cost,
        'nav': nav,
        'nav_cost_geo': nav_cost_geo,
        'nav_geo': nav_geo,
        'pos_long': positions[positions>0].sum(1),
        'pos_short': -positions[positions<0].sum(1),
        'drawdown_cost': drawdown_cost,
        'drawdown': drawdown,
        'drawdown_cost_period': drawdown_cost_period,
        'drawdown_period': drawdown_period
    }

def sharpe(pnl, rate=0.002, N=252):
    return (pnl.mean()*N-rate)/pnl.std()/np.sqrt(N)

def sortino(pnl, rate=0.002, N=252):
    return (pnl.mean()*N-rate)/pnl[pnl<0].std()/np.sqrt(N)

def sharpe_yearly(pnl, rate=0.002, N=252):
    return (pnl.groupby([pnl.index.year]).mean()*N-rate)/pnl.groupby([pnl.index.year]).std()/np.sqrt(N)

def sortino_yearly(pnl, rate=0.002, N=252):
    pnln = pnl[pnl<0]
    return (pnl.groupby([pnl.index.year]).mean()*N-rate)/pnln.groupby([pnln.index.year]).std()/np.sqrt(N)


def ic(factor, ret1d):  # lower is better
    # desensitize factor via ranking operation, lower rank is better
    f = factor.rank(1, ascending=True, method='first')
    f_demean = -f.sub(f.mean(axis=1), axis=0)
    ret_demean = ret1d.sub(ret1d.mean(axis=1), axis=0)
    rho = (ret_demean*f_demean).sum(axis=1) / np.sqrt(((ret_demean**2).sum(axis=1) * (f_demean**2).sum(axis=1)))
    return rho


def quick_backtesting(name, f3, ret_c2c, cost_buy=0, cost_sell=0):
    '''
    quick & dirty backtesting function
    equal weights, rebalance all
    '''
    f3 = f3.shift(1)  # use signal from yesterday to select stocks
    r_keep = ret_c2c.loc[f3.index].fillna(0)
    r_buy, r_sell = r_keep*0, r_keep*0  # trade at close

    # equal weights in each portfolio
    f3_counts = f3.apply(pd.Series.value_counts, axis=1)
    f3_weights = 1.0/f3_counts

    weight_curr = f3.apply(lambda x: x.map(f3_weights.loc[x.name]), axis=1) * (r_keep+1) # at close
    weight_prev = weight_curr.shift(1)
    weight_prev0 = weight_prev*(1+r_keep)                                               # hold from t-1 to t

    weight_curr = weight_curr.fillna(0).values
    weight_prev = weight_prev.fillna(0).values
    weight_prev0 = weight_prev0.fillna(0).values
    rank_curr, rank_prev = f3.values, f3.shift(1).values

    weight_buy = np.where(rank_curr == rank_prev, np.where(weight_curr <= weight_prev0, 0, weight_curr-weight_prev0), weight_curr)

    weight_sell = np.where(rank_curr == rank_prev, np.where(weight_curr >= weight_prev0, 0, weight_prev0-weight_curr), weight_prev0)

    weight_keep = np.where(rank_curr == rank_prev, np.where(weight_curr < weight_prev0, weight_curr, weight_prev0), 0)

    weight_buy = pd.DataFrame(weight_buy, columns=f3.columns, index=f3.index)
    weight_sell = pd.DataFrame(weight_sell, columns=f3.columns, index=f3.index)
    weight_keep = pd.DataFrame(weight_keep, columns=f3.columns, index=f3.index)

    pnl_keep = (weight_keep + weight_sell)/(1+r_keep) * r_keep

    # long
    pnl_buy_cost = (weight_buy * (r_buy-cost_buy))/(r_buy+1)
    pnl_sell_cost = (weight_sell * (-r_sell-cost_sell))/(r_sell+1)
    long_stock = pnl_buy_cost + pnl_sell_cost + pnl_keep

    # short
    pnl_buy_cost = (weight_buy * (r_buy+cost_buy))/(r_buy+1)
    pnl_sell_cost = (weight_sell * (-r_sell+cost_sell))/(r_sell+1)
    short_stock = pnl_buy_cost + pnl_sell_cost + pnl_keep

    turnover0 = pd.DataFrame(abs(weight_buy + weight_sell)/2, columns=f3.columns, index=f3.index)
    turnover = pd.concat(pd.DataFrame(turnover0.loc[r].groupby(f3.loc[r]).agg('sum')).T for r in f3.index)
    long_pnl = pd.concat(pd.DataFrame(long_stock.loc[r].groupby(f3.loc[r]).agg('sum')).T for r in f3.index)
    short_pnl = pd.concat(pd.DataFrame(short_stock.loc[r].groupby(f3.loc[r]).agg('sum')).T for r in f3.index)

    return {'name': name,
            'long_stock': long_stock,
            'short_stock': short_stock,
            'weight': weight_curr,
            'long_pnl': long_pnl,
            'short_pnl': short_pnl,
            'counts': f3_counts,
            'turnover': turnover}
