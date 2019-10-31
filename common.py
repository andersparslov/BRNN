import pandas as pd
import numpy as np

def fit_scale(data, order, ref_freq = '15min'):
    means = { }
    scales = { }
    low = { }
    upr = { }

    grouping = data[data['link_travel_time'].notnull()].groupby('link_ref', sort = False)
    for link_ref, data_link in grouping:
        # Fit outlier bounds using MAD
        median = data_link.groupby('Weekday')['link_travel_time'].median()
        error = pd.concat([data_link['Weekday'], np.abs(data_link['link_travel_time'] - median[data_link['Weekday']].values)], axis = 1)
        mad = 1.4826 * error.groupby('Weekday')['link_travel_time'].median()

        _low = median - 3 * mad
        _upr = median + 3 * mad
        mask = (_low[data_link['Weekday']].values < data_link['link_travel_time']) & (data_link['link_travel_time'] < _upr[data_link['Weekday']].values)
        data_link_no = data_link[mask]

        _mean = data_link_no.groupby(['Weekday', 'TOD'])['link_travel_time'].mean()
        means[link_ref] = _mean
        scale = data_link_no.groupby(['Weekday', 'TOD'])['link_travel_time'].std()
        scales[link_ref] = scale

        low[link_ref] = _low
        upr[link_ref] = _upr

    means_df = pd.DataFrame(data=means).interpolate()
    scales_df = pd.DataFrame(data=scales).interpolate()
    low_df = pd.DataFrame(data=low).interpolate()
    upr_df = pd.DataFrame(data=upr).interpolate()

    ## Correct order of links
    means_df = means_df[order]
    scales_df = scales_df[order]
    low_df = low_df[order]
    upr_df = upr_df[order]

    # Fill NaNs    
    means_df = means_df.fillna(method='pad').fillna(method='bfill')
    scales_df = scales_df.fillna(method='pad').fillna(method='bfill')
    low_df = low_df.fillna(method='pad').fillna(method='bfill')
    upr_df = upr_df.fillna(method='pad').fillna(method='bfill')
    
    return means_df, scales_df

def transform(data, means_df, scales, order, freq = '15min'):
    tss = { }
    ws = { }
    removed_mean = { }
    removed_scale = { }
    lnk_list = []
    for lnk, data_link in data.groupby('link_ref', sort = False):
        # Link Data Time Indexed
        link_time_ix = pd.DatetimeIndex(data_link.index)
        data_link = data_link.set_index(link_time_ix)
        # Link Reference Data Index
        ix_ref = data_link['Weekday']

        link_travel_time_k = data_link['link_travel_time'].resample(freq).mean()
        removed_mean[lnk] = pd.Series(data = means_df.loc[ix_ref, lnk].values, 
                                      index = link_time_ix).resample(freq).mean()
        removed_scale[lnk] = pd.Series(data = np.repeat(scales[lnk], link_travel_time_k.shape[0]), 
                                       index = link_travel_time_k.index)
        tss[lnk] = (link_travel_time_k - removed_mean[lnk].values) / removed_scale[lnk].values
        ws[lnk] = data_link['link_travel_time'].resample(freq).count()
        lnk_list.append(lnk)
        
    ts = pd.DataFrame(data = tss).fillna(method='pad').fillna(0) 
    df_removed_mean = pd.DataFrame(data = removed_mean, index = ts.index).fillna(method='pad').fillna(method='bfill') 
    df_removed_scale = pd.DataFrame(data = removed_scale, index = ts.index).fillna(method='pad').fillna(method='bfill')    
    w = pd.DataFrame(data = ws).fillna(0) # Link Travel Time Weights, e.g. number of measurements
    
    ## Correct the order
    ts = ts[order]
    df_removed_mean = df_removed_mean[order]
    df_removed_scale = df_removed_scale[order]
    w = w[order]
    return (ts.index, ts.values, df_removed_mean.values, df_removed_scale.values, w.values, lnk_list)

def transform2(data, means_df, scales_df, order, freq = '15min'):
    tss = { }
    ws = { }
    removed_mean = { }
    removed_scale = { }
    lnk_list = []
    for lnk, data_link in data.groupby('link_ref', sort = False):
        # Link Data Time Indexed
        link_time_ix = pd.DatetimeIndex(data_link.index)
        data_link = data_link.set_index(link_time_ix)
        # Link Reference Data Index
        ix_week = data_link['Weekday'].tolist()
        ix_tod = data_link['TOD'].tolist()
        ## Create multi index for the two lists
        mult_ind = pd.MultiIndex.from_arrays([ix_week, ix_tod])

        link_travel_time_k = data_link['link_travel_time'].resample(freq).mean()
        removed_mean[lnk] = pd.Series(data=means_df[lnk].loc[mult_ind].values, 
                                      index = link_time_ix).resample(freq).mean()
        removed_scale[lnk] = pd.Series(data =scales_df[lnk].loc[mult_ind].values, 
                                       index = link_time_ix).resample(freq).mean()
        tss[lnk] = (link_travel_time_k - removed_mean[lnk].values) / removed_scale[lnk].values
        ws[lnk] = data_link['link_travel_time'].resample(freq).count()
        lnk_list.append(lnk)

    ts = pd.DataFrame(data = tss).fillna(method='pad').fillna(0) 
    df_removed_mean = pd.DataFrame(data = removed_mean, index = ts.index).fillna(method='pad').fillna(method='bfill') 
    df_removed_scale = pd.DataFrame(data = removed_scale, index = ts.index).fillna(method='pad').fillna(method='bfill')    
    w = pd.DataFrame(data = ws).fillna(0) # Link Travel Time Weights, e.g. number of measurements
    return ts.index, ts.values, df_removed_mean.values, df_removed_scale.values

def fit_scale2(data, order, ref_freq = '15min'):
    means = { }
    scales = { }
    low = { }
    upr = { }

    grouping = data[data['link_travel_time'].notnull()].groupby('link_ref', sort = False)
    for link_ref, data_link in grouping:
        # Fit outlier bounds using MAD
        median = data_link.groupby('Weekday')['link_travel_time'].median()
        error = pd.concat([data_link['Weekday'], np.abs(data_link['link_travel_time'] - median[data_link['Weekday']].values)], axis = 1)
        mad = 1.4826 * error.groupby('Weekday')['link_travel_time'].median()

        _low = median - 3 * mad
        _upr = median + 3 * mad
        mask = (_low[data_link['Weekday']].values < data_link['link_travel_time']) & (data_link['link_travel_time'] < _upr[data_link['Weekday']].values)
        data_link_no = data_link[mask]

        means[link_ref] = data_link_no.groupby(['Weekday', 'TOD'])['link_travel_time'].mean()
        scales[link_ref] = data_link_no.groupby(['Weekday', 'TOD'])['link_travel_time'].std()

    means_df = pd.DataFrame(data=means).interpolate()
    scales_df = pd.DataFrame(data=scales).interpolate()

    ## Correct order of links
    means_df = means_df[order]
    scales_df = scales_df[order]

    # Fill NaNs    
    means_df = means_df.fillna(method='pad').fillna(method='bfill')
    scales_df = scales_df.fillna(method='pad').fillna(method='bfill')
    return means_df, scales_df

def transform_mcat(data, means_df, scales_df, order, freq = '15min'):
    ts = {}
    for link, data_link in data.groupby('link_ref'):
        dat_standardized = 0
        for w in range(7):
            for t in range(6):
                dat = data_link[np.logical_and(data_link.Weekday == weekday, data_link.TOD == t)]
                
                
        ##dat_standardized = data_link.groupby(['Weekday', 'TOD']).transform(lambda x: (x - x.mean()) / x.std() )
        dat_standardized = (data_link.groupby(['Weekday', 'TOD'])['link_travel_time'] - means_df[link]) / scales_df[link]
        ts[link] = dat_standardized['link_travel_time'].resample(freq).mean()
    ts = pd.DataFrame(data=ts).fillna(method='pad').fillna(0) 
    return ts.index, ts.values

def roll(ix, ts, removed_mean, removed_scale, w, lags, preds):
    X = np.stack([np.roll(ts, i, axis = 0) for i in range(lags, 0, -1)], axis = 1)[lags:-preds,]
    Y = np.stack([np.roll(ts, -i, axis = 0) for i in range(0, preds, 1)], axis = 1)[lags:-preds,]
    Y_ix = ix[lags:-preds]
    Y_mean = np.stack([np.roll(removed_mean, -i, axis = 0) for i in range(0, preds, 1)], axis = 1)[lags:-preds,]
    Y_scale = np.stack([np.roll(removed_scale, -i, axis = 0) for i in range(0, preds, 1)], axis = 1)[lags:-preds,]
    w_y = np.stack([np.roll(w, -i, axis = 0) for i in range(0, preds, 1)], axis = 1)[lags:-preds,]

    return X, Y, Y_ix, Y_mean, Y_scale, w_y

def roll2(ix, ts, removed_mean, removed_scale, lags, preds):
    X = np.stack([np.roll(ts, i, axis = 0) for i in range(lags, 0, -1)], axis = 1)[lags:-preds,]
    Y = np.stack([np.roll(ts, -i, axis = 0) for i in range(0, preds, 1)], axis = 1)[lags:-preds,]
    Y_ix = ix[lags:-preds]
    Y_mean = np.stack([np.roll(removed_mean, -i, axis = 0) for i in range(0, preds, 1)], axis = 1)[lags:-preds,]
    Y_scale = np.stack([np.roll(removed_scale, -i, axis = 0) for i in range(0, preds, 1)], axis = 1)[lags:-preds,]
    return X, Y, Y_ix, Y_mean, Y_scale

## Filipe's prediction evaluation function
def compute_error(trues, predicted):
    corr = np.corrcoef(predicted, trues)[0,1]
    mae = np.mean(np.abs(predicted - trues))
    mse = np.mean((predicted - trues)**2)
    rae = np.sum(np.abs(predicted - trues)) / np.sum(np.abs(trues - np.mean(trues)))
    rmse = np.sqrt(np.mean((predicted - trues)**2))
    r2 = max(0, 1 - np.sum((trues-predicted)**2) / np.sum((trues - np.mean(trues))**2))
    return corr, mae, mse, rae, rmse, r2

## Filipe's quantile evaluation function
def eval_quantiles(lower, upper, trues, preds):
    N = len(trues)
    icp = np.sum(np.logical_and( (trues>lower),(trues<upper) )) / N
    diffs = np.maximum(0, upper-lower)
    mil = np.sum(diffs) / N
    rmil = 0.0
    for i in range(N):
        if trues[i] != preds[i]:
            rmil += diffs[i] / (np.abs(trues[i]-preds[i]))
    rmil = rmil / N
    clc = np.exp(-rmil*(icp-0.95))
    return icp, mil, rmil, clc