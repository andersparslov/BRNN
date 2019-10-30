from load import split_df, sort_links
from common import transform, fit_scale, roll, eval_quantiles, compute_error

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from matplotlib.dates import DateFormatter


def get_dat(lags, preds, start_train, end_train, end_test):
    data = pd.read_csv('link_travel_time_local.csv.gz', compression='gzip', parse_dates = True, index_col = 0)

    ## Sort links by order 
    data, order = sort_links(data, '1416:1417', '7051:2056')
    ## Make a link order column e.g here the neighbouring links for link 1 are 0 and 2.
    data['link_order'] = data['link_ref'].astype('category')
    not_in_list = data['link_order'].cat.categories.difference(order)
    data['link_order'] = data['link_order'].cat.set_categories(np.hstack((order, not_in_list)), ordered=True)
    data['link_order'] = data['link_order'].cat.codes
    ## Add week of day column [Monday, ..., Sunday] = [0, ..., 6]
    data['Weekday'] = data.index.weekday_name
    data = data.sort_values('link_order')

    print("Number of observations = ", len(data))
    print("Number of links = ", len(data['link_ref'].unique()))

    data_train, data_test = split_df(data, start_train = start_train, end_train = end_train, end_test = end_test)
    print("\nTraining from", data_train.sort_index().index[0], "to", data_train.sort_index().index[-1])
    print("Testing  from", data_test.sort_index().index[0], "to", data_test.sort_index().index[-1])

    ## Transform train and test set using the mean and std for train set.
    means_df, scales, low_df, upr_df = fit_scale(data_train, order)
    ix_train, ts_train,  rm_mean_train, rm_scale_train, w_train, lns_train = transform(data_train, 
                                                                                           means_df, 
                                                                                           scales, 
                                                                                           order,
                                                                                           freq = '15min')
    ix_test, ts_test, rm_mean_test, rm_scale_test, w_test, lns_test = transform(data_test, 
                                                                                     means_df, 
                                                                                     scales, 
                                                                                     order,
                                                                                     freq = '15min')
    ## Create rolling window tensor
    ##  - y_mean and y_std are arrays where columns are each link and 
    ##    the rows corresponding to the mean and std of each data point
    ##    at that weekday. 

    ##  - y_num_meas indicates how many measurements are in the time window
    ##    for a given link
    X_train, y_train, y_ix_train, y_mean_train, y_std_train, y_num_meas_train = roll(ix_train, 
                                                                                     ts_train, 
                                                                                     rm_mean_train, 
                                                                                     rm_scale_train, 
                                                                                     w_train, 
                                                                                     lags, 
                                                                                     preds)
    X_test, y_test, y_ix_test, y_mean_test, y_std_test, y_num_meas_test = roll(ix_test, 
                                                                               ts_test, 
                                                                               rm_mean_test, 
                                                                               rm_scale_test, 
                                                                               w_test, 
                                                                               lags, 
                                                                               preds)
    return X_train, X_test, y_train, y_test, y_std_train, y_std_test, y_num_meas_train, y_num_meas_test