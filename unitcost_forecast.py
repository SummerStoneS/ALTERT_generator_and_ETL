"""
@time: 7/8/2019 9:10 PM

@author: 柚子
"""
import pandas as pd
import numpy as np
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
import pmdarima as pm # https://github.com/tgsmith61591/pmdarima
from dateutil import relativedelta
from alert import get_latest_date_from_is_model
from stage_to_app import get_table_from_app, stage_to_app_indireact, add_usd_currency_and_fx_rate_for_cost_app_tables


import warnings
warnings.filterwarnings("ignore", category=Warning)

def get_history_data(data, cur_date, date_col="reporting_month"):
    if data[date_col].dtype == 'O':
        data[date_col] = pd.to_datetime(data[date_col])
    return data.query(f"{date_col} <= @cur_date")

def filter_6_plus_history_months_data(data, month_num=6):
    """
        # need at least 6 months data to do the prediction
    """
    level1_months_num = data.groupby(['bg', 'bu', 'site', 'product', 'cost_item_level1']).size().reset_index()
    level1_months_num.columns = ['bg', 'bu', 'site', 'product', 'cost_item_level1', 'size']
    predictable_index = level1_months_num[level1_months_num['size'] >= month_num]
    return predictable_index

def filter_history_of_one_product_and_one_item(df, kwargs):
    # filter history data for one product one level1 item for next step prediction
    filtered = df.copy()
    for key,value in kwargs.items():
        filtered = filtered.query(f"{key} == @value")
    filtered = filtered.sort_values(by='reporting_month', ascending=True)
    return filtered

class ForecastUnitcost:
    def __init__(self, temp, y_name,standard="shipment",lag=3):
        self.initial = temp.copy()
        self.train = None
        self.y = y_name
        self.y_ts = None
        self.standard = standard
        self.lag = lag

    def remove_nan_inf_y(self):
        self.train = self.initial[(self.initial[self.y] != np.inf) & (self.initial[self.y] != -np.inf)
                         & (self.initial[self.y].notnull())]

    def add_prediction_row(self, dict_row):
        self.train = self.train.append(dict_row, ignore_index=True)

    def cal_moving_average(self, window=None):
        if not window:
            window = self.lag
        self.train[f'ma_unit_cost_{self.standard}'] = self.train.rolling(window=window)[self.y].mean()
        self.train[f'ma_unit_cost_{self.standard}'] = self.train[f'ma_unit_cost_{self.standard}'].shift(1)

    def get_single_y_ts(self):
        self.y_ts = self.train[self.y]
        self.y_ts.drop(self.y_ts.tail(1).index, inplace=True)

    def cal_exp_smooth(self):
        # (b) Simple exponential smoothing
        self.get_single_y_ts()
        # fit model -- libraby bug: division by zero
        try:
            model = SimpleExpSmoothing(self.y_ts)
            model_fit = model.fit()
            # make prediction
            yhat = model_fit.predict(0, len(self.y_ts))
            yhat[:self.lag] = np.nan
            self.train[f'es_unit_cost_{self.standard}'] = yhat.values
        except Exception as e:
            # print(s, p, c)
            # print('exp smoothing occurs error: divide by zero')
            print(e)
            self.train[f'es_unit_cost_{self.standard}'] = np.nan

    def cal_arima(self):
        stepwise_fit = pm.auto_arima(self.y_ts, start_p=0, start_q=0, max_p=self.lag, max_q=self.lag,
                                     error_action='ignore',  # don't want to know if an order does not work
                                     suppress_warnings=True,  # don't want convergence warnings
                                     stepwise=True)  # set to stepwise

        y_in_sample = stepwise_fit.predict_in_sample()
        y_in_sample[:self.lag] = np.nan
        y_predict = stepwise_fit.predict(n_periods=1)
        self.train[f'arima_unit_cost_{self.standard}'] = y_in_sample.tolist() + y_predict.tolist()

    def get_3_predictions(self):
        predictions = self.train[['reporting_month', f'ma_unit_cost_{self.standard}', f'es_unit_cost_{self.standard}',
                                  f'arima_unit_cost_{self.standard}']]
        return predictions


if __name__ == '__main__':

    current_month = get_latest_date_from_is_model()
    next_month = pd.to_datetime(current_month.date() + relativedelta.relativedelta(months=1))   # only predict next month's data

    cost_overview = get_table_from_app("cost_overview_dtl", index="id")             # load data
    df = get_history_data(cost_overview, next_month)

    # remove useless site-item_level1
    df = df.drop(df[(df['site']=='WMI') & (df['cost_item_level1'] == 'MOH_WEKS')].index)
    df = df.drop(df[(df['site']=='WKS') & (df['cost_item_level1'] == 'MOH_WMI')].index)
    # only keep relevant columns
    df = df[['reporting_month', 'bg', 'bu', 'site', 'product', 'cost_item_level1',
             'actual_total_cost', 'actual_shipment', 'actual_production', 'r3m_shipment', 'r3m_production']]
    # replace nulls with 0 - important for aggregration, otherwise return empty dataframe
    # currently, 'bu' is null
    df.fillna(0, inplace=True)
    # aggregate to cost level 1
    df = df.groupby(['reporting_month', 'bg', 'bu', 'site', 'product', 'cost_item_level1'])\
    .agg({'actual_total_cost':'sum','actual_shipment':'mean', 'actual_production':'mean', 'r3m_shipment':'mean', 'r3m_production':'mean'}).reset_index()
    # predict unit cost
    df['actual_unit_cost_per_ship'] = df['actual_total_cost'] / df['actual_shipment']       # NaNs will be removed later
    df['actual_unit_cost_per_prod'] = df['actual_total_cost'] / df['actual_production']

    df_index =filter_6_plus_history_months_data(df, month_num=6)        # need train data points nums to be at least 6

    results = pd.DataFrame([])
    for i, row in df_index.iterrows():
        # obtain site, product and cost item level 1
        filter_dict = dict(row[:5])
        temp = filter_history_of_one_product_and_one_item(df, filter_dict)
        filter_dict["reporting_month"] = pd.to_datetime(next_month)

        for target_y, quantity_standard in [('actual_unit_cost_per_ship',"shipment"),
                                            ('actual_unit_cost_per_prod',"production")]:
            model = ForecastUnitcost(temp, target_y, quantity_standard, lag=3)
            model.remove_nan_inf_y()
            model.add_prediction_row(filter_dict)
            if len(model.train) >= 3:
                model.cal_moving_average(window=3)          # (a) simple moving average - 3 months
                model.cal_exp_smooth()
                model.cal_arima()
                predict = model.get_3_predictions()
                temp = temp.merge(predict, on='reporting_month', how='left')
        results = results.append(temp)

    results = add_usd_currency_and_fx_rate_for_cost_app_tables(results)
    stage_to_app_indireact("cost_prediction_dtl", results)
