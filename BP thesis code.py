# this code was written with the help of ORF 498 & ORF 499 preceptors and ChatGPT

#two sample t test

import pandas as pd
from scipy import stats
from datetime import timedelta

path_trades = "/content/cleandata.xlsx"
path_closed_meetings = "/content/Congressional Meetings.xlsx"
path_open_meetings = "/content/open meetings.xlsx"

trades_df = pd.read_excel(path_trades)
closed_meetings_df = pd.read_excel(path_closed_meetings)
open_meetings_df = pd.read_excel(path_open_meetings)

trades_df['date'] = pd.to_datetime(trades_df['date'])
closed_meetings_df['date'] = pd.to_datetime(closed_meetings_df['date'])
open_meetings_df['date'] = pd.to_datetime(open_meetings_df['date'])

def calculate_differences(meetings_df, trades_df):
    differences = []
    for meeting_date in meetings_df['date']:
        day_before = meeting_date - timedelta(days=1)
        day_before = adjust_for_weekend(day_before)
        value_day_before = trades_df.loc[trades_df['date'] == day_before, 'value'].sum()
        value_day_of = trades_df.loc[trades_df['date'] == meeting_date, 'value'].sum()
        differences.append(value_day_of - value_day_before)
    return differences

def adjust_for_weekend(date):
    if date.weekday() == 5:
        date -= timedelta(days=1)
    elif date.weekday() == 6:
        date += timedelta(days=1)
    return date

differences_closed = calculate_differences(closed_meetings_df, trades_df)
differences_open = calculate_differences(open_meetings_df, trades_df)

t_stat, p_value = stats.ttest_ind(differences_closed, differences_open, equal_var=False)

print(f"T-Statistic: {t_stat}, P-Value: {p_value}")

#ARIMA not lagged
import itertools
import warnings
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt
import statsmodels.api as sm

trades = pd.read_excel('/content/cleandata.xlsx')
open_meetings = pd.read_excel('/content/open meetings.xlsx')
closed_meetings = pd.read_excel('/content/Congressional Meetings.xlsx')


trades['date'] = pd.to_datetime(trades['date'])
open_meetings['date'] = pd.to_datetime(open_meetings['date'])
closed_meetings['date'] = pd.to_datetime(closed_meetings['date'])

trades.set_index('date', inplace=True)
open_meetings.set_index('date', inplace=True)
closed_meetings.set_index('date', inplace=True)

start_date = '2015-01-01'
end_date = '2020-12-31'
trades = trades.loc[start_date:end_date]

trades = trades[trades.index.weekday < 5]

trades['OpenMeeting'] = trades.index.isin(open_meetings.index).astype(int)
trades['ClosedMeeting'] = trades.index.isin(closed_meetings.index).astype(int)

trades.fillna(0, inplace=True)

def fit_arima_model(trades, exog_data):
    best_aic = float("inf")
    best_order = None
    p_values = range(0, 2)
    d_values = range(0, 2)
    q_values = range(0, 2)

    for p, d, q in itertools.product(p_values, d_values, q_values):
        order = (p, d, q)
        try:
            model = sm.tsa.ARIMA(trades, exog=exog_data, order=order)
            model_fit = model.fit()
            aic = model_fit.aic
            if aic < best_aic:
                best_aic = aic
                best_order = order
        except Exception as e:
            print(f"Failed to fit model with order {order}: {str(e)}")
            continue

    if best_order is not None:
        model = sm.tsa.ARIMA(trades, exog=exog_data, order=best_order)
        model_fit = model.fit()
        print("Order (p, d, q):", best_order)
        print(model_fit.summary())
        return model_fit
    else:
        print("Failed to find a suitable model.")
        return None

exog_data = trades[['OpenMeeting', 'ClosedMeeting']]
model_fit = fit_arima_model(trades['value'], exog_data=exog_data)

plt.figure(figsize=(15, 6))
plt.plot(trades['value'], label='Original')
plt.plot(model_fit.fittedvalues, color='red', label='Fitted values')
plt.title('ARIMA Model Fit for 2015-2020 (Not Lagged)')
plt.xlabel('Date')
plt.ylabel('Value')
plt.legend()
plt.show()

import numpy as np

model_restricted = sm.tsa.ARIMA(trades['value'], order=(1, 1, 1)).fit()

rss_unrestricted = np.sum(model_fit.resid**2)
rss_restricted = np.sum(model_restricted.resid**2)

n = trades.shape[0]
p = 2
k = len(model_fit.params)

F = ((rss_restricted - rss_unrestricted) / p) / (rss_unrestricted / (n - k))
print("F-statistic:", F)

from scipy.stats import f
p_value = 1 - f.cdf(F, p, n-k)
print("p-value:", p_value)

#ARIMA lagged
trades = pd.read_excel('/content/cleandata.xlsx')
open_meetings = pd.read_excel('/content/open meetings.xlsx')
closed_meetings = pd.read_excel('/content/Congressional Meetings.xlsx')

trades['date'] = pd.to_datetime(trades['date'])
open_meetings['date'] = pd.to_datetime(open_meetings['date'])
closed_meetings['date'] = pd.to_datetime(closed_meetings['date'])

trades.set_index('date', inplace=True)
open_meetings.set_index('date', inplace=True)
closed_meetings.set_index('date', inplace=True)

start_date = '2015-01-01'
end_date = '2020-12-31'
trades = trades.loc[start_date:end_date]

trades = trades[trades.index.weekday < 5]

trades['OpenMeeting'] = trades.index.isin(open_meetings.index).astype(int)
trades['ClosedMeeting'] = trades.index.isin(closed_meetings.index).astype(int)

trades['LaggedOpenMeeting'] = trades['OpenMeeting'].shift(1).fillna(0).astype(int)
trades['LaggedClosedMeeting'] = trades['ClosedMeeting'].shift(1).fillna(0).astype(int)

trades.fillna(0, inplace=True)

def fit_arima_model(trades, exog_data):
    best_aic = float("inf")
    best_order = None
    p_values = range(0, 2)
    d_values = range(0, 2)
    q_values = range(0, 2)

    for p, d, q in itertools.product(p_values, d_values, q_values):
        order = (p, d, q)
        try:
            model = ARIMA(trades, exog=exog_data, order=order)
            model_fit = model.fit()
            aic = model_fit.aic
            if aic < best_aic:
                best_aic = aic
                best_order = order
        except Exception as e:
            print(f"Failed to fit ARIMA model with order {order}: {e}")
            continue

    if best_order is not None:
        model = ARIMA(trades, exog=exog_data, order=best_order)
        model_fit = model.fit()
        print("Order (p, d, q):", best_order)
        print(model_fit.summary())
        return model_fit
    else:
        print("Failed to find a suitable ARIMA model.")
        return None

exog_data = trades[['LaggedOpenMeeting', 'LaggedClosedMeeting']]
model_fit = fit_arima_model(trades['value'], exog_data=exog_data)

import matplotlib.pyplot as plt
plt.figure(figsize=(15, 6))
plt.plot(trades['value'], label='Original')
plt.plot(model_fit.fittedvalues, color='red', label='Fitted values')
plt.title('ARIMA Model Fit (Lagged)')
plt.xlabel('Date')
plt.ylabel('Value')
plt.legend()
plt.show()

model_restricted = sm.tsa.ARIMA(trades['value'], order=(1, 1, 1)).fit()

rss_unrestricted = np.sum(model_fit.resid**2)
rss_restricted = np.sum(model_restricted.resid**2)


n = trades.shape[0]
p = 2
k = len(model_fit.params)

F = ((rss_restricted - rss_unrestricted) / p) / (rss_unrestricted / (n - k))
print("F-statistic:", F)

p_value = 1 - f.cdf(F, p, n-k)
print("p-value:", p_value)

#RDD closed door
import pandas as pd
import numpy as np
import statsmodels.api as sm
from datetime import timedelta
from pandas.tseries.offsets import BDay

cleandata_path = "/content/cleandata.xlsx"
congressional_meetings_path = "/content/Congressional Meetings.xlsx"

try:
    cleandata = pd.read_excel(cleandata_path)
    congressional_meetings = pd.read_excel(congressional_meetings_path)
except FileNotFoundError:
    print("File not found. Please provide correct file paths.")
    exit(1)
except Exception as e:
    print(f"An error occurred: {str(e)}")
    exit(1)

cleandata['date'] = pd.to_datetime(cleandata['date'])
congressional_meetings['date'] = pd.to_datetime(congressional_meetings['date'])

def extract_event_windows_with_date(event_dates, data):
    windows_with_date = []
    for event_date in event_dates:
        start_date = event_date - BDay(5)
        end_date = event_date + BDay(5)
        window = data[(data['date'] >= start_date) & (data['date'] <= end_date)]
        window = window[window['date'].dt.dayofweek < 5]
        window = window[window['date'] != event_date]
        windows_with_date.append((event_date, window))
    return windows_with_date

event_windows_with_date = extract_event_windows_with_date(congressional_meetings['date'], cleandata)

results = []
for event_date, window in event_windows_with_date:
    if not window.empty:
        window['time'] = (window['date'] - window['date'].min()).dt.days
        event_day = window['time'].min() + window['time'].median()
        window['post_event'] = (window['time'] > event_day).astype(int)
        model = sm.OLS(window['value'], sm.add_constant(window[['time', 'post_event']]))
        result = model.fit()
        level_change = result.params['post_event']
        p_value = result.pvalues['post_event']
        results.append({'event_date': event_date, 'level_change': level_change, 'p_value': p_value})

all_results_df = pd.DataFrame(results)
print(all_results_df)

#RDD open door
cleandata_path = "/content/cleandata.xlsx"
congressional_meetings_path = "/content/open meetings.xlsx"

try:
    cleandata = pd.read_excel(cleandata_path)
    congressional_meetings = pd.read_excel(congressional_meetings_path)
except FileNotFoundError:
    print("File not found. Please provide correct file paths.")
    exit(1)
except Exception as e:
    print(f"An error occurred: {str(e)}")
    exit(1)

cleandata['date'] = pd.to_datetime(cleandata['date'])
congressional_meetings['date'] = pd.to_datetime(congressional_meetings['date'])

def extract_event_windows_with_date(event_dates, data):
    windows_with_date = []
    for event_date in event_dates:
        start_date = event_date - BDay(5)
        end_date = event_date + BDay(5)
        window = data[(data['date'] >= start_date) & (data['date'] <= end_date)]
        window = window[window['date'].dt.dayofweek < 5]
        window = window[window['date'] != event_date]
        windows_with_date.append((event_date, window))
    return windows_with_date

event_windows_with_date = extract_event_windows_with_date(congressional_meetings['date'], cleandata)

results = []
for event_date, window in event_windows_with_date:
    if not window.empty:
        window['time'] = (window['date'] - window['date'].min()).dt.days
        event_day = window['time'].min() + window['time'].median()
        window['post_event'] = (window['time'] > event_day).astype(int)
        model = sm.OLS(window['value'], sm.add_constant(window[['time', 'post_event']]))
        result = model.fit()
        level_change = result.params['post_event']
        p_value = result.pvalues['post_event']
        results.append({'event_date': event_date, 'level_change': level_change, 'p_value': p_value})

all_results_df = pd.DataFrame(results)
print(all_results_df)

#RDD market volume

cleandata_path = "/content/market volume.xlsx"
congressional_meetings_path = "/content/Congressional Meetings.xlsx"

try:
    cleandata = pd.read_excel(cleandata_path)
    congressional_meetings = pd.read_excel(congressional_meetings_path)
except FileNotFoundError:
    print("File not found. Please provide correct file paths.")
    exit(1)
except Exception as e:
    print(f"An error occurred: {str(e)}")
    exit(1)

cleandata['date'] = pd.to_datetime(cleandata['date'])
congressional_meetings['date'] = pd.to_datetime(congressional_meetings['date'])

def extract_event_windows_with_date(event_dates, data):
    windows_with_date = []
    for event_date in event_dates:
        start_date = event_date - BDay(5)
        end_date = event_date + BDay(5)
        window = data[(data['date'] >= start_date) & (data['date'] <= end_date)]
        window = window[window['date'].dt.dayofweek < 5]
        window = window[window['date'] != event_date]
        windows_with_date.append((event_date, window))
    return windows_with_date

event_windows_with_date = extract_event_windows_with_date(congressional_meetings['date'], cleandata)


results = []
for event_date, window in event_windows_with_date:
    if not window.empty:
        window['time'] = (window['date'] - window['date'].min()).dt.days
        event_day = window['time'].min() + window['time'].median()
        window['post_event'] = (window['time'] > event_day).astype(int)
        model = sm.OLS(window['value'], sm.add_constant(window[['time', 'post_event']]))
        result = model.fit()
        level_change = result.params['post_event']
        p_value = result.pvalues['post_event']
        results.append({'event_date': event_date, 'level_change': level_change, 'p_value': p_value})


all_results_df = pd.DataFrame(results)
print(all_results_df)
