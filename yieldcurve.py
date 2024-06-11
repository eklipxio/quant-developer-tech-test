# -*- coding: utf-8 -*-
"""
Created on Mon Jun 10 23:42:43 2024

@author: gsamu
"""
import QuantLib as ql
import pandas as pd

today = ql.Date(10, 6, 2024)
calendar = ql.TARGET()
day_count = ql.Actual365Fixed()

yield_curve_data = pd.read_csv('usd_yield.csv')


mask = yield_curve_data.apply(lambda row: row.astype(
    str).str.contains("Month", case=False).any(), axis=1)
deposit_curve = yield_curve_data[yield_curve_data.apply(lambda row: row.astype(
    str).str.contains("Month", case=False).any(), axis=1)]
deposit_curve = deposit_curve.drop(columns=['maturity'])
deposit_curve['name'] = deposit_curve['name'].str.replace(" Month", '')
deposit_curve = deposit_curve.rename(columns={'name': 'maturity'})

print("Original DataFrame:")
print(yield_curve_data)
print("Filtered DataFrame:")
print(deposit_curve)

deposit_maturities = [ql.Period(int(element), ql.Months)
                      for element in deposit_curve['maturity']]
deposit_rates = deposit_curve['rate'].astype(float).tolist()

for rate, maturity in zip(deposit_rates, deposit_maturities):
    print(float(rate), maturity)

DepositRateHelper = [ql.DepositRateHelper(ql.QuoteHandle(ql.SimpleQuote(rate)),
                                          maturity, 2, calendar,
                                          ql.ModifiedFollowing, False, day_count)
                     for rate, maturity in zip(deposit_rates, deposit_maturities)]
