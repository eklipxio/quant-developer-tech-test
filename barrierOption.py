# -*- coding: utf-8 -*-
"""
Created on Mon Jun 10 16:09:18 2024

@author: gsamu
"""

import QuantLib as ql
import pandas as pd

# Load the volatility surface data from CSV file
raw_vol_data = pd.read_csv('volatility_surface_from_clean.csv')

# function used to reformate the raw vol data for ql


def prepare_volatility_matrix(raw_vol_dData, vol_bump):

    expiry_dates = []
    strike_values = []
    vol_values = []

    for index, row in raw_vol_dData.iterrows():
        expiry_date = ql.DateParser.parseISO(row['Expiry'])
        if expiry_date not in expiry_dates:
            expiry_dates.append(expiry_date)
        if row['Strike'] not in strike_values:
            strike_values.append(row['Strike'])
        vol_values.append([expiry_date, row['Strike'], row['Volatility']])

    expiry_dates = sorted(expiry_dates)
    strike_values = sorted(strike_values)

    # Create a matrix of volatilities
    volatility_matrix = ql.Matrix(len(strike_values), len(expiry_dates))
    for k in range(len(strike_values)):
        for t in range(len(expiry_dates)):
            for vol in vol_values:
                if vol[0] == expiry_dates[t] and vol[1] == strike_values[k]:
                    volatility_matrix[k][t] = vol[2]+vol_bump
    return volatility_matrix, expiry_dates, strike_values


###########################
# Option parameters
spot_price = 1174.75  # +11.7475
strike_price = 1200
barrier_type = ql.Barrier.DownOut
barrier_price = 1100
rebate = 0
today = ql.Date(10, 6, 2024)
expiry_date = ql.Date(26, 7, 2024)
rfr = 0.0548  # not able yet to use properly the full yield curve contained in usd_yiueld.csv

dividend_yield = 0.0
option_type = ql.Option.Put
day_count = ql.Actual365Fixed()
calendar = ql.TARGET()

# Construction of the option
payoff = ql.PlainVanillaPayoff(option_type, strike_price)
ql.Settings.instance().evaluationDate = today
exercise = ql.EuropeanExercise(expiry_date)
option = ql.BarrierOption(barrier_type, barrier_price,
                          rebate,  payoff, exercise)

# Market data ex vol
spot_handle = ql.QuoteHandle(ql.SimpleQuote(spot_price))
yc = ql.YieldTermStructureHandle(ql.FlatForward(
    today, rfr, day_count))
dividends = ql.YieldTermStructureHandle(ql.FlatForward(
    today, dividend_yield, day_count))

# volatility preparation
volatility_matrix, expiry_dates, strike_values = prepare_volatility_matrix(
    raw_vol_data, 0.00)
vol_surface = ql.BlackVarianceSurface(
    today, calendar, expiry_dates, strike_values, volatility_matrix, day_count)
vol_for_process = ql.BlackVolTermStructureHandle(vol_surface)

# prepare the pricing engine
process = ql.BlackScholesMertonProcess(
    spot_handle, dividends, yc, vol_for_process)
engine = ql.AnalyticBarrierEngine(process)
option.setPricingEngine(engine)

###########################
# Setup the vanilla option (used to validate this file with a degenerated case of barrier call option)
voption = ql.EuropeanOption(payoff, exercise)
vprocess = ql.BlackScholesProcess(spot_handle, yc, vol_for_process)
vengine = ql.AnalyticEuropeanEngine(vprocess)
voption.setPricingEngine(vengine)
print("-----------------")
print("Vanilla option")
print(f"Price: { voption.NPV():.4}")
print(f"Delta: {voption.delta():.4f}")
print(f"Gamma: {voption.gamma():.10f}")
print(f"Vega: {voption.vega():.4f}")
print(f"Theta: {voption.theta():.4f}")
print("-----------------")

###########################
## Price of the option##
price = option.NPV()
print("Barrier Down & Out option")
print(f"Price: {price:.10f}")

###########################
## simulate delta and gamma##
spot_bump = 0.01*spot_price
spot_handle = ql.QuoteHandle(ql.SimpleQuote(spot_price + spot_bump))
process = ql.BlackScholesMertonProcess(
    spot_handle, dividends, yc, vol_for_process)
engine = ql.AnalyticBarrierEngine(process)
option.setPricingEngine(engine)
price_up = option.NPV()

spot_handle = ql.QuoteHandle(ql.SimpleQuote(spot_price - spot_bump))
process = ql.BlackScholesMertonProcess(
    spot_handle, dividends, yc, vol_for_process)
engine = ql.AnalyticBarrierEngine(process)
option.setPricingEngine(engine)
price_dn = option.NPV()

delta = (price_up - price_dn) / (2 * spot_bump)
print(f"Delta: {delta:.10f}")
gamma = (price_up - 2 * price + price_dn) / (spot_bump * spot_bump)
print(f"Gamma: {gamma:.10f}")

###########################
## simulate vega##
vol_bump = 0.01

spot_handle = ql.QuoteHandle(ql.SimpleQuote(spot_price))
# positive vol shock
volatility_matrix, expiry_dates, strike_values = prepare_volatility_matrix(
    raw_vol_data, vol_bump)
vol_surface = ql.BlackVarianceSurface(
    today, calendar, expiry_dates, strike_values, volatility_matrix, day_count)
vol_up = ql.BlackVolTermStructureHandle(vol_surface)
process = ql.BlackScholesMertonProcess(
    spot_handle, dividends, yc, vol_up)
engine = ql.AnalyticBarrierEngine(process)
option.setPricingEngine(engine)
price_vol_up = option.NPV()

# positive vol shock
volatility_matrix, expiry_dates, strike_values = prepare_volatility_matrix(
    raw_vol_data, -vol_bump)
vol_surface = ql.BlackVarianceSurface(
    today, calendar, expiry_dates, strike_values, volatility_matrix, day_count)
vol_dn = ql.BlackVolTermStructureHandle(vol_surface)
process = ql.BlackScholesMertonProcess(
    spot_handle, dividends, yc, vol_dn)
engine = ql.AnalyticBarrierEngine(process)
option.setPricingEngine(engine)
price_vol_dn = option.NPV()

vega = (price_vol_up - price_vol_dn) / (2 * vol_bump)
print(f"Vega: {vega:.10f}")

###########################
## simulate  theta##
ql.Settings.instance().evaluationDate = today+1
exercise = ql.EuropeanExercise(expiry_date)
option = ql.BarrierOption(barrier_type, barrier_price,
                          rebate,  payoff, exercise)

# Market data ex vol
spot_handle = ql.QuoteHandle(ql.SimpleQuote(spot_price))
yc = ql.YieldTermStructureHandle(ql.FlatForward(
    today+1, rfr, day_count))
dividends = ql.YieldTermStructureHandle(ql.FlatForward(
    today+1, dividend_yield, day_count))

# volatility preparation
volatility_matrix, expiry_dates, strike_values = prepare_volatility_matrix(
    raw_vol_data, 0.00)
vol_surface = ql.BlackVarianceSurface(
    today+1, calendar, expiry_dates, strike_values, volatility_matrix, day_count)
vol_for_process = ql.BlackVolTermStructureHandle(vol_surface)

# prepare the pricing engine
process = ql.BlackScholesMertonProcess(
    spot_handle, dividends, yc, vol_for_process)
engine = ql.AnalyticBarrierEngine(process)
option.setPricingEngine(engine)
price_tom = option.NPV()
theta = (price_tom - price)*365
print(f"Theta: {theta:.10f}")
