import pandas as pd
import numpy as np
import glob

## Input file Paths
path = './'
contracts_path = path + '/contracts'
fuel_prices_path = path + '/fuelPrices'
plant_parameters_path = path + '/plantParameters'
power_prices_path = path + '/powerPrices'

output_path = path + '/output'


# Function to read all CSV Files in the folder in one dataframe.
def ConcatCSV(dir_path):
    all_files = glob.glob(dir_path + "/*.csv")
    li = []
    for filename in all_files:
        df = pd.read_csv(filename, index_col = 'Date', parse_dates= True)
        li.append(df)
    data = pd.concat(li, axis = 0)
    return data


# Task 1: Read all CSV Files
contracts = pd.read_csv(contracts_path +"/Contracts.csv")
plant_parameters = pd.read_csv(plant_parameters_path+ "/Plant_Parameters.csv")
fuel_prices = ConcatCSV(fuel_prices_path)
power_prices = ConcatCSV(power_prices_path)


# Function to calculate basic stats like Mean, Max, min, and Standard deviation of the prices.
def GetBasicStats(df):
    basic_stats = df.groupby(by=['SettlementPoint', df.index.year, df.index.month]).agg({"Price": [np.mean, min, max, np.std]})
    basic_stats.rename_axis(index = ['SettlementPoint', 'Year', 'Month'], inplace = True)
    basic_stats.reset_index(inplace = True)
    basic_stats.columns = basic_stats.columns.droplevel(0)
    basic_stats.set_axis(['SettlementPoint',  'Year', 'Month', 'Mean', 'Min', 'Max', 'SD'], axis='columns', inplace=True)
    return basic_stats


# Task 2: Calculate basic Descriptive statistics.
basic_stats = GetBasicStats(power_prices)


# Helper function to calculate volatility on log returns.
def CalcVolatility(df):
    log_ret  = np.log(df.Price) - np.log(df.Price.shift(1))
    return np.std(log_ret)


# Function to Calculate the Volatility in Monthly-Yearly format.
def CalcHourlyVolatilityByMonth(df):
    prices = df[(df['Price']>0)]
    volatility = prices.groupby(by = ['SettlementPoint', prices.index.year, prices.index.month]).apply(CalcVolatility)
    volatility.rename_axis(index = ['SettlementPoint', 'Year', 'Month'], inplace = True)
    vol = volatility.to_frame('Volatility').reset_index()
    return vol


# Task 3: Calculate Volatility.
vol = CalcHourlyVolatilityByMonth(power_prices)

# Task 4: Write power price statistics to the file.
stats = pd.merge(basic_stats, vol,  how='left', left_on=['SettlementPoint','Year', 'Month'], right_on = ['SettlementPoint','Year', 'Month'])
stats.to_csv(output_path + '/MonthlyPowerPriceStatistics.csv', index = False)


################## Contract Valuations ######################

# Function to expand dataframe in given date ranges.
def GetExpandedContracts(df, freq):
    return pd.concat([pd.DataFrame({'Start': pd.date_range(row.StartDate, row.EndDate, freq=freq),
               'ContractName': row.ContractName,
               'DealType': row.DealType,
               'Volume': row.Volume,
               'Granularity': row.Granularity,
               'StrikePrice': row.StrikePrice,
               'Premium': row.Premium,
               'PriceName': row.PriceName,}, columns=['Start', 'ContractName', 'DealType', 'Volume', 'Granularity', 'StrikePrice', 'Premium', 'PriceName'])
           for i, row in df.iterrows()], ignore_index=True)


daily_contracts = contracts[contracts.Granularity == 'Daily']
hourly_contracts = contracts[contracts.Granularity == 'Hourly']

# Task 5 : Expand the contracts across relevant time periods.
daily_expanded =  GetExpandedContracts(daily_contracts, 'D')
hourly_expanded = GetExpandedContracts(hourly_contracts, 'H')

# Task 6: Join relevant prices

# Joining fuel prices to daily contracts.
fuel_prices = fuel_prices.reset_index()
daily_prices = pd.merge(daily_expanded, fuel_prices,  how='left', left_on=['Start','PriceName'], right_on = ['Date','Variable'])

# Joining power prices to hourly contracts.
power_prices.reset_index(inplace=True)
hourly_prices = pd.merge(hourly_expanded, power_prices,  how='left', left_on=['Start','PriceName'], right_on = ['Date','SettlementPoint'])


# Function to calculate payoffs according to contract type.
def CalculatePayoff(df):
    filt = df['DealType'] == 'Swap'
    df.loc[filt, 'Payoff'] = (df[filt].Price - df[filt].StrikePrice) * df[filt].Volume

    opt_filt = df['DealType'] == 'European option'
    df.loc[opt_filt, 'Payoff'] = ((df[opt_filt].Price - df[opt_filt].StrikePrice).clip(0, None) - df[
        opt_filt].Premium) * df[opt_filt].Volume

    return df


# Task 7: Calculating the payoffs.
daily_payoffs = CalculatePayoff(daily_prices)
hourly_payoffs = CalculatePayoff(hourly_prices)


# Function to calculate aggregate payoffs.
def CalculateTotalPayoffs(df):
    payoffs =  df.groupby(by = ['ContractName', df.Date.dt.year, df.Date.dt.month])['Payoff'].sum()
    payoffs.rename_axis(index = ['ContractName', 'Year', 'Month'], inplace = True)
    total_payoffs = payoffs.to_frame('TotalPayoff').reset_index()
    return total_payoffs


# Task 8: Calculate aggregate Payoffs.
daily_total_payoffs = CalculateTotalPayoffs(daily_prices)
hourly_total_payoffs = CalculateTotalPayoffs(hourly_prices)

total_payoffs = pd.concat([daily_total_payoffs, hourly_total_payoffs])

# Task 9: Write results to the file.
total_payoffs.to_csv(output_path + '/MonthlyContractPayoffs.csv', index= False)


# ############################# Plant Dispatch Modelling #########################################

# Task 10: Calculate the hourly running cost of each power plant included in the Plant_Parameters.csv input file.

fuel_prices['date'] = fuel_prices.Date.dt.date
fuel_prices['Month'] = fuel_prices.Date.dt.month
fuel_prices['Year'] = fuel_prices.Date.dt.year

daily_running_costs = pd.merge(fuel_prices, plant_parameters,  how='right', left_on=['Variable', 'Month', 'Year'], right_on = ['FuelPriceName', 'Month', 'Year'])
daily_running_costs['RunningCost'] = ((daily_running_costs['Price']+ daily_running_costs['FuelTransportationCost'])*daily_running_costs['HeatRate']) + daily_running_costs['VOM']

# Task 11: Join the hourly power prices.

power_prices['date'] = power_prices.Date.dt.date
hourly_running_costs = pd.merge(power_prices, daily_running_costs,  how='right', left_on=['SettlementPoint', 'date'], right_on = ['PowerPriceName', 'date'])
hourly_running_costs.drop(['date', 'Date_y', 'Month', 'Year'], inplace= True, axis=1)
hourly_running_costs.rename(columns={'Date_x':'Date', 'Price_x':'PowerPrice', 'Price_y':'FuelPrice'}, inplace=True)

# Task 12: Task 12: Identify hours in which the power plant should be on.

hourly_running_costs.loc[(hourly_running_costs['PowerPrice'] > hourly_running_costs['RunningCost']), 'Generation'] = hourly_running_costs['Capacity']
hourly_running_costs.loc[(hourly_running_costs['PowerPrice'] <= hourly_running_costs['RunningCost']), 'Generation'] = 0

hourly_running_costs['RunningMargin'] = (hourly_running_costs['PowerPrice'] - hourly_running_costs['RunningCost'])*hourly_running_costs['Generation']

# Task 13: Account for start costs
# Wan't able to finish due to time constraints.

# Task 14 : Write the results to file
hourly_running_costs.to_csv(output_path + '/PowerPlantDispatch.csv')


# Steps for analysis -
# There could have been great insight through the plots of the generation2 and running margin with respect to that. from early analysis of the data, I think we could have
# Got good patterns about when it is feasiable to operate the power plant and when does it makes sense to keep it close.

# There is scope to run some machine learning algorithms as well, using features like date, time, month, year, fuel cost, plant name etc.
