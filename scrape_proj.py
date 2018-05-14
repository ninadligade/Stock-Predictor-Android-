import datetime as dt
from iexfinance import get_historical_data


def printHistoricalData(df):
    '''
    The function printHistoricalData takes a dataframe as an input and print the data
    df - The dataframe
    '''
    print(df)


def getStockHistoricalData(stock, start_date, end_date):
    '''
    The function getStockHistoricalData takes in following parameters
    stock - Name of the company
    start_date - The start date of the historical data
    end_date - The end date of the historical data
    '''


    df = get_historical_data(stock, start=start_date, end=end_date, output_format='pandas')
    df.to_csv('out.csv', sep='\t', encoding='utf-8')
    #printHistoricalData(df)



def main():
    '''
    The main function uses a subroutine named getStockHistoricalData to fetch the financial data
    for a given time period
    '''
    print ("Enter the company tag name:")
    company = str(input(""))
    #print ("company ", company)
    date_entry_for_startDate = input('Enter a start date in YYYY-MM-DD format\n')
    year, month, day = map(int, date_entry_for_startDate.split('-'))
    start_date = dt.date(year, month, day)

    date_entry_for_endDate = input('Enter an end date in YYYY-MM-DD format\n')
    year, month, day = map(int, date_entry_for_endDate.split('-'))
    end_date = dt.date(year, month, day)

    getStockHistoricalData(company, start_date, end_date)


if __name__ == '__main__':
    main()