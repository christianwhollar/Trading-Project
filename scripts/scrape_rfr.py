import requests
from bs4 import BeautifulSoup
import pandas as pd

def scrape_treasury_rates(year: int) -> pd.DataFrame:
    """
    Scrape treasury rates for a specific year from the U.S. Department of the Treasury website.

    Parameters:
    year (int): The year to scrape treasury rates for.

    Returns:
    pd.DataFrame: A DataFrame containing the treasury rates for the specified year.
    """
    url = f"https://home.treasury.gov/resource-center/data-chart-center/interest-rates/TextView?type=daily_treasury_bill_rates&field_tdr_date_value={year}"
    response = requests.get(url)
    response.raise_for_status()
    soup = BeautifulSoup(response.content, 'html.parser')
    table = soup.find('table', class_='views-table')
    rows = table.find_all('tr')
    headers = [header.text.strip() for header in rows[0].find_all('th')]
    data = [[ele.text.strip() for ele in row.find_all('td') if ele.text.strip()] for row in rows[1:]]
    df = pd.DataFrame(data, columns=headers)
    df.set_index('Date', inplace=True)
    return pd.DataFrame(df.iloc[:, 8])

def get_rfr_df(start_year: int, end_year: int) -> pd.DataFrame:
    """
    Get a DataFrame of risk-free rates for a range of years.

    Parameters:
    start_year (int): The start year of the range.
    end_year (int): The end year of the range.

    Returns:
    pd.DataFrame: A DataFrame containing the risk-free rates for the specified range of years.
    """
    output_dfs = [scrape_treasury_rates(year) for year in range(start_year, end_year - 1, -1)]
    output_df = pd.concat(output_dfs)
    output_df.index = pd.to_datetime(output_df.index)
    return output_df