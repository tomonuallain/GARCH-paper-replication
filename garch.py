#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 18 19:25:10 2023

@author: tomonuallain
"""

import yfinance as yf
import datetime
import pandas as pd
import numpy as np

from arch import arch_model
import matplotlib.pyplot as plt
import warnings
from datetime import datetime as dt
from tqdm import tqdm

warnings.filterwarnings('ignore')


end = '2023-11-08' #set last date for analysis

# Set up the phases, 1 and 2 as in paper, and phase 3 is out of sample
phase1 = ['1996-01-02', '2003-09-19']
phase2 = ['2003-09-22', '2012-01-31']
phase3 = ['2012-02-01', end]



def dates_to_indices(start_str, end_str, spx):
    """
    Parameters
    ----------
    start_str : str
        start date in string format.
    end_str : str
        end date in string format.
    spx : DataFrame
        a DataFrame object of S&P 500 returns for full period.

    Returns
    -------
    index_start : int
        integer index location of start date.
    index_end : int
        integer index location of end date.
    """
    formatted_start = pd.to_datetime(start_str)
    formatted_end = pd.to_datetime(end_str)
    index_start = spx.index.get_loc(formatted_start)
    index_end = spx.index.get_loc(formatted_end)
    
    return index_start, index_end

    


def gather_data(end):
    """
    Parameters
    ----------
    end : str
        last date to gather data for, should be after end date of analysis.

    Returns
    -------
    vix : DataFrame
        VIX data from 1980 to end date.
    spx : DataFrame
        S&P 500 data from 1980 to end date (log returns).
    """

    data = yf.download("^VIX ^SPX",
                       start="1980-01-01", end=pd.to_datetime('2023-11-28'))


    data = data['Adj Close'].copy()
    data.rename(columns={'^VIX': 'VIX', '^SPX': 'SPX'}, inplace=True)

    vix = data[['VIX']]
    spx = data[['SPX']]

    spx = np.log(spx/spx.shift(1))
    spx.dropna(inplace=True)
    return vix, spx


vix, spx = gather_data(end)


class GARCH11Model:
    def __init__(self):

        self.model_name = "GARCH11"

    def get_persistency(self, fit):
        """
        Parameters
        ----------
        fit : GARCH model object
            the GARCH model which you fitted.

        Returns
        -------
        int
            the value for xi for that particular time point.
        """
        self.alpha = fit.params[['alpha[1]']][0]
        self.beta = fit.params[['beta[1]']][0]
        return self.alpha + self.beta

    def get_c(self, xi):
        """
        Parameters
        ----------
        xi : int
            the persistency.

        Returns
        -------
        c : int
            the value for c, as described in the paper.
        """
        c = (1 - ((105/365) * (xi**20)) - (260/365) * (xi**21)) / (
            30 * (1 - xi))
        return c

    def get_Vl(self, fit, xi):
        """
        Parameters
        ----------
        fit : GARCH model object
            the GARCH model which you fitted.
        xi : int
            the value for xi (persistency) for that particular time point.

        Returns
        -------
        Vl : int
            the value for Vl, as outlined in the paper.
        """
        self.omega = fit.params[['omega']][0]
        return self.omega / (1 - xi)

    def get_d(self, Vl, c):
        """
        Parameters
        ----------
        Vl : int
            the value for Vl, as outlined in the paper.
        c : int
            the value for c, as described in the paper.

        Returns
        -------
        d : int
            the value for d, as outlined in the paper.
        """
        return Vl * (252/365 - c)

    def get_evix2(self, fit, c, d):  
        """
        Parameters
        ----------
        fit : GARCH model object
            the GARCH model which you fitted.
        c : int
            the value for c, as described in the paper.
        d : int
            the value for d, as described in the paper.
    
        Returns
        -------
        evix2 : int
            the value for eVIX^2, using formula in the paper.
        """
        v_t_1 = float(fit.forecast(horizon=1, reindex=False).variance.values)
        evix2 = ((c * v_t_1) + d) * 365 * (100**2)
        return evix2

    def get_evix(self, fit):  # calcs all parameters and gets eVIX
        """
        Parameters
        ----------
        fit : GARCH model object
            the GARCH model which you fitted.
        Returns
        -------
        evix : int
            calculates all the parameters and returns the value for eVIX, as 
            outlined in the paper.
        """
        self.xi = self.get_persistency(fit)
        self.c = self.get_c(self.xi)
        self.Vl = self.get_Vl(fit, self.xi)
        self.d = self.get_d(self.Vl, self.c)
        
        # divide by 100^2 (quadratic) here as we scaled data up previously
        evix2 = self.get_evix2(fit, self.c, self.d) / 100 ** 2
        return np.sqrt(evix2)

    def get_evix_at_t(self, t, vix, spx, evix_dict):
        """
        Parameters
        ----------
        t : int
            time index to calculate eVIX (t+1) at.
        vix : DataFrame
            a DataFrame object of VIX for full period.
        spx : DataFrame
            a DataFrame object of S&P 500 returns for full period.
        evix_dict : Dictionary
            a Dictionary to store the eVIX at each time point
        -------
        fits the GARCH model at t
        calculates and stores all the eVIX values for t+1 in a dictionary
        """
        # S&P x 100 so the optimser doesn't break, easier with larger numbers
        garch_model = arch_model((spx*100).iloc[t-3500:t],
                                 vol='GARCH',
                                 p=1, q=1)
        self.fit = garch_model.fit(disp='off')
        evix = self.get_evix(self.fit)
        evix_dict[t] = evix

    def get_range_of_evix_vix(self, start_str, end_str, vix, spx, plot=False):
        """
        Parameters
        ----------
        start_str : str
            start date in string format.
        end_str : str
            end date in string format.
        vix : DataFrame
            a DataFrame object of VIX for full period.
        spx : DataFrame
            a DataFrame object of S&P 500 returns for full period.
        plot : bool, optional
            whether or not you want to plot results. The default is False.

        Returns
        -------
        evix_vix : DataFrame 
            DataFrame of the eVIX and actual VIX over relevant period.
        """
        
        index_start, index_end = dates_to_indices(start_str, end_str, spx)
        
        evix_dict = {}
        
        for t in tqdm(range(index_start, index_end)):  # adds progress bar
            self.get_evix_at_t(t, vix, spx, evix_dict)
        
        evix = pd.DataFrame.from_dict(evix_dict,
                                      orient='index',
                                      columns=['eVIX'])
        
        evix.index = spx.index[evix.index]
        evix_vix = pd.concat([evix, vix], axis=1, join='inner')

        if plot == True:
            evix_vix.plot(figsize=(12, 8),
                          title=f"Plot of eVIX and VIX for {self.model_name} Model ({start_str} to {end_str})",
                          ylabel="%")
            plt.tight_layout()
            plt.show()

        return evix_vix



class GJRModel:
    def __init__(self):
        self.model_name = "GJR"

    def get_persistency(self, fit):
        """
        Parameters
        ----------
        fit : GARCH model fit object
            the GARCH model which you fitted.

        Returns
        -------
        int
            the value for xi for that particular time point.
        """
        self.alpha = fit.params[['alpha[1]']][0]
        self.beta = fit.params[['beta[1]']][0]
        self.gamma = fit.params[['gamma[1]']][0]
        return self.alpha + self.beta + 0.5*self.gamma

    def get_c(self, xi):
        """
        Parameters
        ----------
        xi : int
            persistency.

        Returns
        -------
        c : int
            the value for c, as described in the paper.
        """
        self.c = (1 - ((105/365) * (xi**20)) - (260/365)
                  * (xi**21)) / (30 * (1 - xi))
        return self.c

    def get_Vl(self, fit, xi):
        """
        Parameters
        ----------
        fit : GARCH model object
            the GARCH model which you fitted.
        xi : int
            the value for xi (persistency) for that particular time point.

        Returns
        -------
        Vl : int
            the value for Vl, as outlined in the paper.
        """
        self.omega = fit.params[['omega']][0]
        return self.omega / (1 - xi)

    def get_d(self, Vl, c):
        """
        Parameters
        ----------
        Vl : int
            the value for Vl, as outlined in the paper.
        c : int
            the value for c, as described in the paper.

        Returns
        -------
        d : int
            the value for d, as outlined in the paper.
        """
        return Vl * (252/365 - c)

    def get_evix2(self, fit, c, d):  # uses formula for eVIX^2
        """
        Parameters
        ----------
        fit : GARCH model object
            the GARCH model which you fitted.
        c : int
            the value for c, as described in the paper.
        d : int
            the value for d, as described in the paper.
        
        Returns
        -------
        evix2 : int
            the value for eVIX^2, as outlined in the paper.
        
        """
        v_t_1 = float(fit.forecast(horizon=1, reindex=False).variance.values)
        evix2 = ((c * v_t_1) + d) * 365 * (100**2)
        return evix2

    def get_evix(self, fit):  # calcs all parameters and gets eVIX
        """
        Parameters
        ----------
        fit : GARCH model object
            the GARCH model which you fitted.
        Returns
        -------
        evix : int
            calculates all the parameters and returns the value for eVIX, as 
            outlined in the paper.
        """
        self.xi = self.get_persistency(fit)
        self.c = self.get_c(self.xi)
        self.Vl = self.get_Vl(fit, self.xi)
        self.d = self.get_d(self.Vl, self.c)
        # divide by 100^2 (quadratic) here as we scaled data up previously
        evix2 = self.get_evix2(fit, self.c, self.d) / 100 ** 2
        return np.sqrt(evix2)

    def get_evix_at_t(self, t, vix, spx, evix_dict):
        """
        Parameters
        ----------
        t : int
            time index to calculate eVIX (t+1) at.
        vix : DataFrame
            a DataFrame object of VIX for full period.
        spx : DataFrame
            a DataFrame object of S&P 500 returns for full period.
        evix_dict : Dictionary
            a Dictionary to store the eVIX at each time point
        -------
        fits the GARCH model at t
        calculates and stores all the eVIX values for t+1 in a dictionary
        """
        # S&P x 100 so the optimser doesn't break, easier with larger numbers
        garch_model = arch_model((spx*100).iloc[t-3500:t],  
                                 p=1, o=1, q=1)
        fit = garch_model.fit(disp='off')
        evix = self.get_evix(fit)
        evix_dict[t] = evix

    def get_range_of_evix_vix(self, start_str, end_str, vix, spx, plot=False):
        """
        Parameters
        ----------
        start_str : str
            start date in string format.
        end_str : str
            end date in string format.
        vix : DataFrame
            a DataFrame object of VIX for full period.
        spx : DataFrame
            a DataFrame object of S&P 500 returns for full period.
        plot : bool, optional
            whether or not you want to plot results. The default is False.

        Returns
        -------
        evix_vix : DataFrame 
            DataFrame of the eVIX and actual VIX over relevant period.
        """
        index_start, index_end = dates_to_indices(start_str, end_str, spx)

        evix_dict = {}
        
        for t in tqdm(range(index_start, index_end)):  # adds progress bar
            self.get_evix_at_t(t, vix, spx, evix_dict)
        evix = pd.DataFrame.from_dict(evix_dict,
                                      orient='index',
                                      columns=['eVIX'])
        evix.index = spx.index[evix.index]
        evix_vix = pd.concat([evix, vix], axis=1, join='inner')

        if plot == True:
            evix_vix.plot(figsize=(12, 8),
                          title=f"Plot of eVIX and VIX for {self.model_name} Model ({start_str} to {end_str})",
                          ylabel="%")
            plt.tight_layout()
            plt.show()

        return evix_vix

# ERRORS
def plot_forecast_error_over_time(model, df, actual_col, forecast_col):
    """
    Plot the forecast error (as a percentage) over time from a DataFrame.

    Parameters:
    - df: DataFrame containing columns of actual and forecasted values over time.
    - actual_col: Name of the column containing actual values.
    - forecast_col: Name of the column containing forecasted values.
    - title: Title of the plot (default is "Forecast Error Over Time").

    Returns:
    - None (displays the plot).
    """
    # calculate forecast error as a percentage
    error_percentage = ((df[forecast_col] - df[actual_col]) / df[actual_col]) * 100

    # plotting
    plt.figure(figsize=(10, 6))
    plt.plot(error_percentage, marker=None, linestyle='-', linewidth=0.8, color='k', label='Forecast Error (%)')
    plt.axhline(color='r', linestyle='--', zorder=-1)
    plt.axvspan(phase3[0], phase3[1], alpha=1, color='lightgrey', zorder=-2, label='Out of Sample')
    plt.title(f"{model.model_name} Forecasting Error (%)")
    plt.xlabel('Time')
    plt.ylabel('Forecast Error (%)')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.grid(False)
    plt.savefig(f"{model.model_name}_forecast_errors.png", dpi=300, bbox_inches='tight')
    plt.show()

def calculate_mfe(df, actual_col, forecast_col):
    """
    Calculate Mean Forecast Error (MFE) as a percentage.

    Parameters:
    - df: DataFrame containing actual and forecast columns.
    - actual_col: Name of the column containing actual values.
    - forecast_col: Name of the column containing forecasted values.

    Returns:
    - MFE as a percentage.
    """
    actual = df[actual_col]
    forecast = df[forecast_col]
    mfe = np.mean(forecast - actual)
    mfe_percentage = (mfe / np.mean(actual)) * 100
    return mfe_percentage

def calculate_mae(df, actual_col, forecast_col):
    """
    Calculate Mean Absolute Error (MAE) as a percentage.

    Parameters:
    - df: DataFrame containing actual and forecast columns.
    - actual_col: Name of the column containing actual values.
    - forecast_col: Name of the column containing forecasted values.

    Returns:
    - MAE as a percentage.
    """
    actual = df[actual_col]
    forecast = df[forecast_col]
    mae = np.mean(np.abs(forecast - actual))
    mae_percentage = (mae / np.mean(actual)) * 100
    return mae_percentage

def calculate_rmse(df, actual_col, forecast_col):
    """
    Calculate Root Mean Squared Error (RMSE) as a percentage.

    Parameters:
    - df: DataFrame containing actual and forecast columns.
    - actual_col: Name of the column containing actual values.
    - forecast_col: Name of the column containing forecasted values.

    Returns:
    - RMSE as a percentage.
    """
    actual = df[actual_col]
    forecast = df[forecast_col]
    rmse = np.sqrt(np.mean((actual - forecast)**2))

    return rmse

def calculate_error_metrics(model, df, phase, actual_col, forecast_col):
    """
    Parameters
    - model: some GARCH model object (G11 or GJR)
    - df: DataFrame containing actual and forecast columns.
    - actual_col: Name of the column containing actual values.
    - forecast_col: Name of the column containing forecasted values.
    
    Returns:
    - a DataFrame of each error metric for the relevant phase
    """
    error_dict = {}
    error_dict['MFE (%)'] = calculate_mfe(df, actual_col, forecast_col)
    error_dict['MAE (%)'] = calculate_mae(df, actual_col, forecast_col)
    error_dict['RMSE'] = calculate_rmse(df, actual_col, forecast_col)

    print(
        f"""The {model.model_name} Model has the following error metrics for {phase}:
          MFE:\t{error_dict['MFE (%)']:.2f}
          MAE:\t{error_dict['MAE (%)']:.2f}
          RMSE:\t{error_dict['RMSE']:.2f}

          """)
    error_df = pd.DataFrame(error_dict, index=[model.model_name+phase]) 
    return error_df
        

# Now run the code using the following procedure:
    
    # 1. Create model object for your model (G11 or GJR)
    # 2. Calculate the eVIX and add to DataFrame with actual VIX for each phase
    # 3. Calculate the errors in each phase
    # 4. Plot the forecast error at each step over time if you like
    # 5. Save the errors to a .csv for presentation in poster

# G11 MODEL
g11 = GARCH11Model()
g11_df_phase1 = g11.get_range_of_evix_vix(phase1[0], phase1[1],
                                          vix, spx,
                                          plot=True)
g11_df_phase2 = g11.get_range_of_evix_vix(phase2[0], phase2[1],
                                          vix, spx,
                                          plot=True)
g11_df_phase3 = g11.get_range_of_evix_vix(phase3[0], phase3[1],
                                          vix, spx,
                                          plot=True)
g11_df_full = pd.concat([g11_df_phase1, g11_df_phase2, g11_df_phase3])

g11_phase1_errors = calculate_error_metrics(g11, g11_df_phase1, 'Phase 1', "VIX", "eVIX")
g11_phase2_errors = calculate_error_metrics(g11, g11_df_phase2, 'Phase 2', "VIX", "eVIX")
g11_phase3_errors = calculate_error_metrics(g11, g11_df_phase3, 'Phase 3', "VIX", "eVIX")
g11_errors = pd.concat([g11_phase1_errors, g11_phase2_errors, g11_phase3_errors])
# plot_forecast_error_over_time(g11, g11_df_full, "VIX", "eVIX")


# GJR MODEL
gjr = GJRModel()
gjr_df_phase1 = gjr.get_range_of_evix_vix(phase1[0], phase1[1],
                                          vix, spx,
                                        plot=True)  
gjr_df_phase2 = gjr.get_range_of_evix_vix(phase2[0], phase2[1],
                                          vix, spx,
                                          plot=True)  
gjr_df_phase3 = gjr.get_range_of_evix_vix(phase3[0], phase3[1],
                                          vix, spx,
                                          plot=True)  
gjr_df_full = pd.concat([gjr_df_phase1, gjr_df_phase2, gjr_df_phase3])

gjr_phase1_errors = calculate_error_metrics(gjr, gjr_df_phase1, 'Phase 1', "VIX", "eVIX")
gjr_phase2_errors = calculate_error_metrics(gjr, gjr_df_phase2, 'Phase 2', "VIX", "eVIX")
gjr_phase3_errors = calculate_error_metrics(gjr, gjr_df_phase3, 'Phase 3', "VIX", "eVIX")
gjr_errors = pd.concat([gjr_phase1_errors, gjr_phase2_errors, gjr_phase3_errors])
plot_forecast_error_over_time(gjr, gjr_df_full, "VIX", "eVIX")


all_errors = pd.concat([g11_errors, gjr_errors])
all_errors.to_csv('all_errors.csv')

spx_to_plot = spx.iloc[544:]
spx_to_plot.to_csv('spx_data.csv')


