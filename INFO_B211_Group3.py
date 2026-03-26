import os
import pandas as pd
import numpy as np
from scipy import stats, interpolate, integrate # WE CAN ADD MORE AS WE GO THESE ARE JUST SO WE DON'T HAVE TO TYPE AS MUCH LATER
from sklearn import preprocessing, model_selection, metrics # WE CAN ADD MORE AS WE GO THESE ARE JUST SO WE DON'T HAVE TO TYPE AS MUCH LATER

# =========================================================
# TEMPLATE FOR FUNCTION DOCUMENTATION (FOR JOSIE)
# =========================================================

def example_function_description():
    """
    Brief description of what the brainstormed function will do.
    
    Notes for Josie:
    - Describe the purpose of the function.
    - Mention important arguments (args) and what the function returns 
      (LATER ON ONCE WE KNOW WHAT TO DO AND ALL THAT JAZZ)
    - Keep descriptions concise but clear.
    """
    pass

# example hypothetical function 
def country_mortality_comparison():
    """
    Compares mortality rates between countries.
    Inputs: hypothetical df, country list, metric selection.
    Returns: summary statistics or comparison table.
    """
    pass


# =========================================================
# DATA LOADING
# =========================================================

def load_data(script_dir):
    """
    Loads the raw dataset using dynamic pathing.

    Args:
        script_dir (str): Directory where this script is located.

    Returns:
        pd.DataFrame: Raw dataset loaded into a DataFrame.
    """
    # load data file with dynamic pathing
    raw_file = os.path.join(script_dir, "Data.csv")
    # read data file and return contents within a pd df
    df = pd.read_csv(raw_file)
    return df


# =========================================================
# DATA CLEANING (JULIA THIS IS WHERE YOU WILL PUT ALL YOUR FUNCS FOR CLEANING 
#                  AND THEN CALL IN THE MAIN — EXAMPLE IN CODE COMMENTS OF MAIN EXECUTION)
# =========================================================

def clean_data(df):
    """
    Placeholder for data cleaning steps.
    This is where we will:
    - handle missing values
    - standardize column names
    - filter out irrelevant rows
    - convert data types
    """
    # TODO: fill in once we inspect the dataset
    return df


# =========================================================
# EXPLORATORY DATA ANALYSIS
# =========================================================



# =========================================================
# IN‑DEPTH DATA ANALYSIS (SCIPY)
# =========================================================



# =========================================================
# PREDICTIVE MODELING (SCIKIT‑LEARN)
# =========================================================



# =========================================================
# DATA VISUALIZATION
# =========================================================



# =========================================================
# MAIN EXECUTION
# =========================================================

if __name__ == "__main__":
    # dynamic path setup for script dir
    script_dir = os.path.dirname(__file__)
    # call for data to be loaded into the program 
    df = load_data(script_dir)
    # df = clean_data(df)

    print(df.head())
