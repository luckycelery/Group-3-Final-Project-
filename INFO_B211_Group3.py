import os
import pandas as pd
import numpy as np
from scipy import stats, interpolate, integrate # WE CAN ADD MORE AS WE GO THESE ARE JUST SO WE DON'T HAVE TO TYPE AS MUCH LATER
from sklearn import preprocessing, model_selection, metrics, ensemble, linear_model, neighbors, tree # WE CAN ADD MORE AS WE GO THESE ARE JUST SO WE DON'T HAVE TO TYPE AS MUCH LATER
import matplotlib.pyplot as plt
import seaborn as sns
 
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
    
    df_head = df.head()
    df_info = df.info()
    df_desc = df.describe()
    df_shape = df.shape

    return df, df_head, df_info, df_desc, df_shape


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
    #Remove missing values 
    df = df.rename(columns={
        "Series Name": "series",
        "Country Name": "country"
    })

    df = df.dropna(subset=["country"])

    #Unique Mortality Types 
    print(df["series"].unique())
    #output of post mortality types unique values before filtering for mortality related series
    pd.Series(df["series"].unique()).to_csv("unique_mortality_types.csv", index=False)

    #Filter the dataset to include only mortality-related series
    mortality_df = df.loc[
        df["series"].str.contains("Mortality rate|Death rate|Mortality", case=False)]
    print(mortality_df["series"].unique())
    # output of post filter mortality types unique values before dropping rows with missing country or year data
    pd.Series(mortality_df["series"].unique()).to_csv("post_filter_mortality_types.csv", index=False)

    print(mortality_df.head())

    #Drop rows missing country
    mortality_df = mortality_df.dropna(subset=["country"])

    #Drop rows missing all year data
    year_columns = [str(y) for y in range(2000, 2024)]
    mortality_df = mortality_df.dropna(subset=year_columns)
    print(mortality_df.head())

    #Saved cleaned dataset 
    mortality_df.to_csv("cleaned_mortality_data.csv", index=False)
    mortality_df.to_excel("cleaned_mortality_data.xlsx", index=False)
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
    df, df_head, df_info, df_desc, df_shape = load_data(script_dir)
    df = clean_data(df)

    print(df.head())
