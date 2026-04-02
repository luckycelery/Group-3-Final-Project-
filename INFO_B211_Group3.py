import os
import pandas as pd
import numpy as np
from scipy import stats, interpolate, integrate # WE CAN ADD MORE AS WE GO THESE ARE JUST SO WE DON'T HAVE TO TYPE AS MUCH LATER
from sklearn import preprocessing, model_selection, metrics, ensemble, linear_model, neighbors, tree # WE CAN ADD MORE AS WE GO THESE ARE JUST SO WE DON'T HAVE TO TYPE AS MUCH LATER

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
    return mortality_df


# =========================================================
# EXPLORATORY DATA ANALYSIS
# =========================================================



# =========================================================
# IN‑DEPTH DATA ANALYSIS (SCIPY)
# =========================================================



# =========================================================
# PREDICTIVE MODELING (SCIKIT‑LEARN)
# =========================================================
def run_predictive_modeling(df):
    """
    Runs simple predictive models for each mortality indicator.
    """
    #List of all mortality indicators in model
    mortality_indicators = [
        'Death rate, crude (per 1,000 people)',
        'Maternal mortality ratio (modeled estimate, per 100,000 live births)',
        'Maternal mortality ratio (national estimate, per 100,000 live births)',
        'Mortality caused by road traffic injury (per 100,000 population)',
        'Mortality from CVD, cancer, diabetes or CRD between exact ages 30 and 70 (%)',
        'Mortality from CVD, cancer, diabetes or CRD between exact ages 30 and 70, female (%)',
        'Mortality from CVD, cancer, diabetes or CRD between exact ages 30 and 70, male (%)',
        'Mortality rate attributed to unintentional poisoning (per 100,000 population)',
        'Mortality rate attributed to unintentional poisoning, female (per 100,000 female population)',
        'Mortality rate attributed to unintentional poisoning, male (per 100,000 male population)',
        'Mortality rate, adult, female (per 1,000 female adults)',
        'Mortality rate, adult, male (per 1,000 male adults)',
        'Mortality rate, infant (per 1,000 live births)',
        'Mortality rate, infant, female (per 1,000 live births)',
        'Mortality rate, infant, male (per 1,000 live births)',
        'Mortality rate, neonatal (per 1,000 live births)',
        'Mortality rate, under-5 (per 1,000 live births)',
        'Mortality rate, under-5, female (per 1,000 live births)',
        'Mortality rate, under-5, male (per 1,000 live births)',
        'Suicide mortality rate (per 100,000 population)',
        'Suicide mortality rate, female (per 100,000 female population)',
        'Suicide mortality rate, male (per 100,000 male population)'
    ]

    #List for model results
    results = []

    # Loop through each mortality indicator
    for indicator in mortality_indicators:

        print(f"\nProcessing indicator: {indicator}")

        # Filter the dataset to only this mortality type
        mort = df.loc[df["series"] == indicator].copy()

        # If the indicator is not present, skip it
        if mort.empty:
            print("  → Not found in dataset, skipping.")
            continue

        rows = []

        for year in range(2000, 2024):  # 2000–2023
            year_str = str(year)

            # Extract country + that year's value
            temp = mort[["country", year_str]].copy()

            # Rename the year column to a generic name
            temp = temp.rename(columns={year_str: "mortality_rate"})

            # Add the year as its own column
            temp["year"] = year

            # Store this year's data
            rows.append(temp)

        # Combine all years into one long dataframe
        mort_long = pd.concat(rows, ignore_index=True)

        # Convert mortality_rate to numeric (some values may be strings)
        mort_long["mortality_rate"] = pd.to_numeric(mort_long["mortality_rate"], errors="coerce")

        # Remove missing values
        mort_long = mort_long.dropna(subset=["mortality_rate"])

        # Use "year" as the predictor and "mortality_rate" as the target variable
        X = mort_long[["year"]]
        y = mort_long["mortality_rate"]

        # Train/test split
        x_train, x_test, y_train, y_test = model_selection.train_test_split(
            X, y, train_size=0.8, random_state=20
        )

        # Linear Regression Model
        lin_model = linear_model.LinearRegression()
        lin_model.fit(x_train, y_train)
        lin_preds = lin_model.predict(x_test)

        # Random Forest Regressor Model
        rf_model = ensemble.RandomForestRegressor(
            n_estimators=200,
            criterion="squared_error",
            max_features="sqrt",
            random_state=42
        )
        rf_model.fit(x_train, y_train)
        rf_preds = rf_model.predict(x_test)

       # Store results for this indicator
        results.append({
            "indicator": indicator,
            "linear_r2": metrics.r2_score(y_test, lin_preds),
            "linear_mae": metrics.mean_absolute_error(y_test, lin_preds),
            "rf_r2": metrics.r2_score(y_test, rf_preds),
            "rf_mae": metrics.mean_absolute_error(y_test, rf_preds),
            "rf_explained_variance": metrics.explained_variance_score(y_test, rf_preds)
        })

    # Covert results to a dataframe
    results_df = pd.DataFrame(results)
    sklearn_mortality_model_results = results_df

    print("\n\n=== MODELING SUMMARY ===")
    print(sklearn_mortality_model_results)

    # Save results to CSV
    sklearn_mortality_model_results.to_csv("sklearn_mortality_model_results.csv", index=False)
    return sklearn_mortality_model_results
    


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
    sklearn_mortality_model_results = run_predictive_modeling(df)