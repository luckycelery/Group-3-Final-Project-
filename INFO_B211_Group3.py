import os
import pandas as pd
import numpy as np

# ============================
# SCIPY (STATISTICS + MODELING) IMPORTS
# ============================
from scipy.stats import linregress, ttest_ind
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit

# ============================
# SCIKIT-LEARN (ML) IMPORTS
# ============================
from sklearn import (
    preprocessing,
    model_selection,
    metrics,
    ensemble,
    linear_model,
    neighbors,
    tree
)

# ============================
# VISUALIZATION IMPORTS
# ============================
import matplotlib.pyplot as plt
import seaborn as sns

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
# DATA CLEANING AND PREPARATION
# =========================================================

def clean_data(df):
    """
    Cleans the raw mortality dataset by standardizing column names,
    filtering to mortality-related indicators, removing unusable rows,
    and converting year columns to numeric types.

    Returns a cleaned dataframe ready for analysis.
    """

    # Standardize key column names for consistency
    df = df.rename(columns={
        "Series Name": "series",
        "Country Name": "country"
    })

    # Remove rows missing a country name (cannot be used in analysis)
    df = df.dropna(subset=["country"])

    # Inspect all unique series names before filtering
    print(df["series"].unique())
    pd.Series(df["series"].unique()).to_csv("unique_mortality_types.csv", index=False)

    # Keep only rows where the series name indicates a mortality-related metric
    mortality_df = df.loc[
        df["series"].str.contains("Mortality rate|Death rate|Mortality", case=False)
    ]

    # Inspect unique series names after filtering
    print(mortality_df["series"].unique())
    pd.Series(mortality_df["series"].unique()).to_csv("post_filter_mortality_types.csv", index=False)

    # Preview the filtered dataset
    print(mortality_df.head())

    # Remove rows missing country (extra safety check)
    mortality_df = mortality_df.dropna(subset=["country"])

    # Remove rows missing *all* year values (no usable time series)
    year_columns = [str(y) for y in range(2000, 2024)]
    mortality_df = mortality_df.dropna(subset=year_columns)

    # Preview after dropping incomplete rows
    print(mortality_df.head())

    # Save cleaned dataset for debugging and transparency
    mortality_df.to_csv("cleaned_mortality_data.csv", index=False)
    mortality_df.to_excel("cleaned_mortality_data.xlsx", index=False)

    # Convert year columns to numeric (ensures compatibility with regression)
    mortality_df[year_columns] = mortality_df[year_columns].apply(
        pd.to_numeric, errors="coerce"
    )

    return mortality_df


# =========================================================
# IN‑DEPTH DATA ANALYSIS (SCIPY)
# =========================================================
def analyze_trends_linreg(df, output_path="trend_coefficients.csv"):
    """
    Computes linear regression trend statistics for each (country, series) pair.

    This function extracts year columns (2000–2023), converts them to numeric,
    and fits a simple linear regression model using SciPy's linregress.
    It returns slope, intercept, r-squared, p-value, and standard error for
    each mortality indicator within each country.

    Args:
        df (pd.DataFrame): Cleaned mortality dataframe containing:
            - 'country'
            - 'series'
            - year columns 2000–2023
        output_path (str): File path for saving the regression summary CSV.

    Returns:
        pd.DataFrame: A dataframe containing regression coefficients and
                      statistical metrics for each country × series.
    """

    # Make acopy so we never modify the original dataframe.
    # This protects upstream functions and ensures reproducibility.
    df = df.copy()

    # ---------------------------------------------------------------
    # Identify all year columns (2000–2023) and ensure they are numeric.
    # Converts invalid entries to NaN so regression won't break.
    # ---------------------------------------------------------------
    year_cols = [str(y) for y in range(2000, 2024)]
    df[year_cols] = df[year_cols].apply(pd.to_numeric, errors="coerce")

    # Convert year labels to integers so SciPy can treat them as numeric
    # predictor variables in the regression model.
    years = np.array([int(y) for y in year_cols])

    # This list will accumulate one dictionary per (country, series) pair.
    results = []

    # ---------------------------------------------------------------
    # Group the dataset by country AND series.
    # Each group represents one mortality indicator for one country.
    # Example:
    #   ("Afghanistan", "Infant mortality") → one regression
    # ---------------------------------------------------------------
    for (country, series), group in df.groupby(["country", "series"]):

        # Get the row of year values for this specific indicator.
        # group[year_cols] returns a dataframe; .iloc[0] selects the row.
        values = group[year_cols].iloc[0].values.astype(float)

        # -----------------------------------------------------------
        # Skip this if:
        #   - all values are NaN (no data reported)
        #   - the series is constant (std = 0 → regression undefined)
        # This prevents SciPy from throwing errors.
        # -----------------------------------------------------------
        if np.isnan(values).all() or np.std(values) == 0:
            continue

        # -----------------------------------------------------------
        # Perform linear regression:
        #   slope      → yearly change in mortality
        #   intercept  → model's baseline value
        #   r_value    → correlation coefficient
        #   p_value    → significance of slope
        #   stderr     → uncertainty of slope estimate
        # -----------------------------------------------------------
        slope, intercept, r_value, p_value, stderr = linregress(years, values)

        # Store all regression statistics in a structured dictionary.
        # This will later become one row in the output CSV.
        results.append({
            "country": country,
            "series": series,
            "slope": slope,
            "intercept": intercept,
            "r_squared": r_value**2,  # convert correlation to r²
            "p_value": p_value,
            "stderr": stderr
        })

    # Convert the list of dictionaries into a dataframe.
    results_df = pd.DataFrame(results)

    # Save the regression summary to CSV for downstream analysis.
    results_df.to_csv(output_path, index=False)

    # Return the dataframe
    return results_df


def generate_linear_trendlines(df, output_path="trendline_predictions.csv"):
    """
    Generates predicted linear trendline values for each (country, series) pair.

    This function uses the same linear regression approach as
    `analyze_trends_linreg()`, but instead of returning only the regression
    coefficients (slope, intercept, r-squared, etc.), it produces a *long-format*
    dataset containing the predicted mortality value for every year (2000–2023).

    This output is ideal for visualization in Seaborn or Matplotlib, since it
    provides a smooth trendline for each country × series that can be plotted
    directly.

    How this differs from analyze_trends_linreg():
    - analyze_trends_linreg() → returns regression *statistics* (slope, intercept)
    - generate_linear_trendlines() → returns regression *predictions* for each year
    - This function is specifically designed for visualization, not analysis.

    Args:
        df (pd.DataFrame): Cleaned mortality dataframe containing:
            - 'country'
            - 'series'
            - year columns 2000–2023
        output_path (str): File path for saving the trendline prediction CSV.

    Returns:
        pd.DataFrame: Long-format dataframe with predicted values for each year.
    """

    # Make a copy so the original dataframe remains unchanged.
    df = df.copy()

    # Identify all year columns and ensure they are numeric.
    # This is the same preprocessing step as in analyze_trends_linreg().
    year_cols = [str(y) for y in range(2000, 2024)]
    df[year_cols] = df[year_cols].apply(pd.to_numeric, errors="coerce")

    # Convert year labels to integers so they can be used as the
    # predictor variable in the regression model.
    years = np.array([int(y) for y in year_cols])

    # This list will accumulate one row per predicted value.
    # Example row:
    #   {"country": "Afghanistan", "series": "Infant mortality",
    #    "year": 2005, "predicted_value": 92.1}
    results = []

    # ---------------------------------------------------------------
    # Loop through each country × series combination.
    # Each group represents one mortality indicator for one country.
    # ---------------------------------------------------------------
    for (country, series), group in df.groupby(["country", "series"]):

        # Extract the time series values for this indicator.
        # group[year_cols] returns a dataframe; .iloc[0] selects the row.
        y = group[year_cols].iloc[0].values.astype(float)

        # -----------------------------------------------------------
        # Skip this if:
        #   - all values are NaN (no data reported)
        #   - the series is constant (no trend to model)
        # -----------------------------------------------------------
        if np.isnan(y).all() or np.std(y) == 0:
            continue

        # -----------------------------------------------------------
        # Fit a simple linear regression model:
        #   slope      → yearly change in mortality
        #   intercept  → baseline value
        #   r_value    → correlation coefficient
        #   p_value    → significance of slope
        #   stderr     → uncertainty of slope estimate
        #
        # Only slope and intercept are needed for prediction.
        # -----------------------------------------------------------
        slope, intercept, r_value, p_value, stderr = linregress(years, y)

        # -----------------------------------------------------------
        # Use the regression model to generate predicted values for
        # *every* year in the dataset. This creates a smooth trendline
        # that can be plotted directly in Seaborn.
        # -----------------------------------------------------------
        for year in years:
            pred = slope * year + intercept

            # Store one row per predicted value (long format)
            results.append({
                "country": country,
                "series": series,
                "year": year,
                "predicted_value": pred
            })

    # Convert accumulated predictions into a dataframe
    results_df = pd.DataFrame(results)

    # Save the long-format trendline predictions to CSV
    results_df.to_csv(output_path, index=False)

    # Return the dataframe
    return results_df

def pairwise_ttests(df, output_path="pairwise_ttests.csv"):
    """
    Performs pairwise independent t-tests comparing every country's mortality
    time series against every other country's time series.

    Each comparison uses the full set of year columns (2000–2023) and computes
    whether the average mortality levels differ significantly between countries.

    Args:
        df (pd.DataFrame): Cleaned mortality dataframe containing:
            - 'country'
            - year columns 2000–2023
        output_path (str): File path for saving the t-test results.

    Returns:
        pd.DataFrame: A dataframe containing t-statistics and p-values for each
                      country-to-country comparison.
    """

    # Identify the year columns to extract comparable numeric time series
    year_cols = [str(y) for y in range(2000, 2024)]

    # Get the list of unique countries to compare
    countries = df["country"].unique()

    results = []

    # ---------------------------------------------------------------
    # Loop through all unique country pairs WITHOUT repeating pairs.
    # Example: if we compare (A vs B), we do NOT later compare (B vs A).
    # ---------------------------------------------------------------
    for i, c1 in enumerate(countries):
        for c2 in countries[i+1:]:

            # Extract the full time series for each country
            # .values.flatten() converts the row into a 1D numeric array
            series1 = df[df["country"] == c1][year_cols].values.flatten()
            series2 = df[df["country"] == c2][year_cols].values.flatten()

            # -----------------------------------------------------------
            # Perform an independent t-test:
            #   t_stat → magnitude + direction of difference
            #   p_val  → significance of difference
            #
            # nan_policy="omit" ensures missing values do not break the test.
            # -----------------------------------------------------------
            t_stat, p_val = ttest_ind(series1, series2, nan_policy="omit")

            # Store results for this country pair
            results.append({
                "country_1": c1,
                "country_2": c2,
                "t_statistic": t_stat,
                "p_value": p_val
            })

    # Convert results to a dataframe and save to CSV
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_path, index=False)
    return results_df


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

    # === SCIPY ANALYSIS CALLS ===
    linreg_results = analyze_trends_linreg(df)
    trendline_predictions = generate_linear_trendlines(df)
    ttest_results = pairwise_ttests(df)

    print(df.head())
    sklearn_mortality_model_results = run_predictive_modeling(df)
