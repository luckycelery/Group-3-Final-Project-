# Group‑3‑Final‑Project  
INFO‑B211 Final Project — Demonstrate proficiency in utilizing Python for data analysis

This project performs end‑to‑end data cleaning, statistical analysis, predictive modeling, and visualization on global mortality datasets (2000–2023) and pulls relevant insights.
It uses **Pandas**, **NumPy**, **SciPy**, **Scikit‑Learn**, **Matplotlib**, and **Seaborn**, all managed through a shared Conda environment to ensure reproducibility across the team.

---

# Table of Contents
1. [Project Overview](#project-overview)  
2. [How to Run the Program](#how-to-run-the-program)  
   - [1. Set Up the Virtual Environment](#1-set-up-the-virtual-environment)  
   - [2. Running the Program](#2-running-the-program)  
3. [What Goes Where](#what-goes-where)  
4. [Why We Use a Single‑Script Structure](#why-we-use-a-single-script-structure)  
5. [Data Requirements](#data-requirements)  
6. [Project Structure](#project-structure)  
7. [Function Documentation](#function-documentation)  
   - Data Loading  
   - Data Cleaning  
   - SciPy Statistical Analysis  
   - Predictive Modeling  
   - Visualization  
8. [Outputs](#outputs)

---

# Project Overview
This project analyzes global mortality indicators from 2000–2023.  
It performs:

- Data loading and cleaning  
- Mortality indicator filtering  
- Linear regression trend analysis (SciPy)  
- Trendline prediction generation  
- Pairwise t‑tests between countries  
- Predictive modeling using Scikit‑Learn  
- Multi‑chart visualization of global and country‑specific mortality patterns  

All outputs are saved as CSVs and PNG charts for reporting and interpretation.

---

# How to Run the Program

## 1. Set Up the Virtual Environment
This project uses a shared Conda environment so all team members run the same versions of Python and required libraries.

### Steps
1. Install **Anaconda** or **Miniconda**.  
2. Open your terminal.  
3. Navigate to the project folder:

```
cd path/to/your/project/folder
```

4. Create the environment:

```
conda env create -f environment.yml
```

5. Activate the environment:

```
conda activate group3
```

6. If the environment file changes:

```
conda env update -f environment.yml --prune
```

7. If the environment breaks:

```
conda remove --name group3 --all
conda env create -f environment.yml
```

Once activated, you are ready to run the program.

---

## 2. Running the Program
After activating the environment, run the main script:

```
python INFO_B211_Group3.py
```

The program will:

- Load the dataset using dynamic pathing  
- Run cleaning functions  
- Perform statistical and predictive analysis  
- Generate visualizations  
- Save outputs to CSV and PNG files  

More detailed instructions will be added as the project grows.

---

# What Goes Where

### **INFO_B211_Group3.py**
Main script for the entire project.  
Contains all functions, analysis steps, and execution logic.

### **Data.csv & Series‑Metadata.csv**
Datasets used for analysis.  
Must remain in the root directory for dynamic pathing.

### **environment.yml**
Shared Conda environment file.  
Ensures consistent library versions across all machines.

### **README.md**
Documentation for setup, environment usage, and function explanations.

---

# Why We Use a Single‑Script Structure
- Several team members are newer to Python and Git.  
- Splitting into multiple modules increases merge conflict risk.  
- A single script keeps the workflow visible and easy to follow.  
- Easier to debug, maintain, and teach in a short‑term class project.

---

# Data Requirements
The script expects a file named:

```
Data.csv
```

containing:

- `Country Name`  
- `Series Name`  
- Year columns `2000`–`2023`  

---

# Project Structure

```
project/
│
├── Data.csv
├── cleaned_mortality_data.csv
├── cleaned_mortality_data.xlsx
├── trend_coefficients.csv
├── trendline_predictions.csv
├── pairwise_ttests.csv
├── sklearn_mortality_model_results.csv
│
├── charts/
│   ├── 01_global_mortality_trend.png
│   ├── 02_country_specific_mortality_trends.png
│   ├── 03_cause_specific_mortality_trends.png
│   └── 04_birth_rate_vs_mortality_scatter.png
│
└── INFO_B211_Group3.py
```

---

# Function Documentation

---

## DATA LOADING FUNCTIONS

### `load_data(script_dir)`
**Purpose:**  
Loads the raw dataset using dynamic pathing and returns basic metadata.

**Inputs:**  
- `script_dir` — directory where the script is located.

**Outputs:**  
- `df` — raw dataframe  
- `df_head` — first 5 rows  
- `df_info` — dataframe info  
- `df_desc` — descriptive statistics  
- `df_shape` — dataset dimensions  

---

## DATA CLEANING FUNCTIONS

### `clean_data(df)`
**Purpose:**  
Cleans and filters the dataset to include only mortality‑related indicators.

**Operations:**  
- Standardizes column names  
- Filters mortality series  
- Removes rows missing country names  
- Drops rows missing all year values  
- Converts year columns to numeric  
- Saves cleaned dataset  

**Outputs:**  
- `mortality_df` — cleaned dataframe  

---

# SCIPY STATISTICAL ANALYSIS FUNCTIONS

---

## `analyze_trends_linreg(df, output_path)`
**Purpose:**  
Computes linear regression trend statistics for each `(country, series)` pair.

**Outputs include:**  
- slope  
- intercept  
- r²  
- p‑value  
- standard error  

**Notes:**  
- Uses `scipy.stats.linregress`  
- Skips constant or all‑NaN series  

---

## `generate_linear_trendlines(df, output_path)`
**Purpose:**  
Generates predicted mortality values for each year (2000–2023) using linear regression.

**Outputs:**  
- Long‑format dataframe with predicted values  
- CSV file  

**Notes:**  
- Designed for visualization  

---

## `pairwise_ttests(df, output_path)`
**Purpose:**  
Performs independent t‑tests comparing mortality time series between every pair of countries.

**Outputs:**  
- `country_1`  
- `country_2`  
- `t_statistic`  
- `p_value`  

**Notes:**  
- Uses `scipy.stats.ttest_ind`  
- Handles missing values with `nan_policy="omit"`  

---

# PREDICTIVE MODELING FUNCTIONS

---

## `run_predictive_modeling(df)`
**Purpose:**  
Runs machine‑learning models for each mortality indicator.

**Models:**  
- Linear Regression  
- Random Forest Regressor  

**Metrics Returned:**  
- R²  
- MAE  
- Explained variance  

**Outputs:**  
- DataFrame of model performance  
- CSV file  

---

# VISUALIZATION FUNCTIONS

---

## `plot_global_mortality_trends(df, output_dir)`
Plots global average mortality from 2000–2020.

## `plot_country_comparison(df, selected_countries, output_dir)`
Compares mortality trends across selected countries.

## `cause_specific(df, causes, output_dir)`
Plots mortality trends for selected causes of death.

## `birth_rate_vs_mortality_scatter(df, raw_df, output_dir)`
Examines the relationship between global birth rates and mortality.

---

# Outputs

### CSV Files
- cleaned_mortality_data.csv  
- trend_coefficients.csv  
- trendline_predictions.csv  
- pairwise_ttests.csv  
- sklearn_mortality_model_results.csv  

### Charts (PNG)
Saved in `/charts/`:
- Global mortality trend  
- Country comparison  
- Cause‑specific trends  
- Birth rate vs mortality  

---

# End of README
