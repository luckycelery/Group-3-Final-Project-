# Group-3-Final-Project-
INFO-B211 Final Project - Demonstrate proficiency in utilizing Python for data analysis

## How to Run the Program

This project uses a shared Conda environment to ensure that everyone on the team is working with the same versions of Python and the required libraries. Follow the steps below to set up your environment and run the program.

---

### 1. Set Up the Virtual Environment

Before running the program, you must create and activate the project environment using the `environment.yml` file included in this repository.

#### **Steps**

1. Install Anaconda or Miniconda if you haven't already.

2. Open your terminal

3. Navigate to the folder where you cloned this repository: 
    cd path/to/your/project/folder

4. Create the environment using the provided file: 
    conda env create -f environment.yml

5. Activate the environment: 
    conda activate group3

6. If the environment file is updated later, update your environment with: 
    conda env update -f environment.yml --prune

7. If your environment ever breaks, you can delete and recreate it: 
    conda remove --name group3 --all
    conda env create -f environment.yml


Once the environment is activated, you are ready to run the program.

---

### 2. Running the Program

After activating the environment, run the program by navigating to the project folder and executing:

(Or whichever script becomes the main execution file as the project develops.)

The program will:

- load the dataset using dynamic pathing  
- run cleaning functions (once implemented)  
- perform exploratory or statistical analysis  
- print initial output to the terminal  

More detailed instructions will be added as additional functions and modules are completed.

---

### Notes

- Always activate the environment before running the program! 
- If you add new dependencies, update the `environment.yml` so the whole team stays in sync.  
- This section will expand as the project grows (e.g., adding CLI arguments, visualization outputs, model training steps, etc.).


### What Goes Where

**INFO_B211_Group3.py**  
This is the main script for the entire project.  
All functions, analysis steps, and program execution will live here so the team can work in one place without navigating multiple files.

**Data.csv & Series-Metadata.csv**  
These are the datasets used for analysis.  
They must remain in the root directory so the dynamic pathing in the script can locate them.

**environment.yml**  
Shared Conda environment file.  
Everyone uses this to create the same environment so the code runs consistently across all machines.

**README.md**  
Documentation for the project, including setup instructions, environment details, and explanations of how to run the program.

### Why We Are Keeping a Single-Script Structure

- Several team members are newer to Python and Git  
- Splitting code into multiple modules (`/src/cleaning.py`, `/src/modeling.py`, etc.) increases complexity and merge conflict risk.  
- A single script ensures everyone can see the full workflow in one place.  
- This structure is easier to teach, easier to debug, and easier to maintain for a short‑term class project.






