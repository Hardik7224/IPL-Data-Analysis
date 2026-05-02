# IPL Data Analysis

[![Python](https://img.shields.io/badge/Python-3.8+-blue?logo=python&logoColor=white)](https://www.python.org/) &nbsp;&nbsp;&nbsp;
[![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-green?logo=pandas&logoColor=white)](https://pandas.pydata.org/) &nbsp;&nbsp;&nbsp;
[![NumPy](https://img.shields.io/badge/NumPy-Numerical%20Computing-blue?logo=numpy&logoColor=white)](https://numpy.org/) &nbsp;&nbsp;&nbsp;
[![Matplotlib](https://img.shields.io/badge/Matplotlib-Visualization-orange)](https://matplotlib.org/) &nbsp;&nbsp;&nbsp;
[![Seaborn](https://img.shields.io/badge/Seaborn-Statistical%20Plots-teal)](https://seaborn.pydata.org/) &nbsp;&nbsp;&nbsp;
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange?logo=jupyter&logoColor=white)](https://jupyter.org/)

A **data analysis project on IPL datasets** using **Python**, **Pandas**, **NumPy**, and **data visualization libraries**. This project explores match-level and ball-by-ball data to uncover trends, patterns, and statistical insights in the Indian Premier League (IPL).

---

## 🎯 Features

- Data cleaning and preprocessing of IPL datasets  
- Exploratory Data Analysis (EDA)  
- Analysis of **matches and deliveries data**  
- Visualization of trends and patterns  
- Statistical insights on team and player performance  

---

## 💻 Installation

Open your terminal and run the following commands step by step:

```bash
# Clone the repository and navigate into it
git clone https://github.com/Hardik7224/IPL-Data-Analysis.git && cd IPL-Data-Analysis
```

```bash
# Create a virtual environment (optional)
python -m venv venv
```

```bash
# Activate the virtual environment
# On Linux / macOS:
source venv/bin/activate
# On Windows:
venv\Scripts\activate
```

```bash
# Install required dependencies
pip install -r requirements.txt
```

```bash
# Run Jupyter Notebook
jupyter notebook
```

---

## 🛠️ Requirements

- Python 3.8+  
- Pandas  
- NumPy  
- Matplotlib  
- Seaborn  
- Jupyter Notebook  

---

## 📂 Project Structure

```bash
# 📂 IPL-Data-Analysis/
# │
# ├── data/
# │   ├── matches.csv
# │   ├── deliveries.csv
# │
# ├── notebooks/
# │   ├── ipl_data_analysis.ipynb   # Combined matches + deliveries analysis
# │   ├── ipl_eda.ipynb             # Exploratory Data Analysis
# │
# ├── README.md
```

---

## ⚙️ How It Works

1. Loads IPL datasets (`matches` and `deliveries`) using Pandas  
2. Cleans and preprocesses the data  
3. Performs exploratory data analysis (EDA)  
4. Analyzes team performance, player statistics, and match outcomes  
5. Visualizes insights using Matplotlib and Seaborn  

---

## 📊 Key Insights

- Toss decisions can influence match outcomes  
- Certain venues favor specific teams  
- Player performance significantly impacts results  
- Scoring trends have evolved over IPL seasons  
