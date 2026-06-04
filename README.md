# IPL Data Analysis

[![Python](https://img.shields.io/badge/Python-3.8+-blue?logo=python&logoColor=white)](https://www.python.org/) &nbsp;&nbsp;&nbsp;
[![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-green?logo=pandas&logoColor=white)](https://pandas.pydata.org/) &nbsp;&nbsp;&nbsp;
[![NumPy](https://img.shields.io/badge/NumPy-Numerical%20Computing-blue?logo=numpy&logoColor=white)](https://numpy.org/) &nbsp;&nbsp;&nbsp;
[![Matplotlib](https://img.shields.io/badge/Matplotlib-Visualization-orange)](https://matplotlib.org/) &nbsp;&nbsp;&nbsp;
[![Seaborn](https://img.shields.io/badge/Seaborn-Statistical%20Plots-teal)](https://seaborn.pydata.org/) &nbsp;&nbsp;&nbsp;
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange?logo=jupyter&logoColor=white)](https://jupyter.org/)

📌 Overview

The IPL Data Analysis & Match Winner Prediction System is an end-to-end Data Analytics and Machine Learning project designed to analyze historical Indian Premier League (IPL) data and predict match outcomes.

The project performs detailed Exploratory Data Analysis (EDA), generates insights about teams and players, visualizes trends using interactive dashboards, and leverages Machine Learning models to predict match winners and season-level achievements.

---

## Key Capabilities
- 📊 IPL Data Analysis
- 🤖 Match Winner Prediction
- 🧡 Orange Cap Prediction
- 💜 Purple Cap Prediction
- 📉 Team & Player Performance Analytics

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
pip install -r Requirements.txt
```

```bash
# Run Jupyter Notebook
jupyter notebook
```

---

## 🛠️ Requirements
- pandas>=1.5.0
- numpy>=1.23.0
- scikit-learn>=1.2.0
- matplotlib>=3.6.0
- streamlit>=1.25.0


---

## 📂 Project Structure

```bash

```text
IPL-Data-Analysis/
│
├── IPL_Prediction/
│   │
│   ├── data/
│   │
│   ├── models/
│   │   ├── winner_model.pkl
│   │   ├── orange_cap_model.pkl
│   │   └── purple_cap_model.pkl
│   │
│   ├── src/
│   │   ├── __init__.py
│   │   ├── match_winner.py
│   │   ├── orange_cap.py
│   │   └── purple_cap.py
│   │
│   ├── app.py
│   ├── predict.py
│   ├── config.py
│   └── main.py
│
├── notebooks/
│   ├── ipl_data_analysis.ipynb
│   └── ipl_eda.ipynb
│
├── raw_data.zip
├── processed_data.zip
├── Requirements.txt
└──README.md

```

```

---

## 📊 Analysis Performed

### 🏆 Team Analysis

* Most Successful IPL Teams
* Win Percentage Analysis
* Home vs Away Performance
* Team Consistency Evaluation

### 🏟️ Venue Analysis

* Venue-wise Winning Patterns
* Toss Impact by Stadium
* Average First Innings Score
* Venue Performance Trends

### 🏏 Batting Analysis

* Top Run Scorers
* Strike Rate Comparison
* Orange Cap Analysis
* Boundary Statistics
* Season-wise Batting Performance

### 🎯 Bowling Analysis

* Top Wicket Takers
* Economy Rate Comparison
* Purple Cap Analysis
* Venue-wise Bowling Performance
* Bowling Impact on Match Outcomes

### 📈 Match Analysis

* Highest Team Totals
* Highest Successful Chases
* Super Over Matches
* Match Outcome Trends
* Winning Margin Analysis
* Season-wise Match Statistics

---

## 📊 Key Insights

- Toss decisions can influence match outcomes  
- Certain venues favor specific teams  
- Player performance significantly impacts results  
- Scoring trends have evolved over IPL seasons

## 📌 Project Statistics

- Seasons Analyzed: 18+
- Matches Processed: 1,000+
- Deliveries Analyzed: 250,000+
- Teams Analyzed: 10+
- Players Analyzed: 1,500+
- Machine Learning Models Trained: 3+
