Predicting IPL Match Outcomes Using Random Forest 🏏
Overview
This repository contains a complete machine learning workflow designed to predict the winning team of an Indian Premier League (IPL) cricket match. By leveraging historical match data—including venues, toss decisions, and competing teams—this project demonstrates data preprocessing, categorical encoding, and predictive modeling using a Random Forest Classifier.

📂 Repository Structure
IPL-Predictor/
│
├── data/
│   └── matches.csv                 # The raw IPL matches dataset
│
├── model/
│   ├── match_winner_model.pkl      # Trained Random Forest model
│   ├── feature_encoders.pkl        # LabelEncoders for input features
│   └── target_encoder.pkl          # LabelEncoder for the target variable
│
├── notebook/
│   └── Minor Project.ipynb         # Jupyter Notebook with complete EDA and training code
│
├── results/
│   ├── MINOR PROJECT 1.pdf         # Complete project report
│   └── [Screenshots/Plots]         # Confusion matrix and evaluation plots
│
├── train_model.py                  # Python script for model training and artifact generation
└── README.md                       # Project documentation

🎯 Problem Statement
Cricket is a highly unpredictable sport influenced by numerous dynamic factors. The goal of this project is to predict the winning team of an IPL match based strictly on pre-match and toss data. 
Specifically, we want to evaluate how much influence the playing venue, the competing teams, the toss winner, and the toss decision (bat/field) have on the final outcome.

Dataset Description
The dataset utilized for this project is based on historical IPL match data.

Target Variable: winner (The team that won the match)

Features Used: * team1: The first playing team

team2: The second playing team

venue: The stadium where the match was played

toss_winner: The team that won the coin toss

toss_decision: The decision made by the toss winner (bat or field)

Data Cleaning Note: Matches that ended with "no result" (e.g., due to rain) resulted in missing values in the target column. These specific rows were dropped from the training set.

Methodology
Data Preprocessing & Cleaning:

Removed any duplicate records to prevent model bias.

Dropped irrelevant columns (umpires, dates, match types) to focus purely on environmental and team factors.

Handled missing values by dropping corrupted rows.

Feature Encoding:

Machine learning models require numerical inputs. A LabelEncoder was used to convert string categories into numerical representations.

Special Handling: Combined all team names across team1, team2, and toss_winner before fitting the encoder to ensure uniform numerical mapping for every franchise, regardless of which column they appeared in.

Model Selection & Training:

Algorithm: Random Forest Classifier

Hyperparameters: n_estimators=150, max_depth=12, random_state=42

Data Split: 80% Training / 20% Testing

