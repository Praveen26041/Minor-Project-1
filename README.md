Overview
This repository contains a complete machine learning workflow designed to predict the winning team of an Indian Premier League (IPL) cricket match. By leveraging historical match data, this project demonstrates data preprocessing, categorical encoding, and predictive modeling using a Random Forest Classifier.

Repository Structure
data/ – Contains the matches.csv dataset.

model/ – Contains the trained Random Forest model and LabelEncoders (.pkl files).

notebook/ – Contains the Jupyter Notebook (Minor Project.ipynb) used for exploratory data analysis (EDA) and model training.

results/ – Contains the project report (MINOR PROJECT 1.pdf) and generated evaluation plots (confusion matrix, result graphs).

README.md – Complete project details and documentation.

Problem Statement
The objective of this project is to predict the winning team of an IPL cricket match based on historical match data, specifically looking at pre-match factors including the venue, toss decisions, and competing teams.

Dataset
The dataset utilized for this project is the "IPL Complete Dataset" sourced from Kaggle. It contains historical data of IPL matches spanning multiple seasons.

Dataset Link: Kaggle - IPL Complete Dataset

Target Variable: winner

Features Used: team1, team2, toss_winner, toss_decision, and venue.

Methodology
To prepare the raw data and train the machine learning model, the following steps were executed:

Data Preprocessing:

Duplicate Removal: Checked for and removed duplicate rows to prevent model bias.

Feature Selection: Dropped irrelevant columns (like umpires or match dates) to focus strictly on teams, toss dynamics, and venue.

Handling Missing Values: Matches that ended with "no result" resulted in missing values in the winner column. These rows were dropped.

Feature Encoding:

Machine learning models require numerical inputs. A LabelEncoder was used to convert string categories into numerical representations for the teams, venues, and toss decisions.

Model Training:

Algorithm: Random Forest Classifier

Hyperparameters: n_estimators=150, max_depth=12, random_state=42

Data Split: 80% Training Data / 20% Testing Data

Results and Evaluation Metrics
The model was evaluated using standard classification metrics.

Accuracy: 0.4688 (46.88%) - The overall percentage of matches the model predicted correctly.

Precision: 0.4831 - The accuracy of the positive predictions.

Recall: 0.4688 - The model's ability to find all the correct winning instances.

F1-Score: 0.4655 - The harmonic mean of precision and recall.

Confusion Matrix Analysis:
The visual confusion matrix (available in the results/ folder) visualizes the exact breakdown of true positive predictions versus false classifications across all encoded IPL teams, showing exactly where the model succeeded and where it confused one team for another.

Conclusion
This minor project successfully demonstrates a complete machine learning workflow. By utilizing a Random Forest Classifier, we were able to predict IPL match outcomes based on toss decisions and venue locations. The evaluation metrics indicate that while cricket matches are highly unpredictable and depend on many in-game variables, baseline pre-match data still provides measurable predictive signals.
