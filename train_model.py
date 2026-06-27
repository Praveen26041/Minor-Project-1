import os
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# 1. Set up directories
os.makedirs('model', exist_ok=True)

# 2. Load the Dataset
data_path = 'data/matches.csv'
if not os.path.exists(data_path):
    # Fallback if running directly in the root directory
    data_path = 'matches.csv'

print(f"Loading dataset from: {data_path}")
df = pd.read_csv(data_path)

# 3. Data Cleaning & Feature Selection
# Selecting relevant features for predicting the match winner
features = ['team1', 'team2', 'venue', 'toss_winner', 'toss_decision']
target = 'winner'

# Drop rows where the target ('winner') or key features are missing
df = df.dropna(subset=[target] + features)

X = df[features].copy()
y = df[target].copy()

print(f"Dataset loaded successfully. Total records for training: {len(X)}")

# 4. Categorical Encoding
# Since teams, venues, and decisions are strings, we must encode them to integers.
# We create a dictionary to save an individual LabelEncoder for each column.
encoders = {}

print("Encoding categorical features...")
for col in features:
    le = LabelEncoder()
    # Fit on all unique values present in both teams/toss to handle unseen pairs consistently
    if col in ['team1', 'team2', 'toss_winner']:
        # Combine all team names to ensure uniform encoding across team columns
        all_teams = pd.concat([df['team1'], df['team2'], df['toss_winner']]).unique()
        le.fit(all_teams)
    else:
        le.fit(df[col])
        
    X[col] = le.transform(X[col])
    encoders[col] = le

# Encode the target variable (winner)
target_encoder = LabelEncoder()
# Ensure target encoder knows all possible teams
all_possible_winners = pd.concat([df['team1'], df['team2'], df['winner']]).unique()
target_encoder.fit(all_possible_winners)
y = target_encoder.transform(y)

# 5. Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 6. Model Training
print("Training Random Forest Classifier...")
model = RandomForestClassifier(n_estimators=150, max_depth=12, random_state=42)
model.fit(X_train, y_train)

# 7. Evaluation
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("\n" + "="*30)
print(f"Model Accuracy: {accuracy * 100:.2f}%")
print("="*30)

# 8. Save Model and Preprocessors to the 'model/' folder
print("\nSaving artifacts to 'model/' directory...")
joblib.dump(model, 'model/match_winner_model.pkl')
joblib.dump(encoders, 'model/feature_encoders.pkl')
joblib.dump(target_encoder, 'model/target_encoder.pkl')

print("Artifacts saved successfully:")
print(" - model/match_winner_model.pkl (Trained Model)")
print(" - model/feature_encoders.pkl (Feature Label Encoders)")
print(" - model/target_encoder.pkl (Target Label Encoder)")