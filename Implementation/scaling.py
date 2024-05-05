import pandas as pd
from sklearn.preprocessing import StandardScaler
import pickle

# Load the dataset
df = pd.read_csv("data.csv")

# Separate features and target
X = df.drop(['PPE', 'Shimmer:APQ5', 'MDVP:PPQ', 'Shimmer:DDA', 'MDVP:Shimmer(dB)',
              'MDVP:APQ', 'MDVP:RAP', 'HNR', 'MDVP:Jitter(Abs)', 'Jitter:DDP', 'Shimmer:APQ3', 'name', 'status'], axis=1)
y = df["status"]

# Data Normalization
scaler = StandardScaler()
scaler.fit(X)

# Pickle the scaler
pickle.dump(scaler, open('scaling.pkl', 'wb'))
