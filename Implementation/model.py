# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import StackingClassifier
import pickle

# Load the dataset
df = pd.read_csv("data.csv")

# Separate features and target
X = df.drop(['PPE', 'Shimmer:APQ5', 'MDVP:PPQ', 'Shimmer:DDA', 'MDVP:Shimmer(dB)',
              'MDVP:APQ', 'MDVP:RAP', 'HNR', 'MDVP:Jitter(Abs)', 'Jitter:DDP', 'Shimmer:APQ3', 'name', 'status'], axis=1)
y = df["status"]

# Data Normalization
scaler = StandardScaler()
features = scaler.fit_transform(X)

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(features, y, test_size=0.3, random_state=42)

# Define base classifiers

base_classifiers = [
    ('random_forest', RandomForestClassifier(n_estimators=100, criterion='gini', max_depth=None,
                                              min_samples_split=2, min_samples_leaf=1, max_features='sqrt', random_state=22)),
    ('svm', SVC(kernel='rbf', C=1.0, gamma='scale', random_state=22)),
    ('knn', KNeighborsClassifier(n_neighbors=5, weights='uniform', algorithm='auto', leaf_size=30, p=2, metric='minkowski'))
]


# Define the meta-learner
meta_learner = DecisionTreeClassifier(random_state=20)

# Create the stacking classifier
clf1 = StackingClassifier(estimators=base_classifiers, final_estimator=meta_learner)

# Fit the stacking classifier
clf1.fit(X_train, y_train)

# Pickle the stacking classifier model
pickle.dump(clf1, open('model_stacking.pkl', 'wb'))
