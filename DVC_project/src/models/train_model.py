import numpy as np
import pandas as pd
import pickle
import yaml
from sklearn.ensemble import GradientBoostingClassifier

def load_params(params_path: str) -> dict:
    with open(params_path, 'r') as file:
        params = yaml.safe_load(file)
    return params['train_model']

params = load_params('params.yaml')

# fetch the data from data/processed
train_data = pd.read_csv('./DVC_project/data/interim/train_bow.csv')

X_train = train_data.iloc[:,0:-1].values
y_train = train_data.iloc[:,-1].values

# Define and train the XGBoost model

clf = GradientBoostingClassifier(
    n_estimators=params['n_estimator'],
    learning_rate=params['learning_rate']
)
clf.fit(X_train, y_train)

# save
pickle.dump(clf, open('DVC_project/models/model.pkl','wb'))