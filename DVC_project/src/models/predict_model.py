import numpy as np
import pandas as pd
import pickle
import json
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score, recall_score, roc_auc_score

clf = pickle.load(open('./DVC_project/models/model.pkl','rb'))
test_data = pd.read_csv('./DVC_project/data/interim/test_tfidf.csv')

X_test = test_data.iloc[:,0:-1].values
y_test = test_data.iloc[:,-1].values

y_pred = clf.predict(X_test)

# Use the positive class from the trained model and convert labels to binary form
positive_label = clf.classes_[1]
positive_idx = np.where(clf.classes_ == positive_label)[0][0]
y_pred_proba = clf.predict_proba(X_test)[:, positive_idx]
y_true_binary = (y_test == positive_label).astype(int)
y_pred_binary = (y_pred == positive_label).astype(int)

# Calculate evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='macro', zero_division=0)
recall = recall_score(y_test, y_pred, average='macro', zero_division=0)
auc = roc_auc_score(y_true_binary, y_pred_proba)

metrics_dict={
    'accuracy':accuracy,
    'precision':precision,
    'recall':recall,
    'auc':auc
}

with open('DVC_project/reports/metrics.json', 'w') as file:
    json.dump(metrics_dict, file, indent=4)