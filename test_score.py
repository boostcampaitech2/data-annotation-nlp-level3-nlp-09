import pandas as pd
from sklearn.metrics import accuracy_score

data= pd.read_csv('./prediction/submission0.csv')
target= pd.read_csv('./data/test.csv')

pred_y= data['pred_label'].values
target_y= target['class'].values

print(accuracy_score(pred_y, target_y))