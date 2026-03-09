
import numpy as np, pandas as pd, json, sys
from sklearn.metrics import accuracy_score, f1_score

sub=sys.argv[1]

y_true=pd.read_csv("evaluation/test_labels.csv",header=None)[0]
y_pred=pd.read_csv(sub,header=None)[0]

acc=float(accuracy_score(y_true,y_pred))
f1=float(f1_score(y_true,y_pred,average="macro"))

json.dump({"accuracy":acc,"f1_macro":f1},open("score.json","w"))
print("Accuracy:",acc)
