import pandas as pd
from sklearn.metrics import roc_curve, precision_recall_curve, auc

df = pd.read_csv("data/data.csv")
fpr , tpr , threshold  = roc_curve(list(df["labels"]), list(df["predictions"]), drop_intermediate=False)
print("AUC ROC {}".format(auc(fpr,tpr)))

roc_sklearn = pd.DataFrame({"fpr":fpr, "tpr":tpr, "op":threshold})
roc_sklearn.to_csv("data/roc_sklearn.csv",index=False)

precision , recall , _  = precision_recall_curve(list(df["labels"]), list(df["predictions"]))
print("AUC PR {}".format(auc(recall,precision)))

pr_sklearn = pd.DataFrame({"precision":precision, "recall":recall})
pr_sklearn.to_csv("data/pr_sklearn.csv",index=False)
