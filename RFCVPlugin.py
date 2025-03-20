#!/usr/bin/env python
# coding: utf-8

# In[1]:
# NEW PREDICTORS: Intron length, gene length, # of introns
# OLD PREDICTORS: Amino acid (sequence) length, GC content, coding sequence length, transcript length

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_validate
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

def plotImportance(x, y, based, outputfile, data=None, orient=None):
 plt.figure(figsize=(10,6))
 if (orient == ""):
  sns.barplot(x=x, y=y)
 else:
  sns.barplot(x=x, y=y, data=data, orient=orient)
 plt.xlabel("Feature Importance")
 plt.ylabel("Features")
 plt.title("Feature Importances from "+based)
 plt.savefig(outputfile)
 plt.show()
       


def run_with_CV(nem_df, X, column, rs, cv, outputfile):
 X.drop(X.columns[X.columns.str.contains("unnamed", case=False)], axis=1, inplace=True)
 y = nem_df.iloc[:,column] # Classification
 rf = RandomForestClassifier(random_state=rs)
 cv_results = cross_validate(rf, X, y.values.ravel(), cv=cv, return_estimator=True)
 feature_importances = np.array([estimator.feature_importances_ for estimator in cv_results['estimator']])
 mean_importance = feature_importances.mean(axis=0)
 feature_importance_df = pd.DataFrame(mean_importance, index=X.columns, columns=["Importance"])
 feature_importance_df = feature_importance_df.sort_values(by="Importance", ascending=False)
 print(feature_importance_df)
 plotImportance("Importance", feature_importance_df.index, str(cv)+"-Fold Cross-Validation", outputfile, feature_importance_df, "h")
 cv_results = cross_validate(rf, X, y.values.ravel(), cv=cv, return_estimator=False)
 test_scores = cv_results['test_score']
 print("Accuracy scores for each fold:")
 print(test_scores)
 print(np.average(test_scores))

import PyPluMA
import PyIO
class RFCVPlugin:
 def input(self, inputfile):
    self.parameters = PyIO.readParameters(inputfile)
 def run(self):
     pass
 def output(self, outputfile):
   nem_df = pd.read_csv(PyPluMA.prefix()+"/"+self.parameters["original"], encoding = "latin-1")
   X = pd.read_csv(PyPluMA.prefix()+"/"+self.parameters["filtered"], encoding = "latin-1")
   run_with_CV(nem_df, X, int(self.parameters["catcol"]), int(self.parameters["numrds"]), int(self.parameters["fold"]), outputfile)

