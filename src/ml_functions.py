import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style, rcParams
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from sklearn.pipeline import Pipeline
from sklearn.decomposition import NMF, PCA
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV, cross_validate
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import xgboost.sklearn as xgb
import xgboost as x
import gensim
import seaborn as sns
import pickle

style.use('ggplot')
rcParams['font.size'] = 14
rcParams['legend.fontsize'] = 'small'
rcParams['figure.titlesize'] = 'large'

def elbow_plot(pca, filename=None):
    total_variance = np.sum(pca.explained_variance_)
    cum_variance = np.cumsum(pca.explained_variance_)
    prop_var_expl = cum_variance/total_variance
    idx = np.argwhere(prop_var_expl > 0.9)[0][0] + 1

    fig, ax = plt.subplots(figsize=(8,6))
    ax.plot(prop_var_expl, color='red', linewidth=2, label='Explained variance')
    ax.axhline(0.9, label=f'90% goal (at n_components={idx})', linestyle='--', color="black", linewidth=1)
    ax.set_ylabel('cumulative prop. of explained variance')
    ax.set_xlabel('number of principal components')
    ax.set_title('Elbow Plot')
    ax.legend()
    if filename != None:
        plt.savefig('../images/pca5c_cumulsum_elbow.png')
    return idx

def modelfit(alg, X, y, useTrainCV=True, cv_folds=5, early_stopping_rounds=50):

    if useTrainCV:
        xgb_param = alg.get_xgb_params()
        y = pd.get_dummies(y)
        xgtrain = x.DMatrix(X, label=y)
        cvresult = x.cv(xgb_param, xgtrain, num_boost_round=alg.get_params()['n_estimators'], nfold=cv_folds,
            metrics='auc', early_stopping_rounds=early_stopping_rounds)
        alg.set_params(n_estimators=cvresult.shape[0])

    #Fit the algorithm on the data
    alg.fit(X, y, eval_metric='f1_weighted')

    #Predict training set:
    y_pred = alg.predict(X)
#     dtrain_predprob = alg.predict_proba(X)[:,1]

    #Print model report:
    print("\nModel Report")
    print(classification_report(y, y_pred))

#     print "AUC Score (Train): %f" % metrics.roc_auc_score(y, dtrain_predprob)

    feat_imp = pd.Series(alg.booster().get_fscore()).sort_values(ascending=False)
    feat_imp.plot(kind='bar', title='Feature Importances')
    plt.ylabel('Feature Importance Score')
