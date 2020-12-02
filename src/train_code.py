import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from sklearn.pipeline import Pipeline
from sklearn.decomposition import NMF, PCA
from sklearn.model_selection import train_test_split

stopwords_ = set(stopwords.words('english'))

df = pd.read_csv('../data/train.csv')

X = df['Text']
y = df['Labels']

X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.2,
                                                    stratify=y,
                                                    random_state=42
                                                    )

def elbow_plot(pca):
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
    plt.savefig('../images/pca5c_cumulsum_elbow.png');
    return idx

