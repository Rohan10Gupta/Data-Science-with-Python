import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal
from sklearn.metrics import roc_curve, auc

raw_data = pd.read_csv("RidingMowers.csv")
rm_df = pd.DataFrame(raw_data)


def naive_bayes_train(df):
    # Converting De to numerical
    df.Ownership = pd.factorize(df.Ownership)[0]
    # Calculating Prior Probability
    pp_0 = df.Ownership.value_counts()[0] / len(df.Ownership)
    pp_1 = df.Ownership.value_counts()[1] / len(df.Ownership)
    # For calculating Likelihood
    # Mean of each category
    d0 = df[df.Ownership == 0]
    d1 = df[df.Ownership == 1]
    print(d1)
    mn0 = [np.mean(d0.iloc[:, 0]), np.mean(d0.iloc[:, 1])]
    mn1 = [np.mean(d1.iloc[:, 0]), np.mean(d1.iloc[:, 1])]
    print(mn1)
    # Covariance of each category
    cv1 = d1.cov()
    cv1 = cv1.iloc[0:2, 0:2]
    cv0 = d0.cov()
    cv0 = cv0.iloc[0:2, 0:2]
    return pp_0, pp_1, mn0, mn1, cv0, cv1


def naive_bayes_classifier(train, df):
    pp_0 = train[0]
    pp_1 = train[1]
    mn0 = train[2]
    mn1 = train[3]
    cv0 = train[4]
    cv1 = train[5]
    df.Ownership = pd.factorize(df.Ownership)[0]
    prob1 = []
    for ind in df.index:
        inc = df['Income'][ind]
        lot = df['Lot_Size'][ind]
        x = [inc, lot]
        x1 = multivariate_normal.pdf(x, mn1, np.array(cv1))
        post1 = x1 * pp_1
        prob1.append(post1)
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    fpr, tpr, _ = roc_curve(df, prob1)
    roc_auc = auc(fpr, tpr)
    return df


if __name__ == '__main__':
    train_var = naive_bayes_train(rm_df)
    rm_df_class = naive_bayes_classifier(train_var, rm_df)
