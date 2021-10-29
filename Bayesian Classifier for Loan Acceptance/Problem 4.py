import pandas as pd
from scipy.stats import multivariate_normal
from sklearn.model_selection import train_test_split
import numpy as np

raw_data = pd.read_excel("Universalbank_1500.xlsx")
bank_df = pd.DataFrame(raw_data)


def naive_bayes_train(df):
    # Calculating Prior Probability
    pp_0 = df['Personal Loan'].value_counts()[0] / len(df['Personal Loan'])
    pp_1 = df['Personal Loan'].value_counts()[1] / len(df['Personal Loan'])
    # For calculating Likelihood
    # Mean of each category
    d0 = df[df['Personal Loan'] == 0]
    d1 = df[df['Personal Loan'] == 1]
    mn0 = [np.mean(d0.iloc[:, 0]), np.mean(d0.iloc[:, 1])]
    mn1 = [np.mean(d1.iloc[:, 0]), np.mean(d1.iloc[:, 1])]
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
    CC = 1
    online = 1
    x = [CC, online]
    x0 = multivariate_normal.pdf(x, mn0, np.array(cv0))
    x1 = multivariate_normal.pdf(x, mn1, np.array(cv1))
    post0 = x0 * pp_0
    post1 = x1 * pp_1
    return post1


if __name__ == '__main__':
    train_df, tst_df = train_test_split(bank_df, test_size=0.4)
    pvt_table0 = pd.pivot_table(bank_df, index=['CreditCard', 'Personal Loan'], columns=['Online'], aggfunc='count',
                                values='ID')
    print(pvt_table0)
    pvt_table1 = pd.pivot_table(bank_df, index=['Personal Loan'], columns=['Online'], aggfunc='count', values='ID')
    pvt_table2 = pd.pivot_table(bank_df, index=['Personal Loan'], columns=['CreditCard'], aggfunc='count', values='ID')
    print("Pivot Table 1 = \n", pvt_table1)
    print("Pivot Table 2 = \n", pvt_table2)
    train_df = train_df.filter(['CreditCard', 'Online', 'Personal Loan'], axis=1)
    train_var = naive_bayes_train(train_df)
    prob = naive_bayes_classifier(train_var, train_df)
    print("The probability of accepting loan = ", prob)
