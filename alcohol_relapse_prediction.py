# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.3.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# # Introduction
#
# The notebook aims to predict the alcohol relapse based on the activity on the Helping Hand application. 

# +
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import *
from sklearn.model_selection import train_test_split


# +
def train_test_split_by_user(x, y, test_size=0.25):
    # group data by user_id and count the number of records per user
    counts = x.groupby(x.index).count()[x.columns[0]].rename('counts').to_frame()
    # pick the fraction (specified by test_size) of users by random.
    # Try till their total records is close to the specified test_size:
    # (test_size - 0.5%) * data_size <= total records <= (test_size + 0.5%) * data_size
    test_users = counts.sample(frac=test_size, axis=0)
    while test_users.sum()['counts'] <= ((test_size - 0.005) * len(x)) \
            or test_users.sum()['counts'] >= (test_size + 0.005) * len(x):
                
        test_users = counts.sample(frac=test_size, axis=0)
        
    x_test = x.loc[test_users.index]
    y_test = y.loc[test_users.index]
    # create the train set with remaining users' data
    x_train = x.loc[np.setdiff1d(counts.index, x_test.index)]
    y_train = y.loc[np.setdiff1d(counts.index, x_test.index)]

    return x_train, x_test, y_train, y_test


def perform_classification(x, y):
    x_train, x_test, y_train, y_test = train_test_split_by_user(x, y, test_size=0.25)
    
    model_1d, model_3d, model_7d = train_random_forest(x_train, y_train)
    
    metrics_1d = compute_classification_metrics(model_1d.predict_proba(x_test)[:, 1], y_test['Relapse in 1 Day'].to_numpy(), 0.1)
    metrics_3d = compute_classification_metrics(model_3d.predict_proba(x_test)[:, 1], y_test['Relapse in 3 Days'].to_numpy(), 0.2)
    metrics_7d = compute_classification_metrics(model_7d.predict_proba(x_test)[:, 1], y_test['Relapse in 7 Days'].to_numpy(), 0.3)
    
    return model_1d, model_3d, model_7d, metrics_1d, metrics_3d, metrics_7d
           

def train_random_forest(x_train, y_train):
    model_1d = RandomForestClassifier(n_estimators=100)
    model_1d.fit(x_train, y_train['Relapse in 1 Day'].to_numpy())
    model_3d = RandomForestClassifier(n_estimators=100)
    model_3d.fit(x_train, y_train['Relapse in 3 Days'].to_numpy())
    model_7d = RandomForestClassifier(n_estimators=100)
    model_7d.fit(x_train, y_train['Relapse in 7 Days'].to_numpy())
    return model_1d, model_3d, model_7d


def compute_classification_metrics(y_proba, y_true, prediction_threshold):
    y_pred = [1 if y_p > prediction_threshold else 0 for y_p in y_proba]
    try:
        auc_score = roc_auc_score(y_true, y_proba)
    except ValueError:
        auc_score = 0
    return {'TP': sum([1 for (y_p, y_t) in zip(y_pred, y_true) if (y_p == 1 and y_t == 1)]),
            'TN': sum([1 for (y_p, y_t) in zip(y_pred, y_true) if (y_p == 0 and y_t == 0)]),
            'FP': sum([1 for (y_p, y_t) in zip(y_pred, y_true) if (y_p == 1 and y_t == 0)]),
            'FN': sum([1 for (y_p, y_t) in zip(y_pred, y_true) if (y_p == 0 and y_t == 1)]),
            'Accuracy': round(accuracy_score(y_true, y_pred), 4),
            'Precision': round(precision_score(y_true, y_pred), 4),
            'Recall': round(recall_score(y_true, y_pred), 4),
            'F1_score': round(f1_score(y_true, y_pred), 4),
            'AUC': round(auc_score, 4),
            'Cohen_kappa': round(cohen_kappa_score(y_true, y_pred), 4),
            'MCC': round(matthews_corrcoef(y_true, y_pred), 4),
            'Hamming_loss': round(hamming_loss(y_true, y_pred), 4)}


# +
data = pd.read_csv('prediction_data.csv', sep=';', header=0, index_col=0)
# change columns names
features = ['Last Initial Survey Score',
            'Last Diary Sentiment', 'Diaries Sentiment Score Avg', 'Diaries Sentiment Score Std',
            'Last Diary Sentiment Z', 'Diaries Sentiment Score 7',
            'Diaries Sentiment Score Chg 7', 'Diaries Sentiment Score Chg 7 Ratio',
            'Last Diaries Doc2Vec Prediction', 'Diaries Doc2Vec Prediction Chg 7', 'Diaries Doc2Vec Prediction Chg 7 Ratio',
            'Messages Avg Length Chg 7', 'Messages Upper Words Freq Chg 7',
            'Messages Positive Emojis Count 7 Ratio', 'Messages Negative Emojis Count 7 Ratio', 
            'Last Messages Doc2vec Prediction', 'Messages Doc2vec Prediction Chg 7', 'Messages Doc2vec Prediction Chg 7 Ratio',
            'Hunger Surveys Score 7', 'Hunger Surveys Score Change 7', 'Hunger Surveys Score Change 7 Ratio',
            'Last Hunger Survey Score', 'Hunger Surveys Score Avg', 'Hunger Surveys Score Std',
            'Last Hunger Survey Score Z', 'Hunger Surveys Score Slope',
            'Days Since First Activity', 'Days Since Last Activity', 'Days Since Last Relapse',
            'Activity Days Avg', 'Activity Actions Avg', 'Activity Actions Std',
            'Is Weekday in 1 Day', 'Is Weekday in 3 Days',
            'Relapses Count', 'Relapses Frequency', 'Relapses Frequency Change 7', 
            'Relapses Frequency Change 7 Ratio',
            'Relapse in 1 Day', 'Relapse in 3 Days', 'Relapse in 7 Days']


data.columns = ['Timestamp'] + features
# -

# # Relapses distribution

# +
print('1-day relapses fraction: %.2f' % (len(data[data['Relapse in 1 Day'] == 1]) / len(data)))
print('3-days relapses fraction: %.2f' % (len(data[data['Relapse in 3 Days'] == 1]) / len(data)))
print('7-days relapses fraction: %.2f' % (len(data[data['Relapse in 7 Days'] == 1]) / len(data)))

data[['Relapse in 1 Day']].hist(grid=False, bins=2, figsize=[15, 7]);
data[['Relapse in 3 Days']].hist(grid=False, bins=2, figsize=[15, 7]);
data[['Relapse in 7 Days']].hist(grid=False, bins=2, figsize=[15, 7]);
# -

# As we can see, most o the records generated for active users are negative in terms of the target features (no relapse). It means that for the most period of time users that are active in the app do not experience relapses.

# ## Correlation matrix

corr_df = data.corr()
f = plt.figure(figsize=(19, 15))
plt.matshow(corr_df, cmap=cm.tab10, fignum=f.number)
plt.xticks(range(len(features)), features, fontsize=14, rotation=90)
plt.yticks(range(len(features)), features, fontsize=14)
cb = plt.colorbar()
cb.ax.tick_params(labelsize=14)

x = data.drop(columns=['Timestamp', 'Relapse in 1 Day', 'Relapse in 3 Days', 'Relapse in 7 Days'])
y = data[['Relapse in 1 Day', 'Relapse in 3 Days', 'Relapse in 7 Days']]

x_train, x_test, y_train, y_test = train_test_split_by_user(x, y, test_size=0.25)

model_1d, model_3d, model_7d, metrics_1d, metrics_3d, metrics_7d = perform_classification(x, y)

feature_importances_1d = pd.DataFrame(model_1d.feature_importances_,
                                      index = x.columns,
                                      columns=['importance']).sort_values('importance', ascending=False)
feature_importances_3d = pd.DataFrame(model_3d.feature_importances_,
                                      index = x.columns,
                                      columns=['importance']).sort_values('importance', ascending=False)
feature_importances_7d = pd.DataFrame(model_7d.feature_importances_,
                                      index = x.columns,
                                      columns=['importance']).sort_values('importance', ascending=False)

# # Model 1d

# ## Feature importance plot

feature_importances_1d.plot.bar(figsize=(16,7));

#
# ## Top 20 features
# The most important features selected based on the feature importance plot using the elbow method.

top_1d = feature_importances_1d.head(20)
top_1d

#
#
# # Model 3d

# ## Feature importance plot

feature_importances_3d.plot.bar(figsize=(16,7));

# ## Top 20 features
# The most important features selected based on the feature importance plot using the elbow method.

top_3d = feature_importances_3d.head(20)
top_3d

# # Model 7d

# ## Feature importance plot

feature_importances_7d.plot.bar(figsize=(16,7));

# ## Top 20 features
# The most important features selected based on the feature importance plot using the elbow method.

top_7d = feature_importances_7d.head(20)
top_7d

# # Final selection
# The outersection of the most important features from the three models, are the final features to be used in prediction:

top_features = list(set(np.concatenate([top_1d.index, top_3d.index, top_7d.index])))

# # Prediction 
# The predictions using the selected features

# +
x = data[top_features]

y = data[['Relapse in 1 Day', 'Relapse in 3 Days', 'Relapse in 7 Days']]

model_1d, model_3d, model_7d, metrics_1d, metrics_3d, metrics_7d = perform_classification(x, y)
# -

# ## Model 1d

pd.DataFrame(data=metrics_1d.values(), index=metrics_1d.keys(), columns=['Model 1d'])

# ## Model 3d

pd.DataFrame(data=metrics_3d.values(), index=metrics_1d.keys(), columns=['Model 3d'])

# ## Model 7d

pd.DataFrame(data=metrics_7d.values(), index=metrics_1d.keys(), columns=['Model 7d'])

# # Conclusions
# Using the Random forest classifier the most important features can be easily refined. In all three models the most important features are nearly the same, with the biggest difference between the 1d model and the two other models, i.e 3d and 7d.
