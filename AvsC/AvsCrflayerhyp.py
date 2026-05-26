

import pandas as pd
import numpy as np

from sklearn.model_selection import (
    StratifiedKFold,
    GridSearchCV,
    cross_val_predict
)

from sklearn.pipeline import Pipeline

from sklearn.preprocessing import StandardScaler

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score
)

from sklearn.ensemble import VotingClassifier

from sklearn.svm import SVC

from xgboost import XGBClassifier


import torch

print("\n===================================")
print("GPU AVAILABLE:")
print("===================================")

print(torch.cuda.is_available())


features_path = (
    r"C:\Users\sreeh\ML_fiiting\Layerwise_Features.csv"
)

ranksum_path = (
    r"C:\Users\sreeh\ML_fiiting\Ranksum_Layerwise_Features.csv"
)

participants_path = (
    r"C:\Users\sreeh\ML_fiiting\participants.tsv"
)


features_df = pd.read_csv(features_path)

features_df['subject'] = (
    features_df['subject']
    .str.extract(r'(sub-\d+)')
)


participants_df = pd.read_csv(
    participants_path,
    sep='\t'
)

participants_df = participants_df[
    ['participant_id', 'Group']
]

participants_df.columns = [
    'subject',
    'group'
]


df = pd.merge(
    features_df,
    participants_df,
    on='subject'
)


df = df[
    df['group'].isin(['F', 'C'])
]

print("\nDataset shape:")
print(df.shape)

print("\nClass counts:")
print(df['group'].value_counts())


ranksum_df = pd.read_csv(ranksum_path)


significant_features = ranksum_df[
    ranksum_df['F_vs_C_significant'] == True
]['feature'].tolist()

print("\n===================================")
print("SELECTED FEATURES")
print("===================================")

for feat in significant_features:
    print(feat)

print("\nTotal selected features:")
print(len(significant_features))

X = df[significant_features]


y = df['group'].map({
    'F': 0,
    'C': 1
})


X = X.replace(
    [np.inf, -np.inf],
    np.nan
)

X = X.fillna(
    X.mean()
)

cv = StratifiedKFold(

    n_splits=5,

    shuffle=True,

    random_state=42
)


svm = SVC(

    kernel='rbf',

    C=1,

    gamma=0.01,

    probability=True,

    random_state=42
)


xgb = XGBClassifier(

    objective='binary:logistic',

    eval_metric='logloss',

    random_state=42,


    tree_method='hist',

    device='cuda',


    colsample_bytree=0.7,

    gamma=0,

    learning_rate=0.05,

    max_depth=2,

    min_child_weight=3,

    n_estimators=200,

    subsample=1.0
)


ensemble = VotingClassifier(

    estimators=[

        ('svm', svm),

        ('xgb', xgb)
    ],

    voting='soft'
)


pipeline = Pipeline([

    (
        'scaler',
        StandardScaler()
    ),

    (
        'ensemble',
        ensemble
    )
])


y_pred = cross_val_predict(

    pipeline,

    X,
    y,

    cv=cv,

    n_jobs=-1
)

print("\n===================================")
print("CLASSIFICATION REPORT")
print("===================================")

print(

    classification_report(

        y,

        y_pred,

        target_names=['F', 'C']
    )
)


cm = confusion_matrix(
    y,
    y_pred
)

print("\n===================================")
print("CONFUSION MATRIX")
print("===================================")

print(cm)


accuracy = accuracy_score(
    y,
    y_pred
)

f1 = f1_score(
    y,
    y_pred
)

print("\n===================================")
print("FINAL METRICS")
print("===================================")

print("\nAccuracy:")
print(accuracy)

print("\nF1 Score:")
print(f1)