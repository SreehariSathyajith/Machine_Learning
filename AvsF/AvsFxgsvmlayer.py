

import pandas as pd
import numpy as np

from sklearn.model_selection import (
    StratifiedKFold,
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
    df['group'].isin(['A', 'F'])
]

print("\nDataset shape:")
print(df.shape)

print("\nClass counts:")
print(df['group'].value_counts())


ranksum_df = pd.read_csv(ranksum_path)

significant_features = ranksum_df[
    ranksum_df['A_vs_F_significant'] == True
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

    'A': 1,

    'F': 0
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


    C=10,

    class_weight=None,

    coef0=0.0,

    degree=3,

    gamma=1,

    kernel='poly',

    shrinking=True,

    tol=0.001,

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

    gamma=0.3,

    learning_rate=0.1,

    max_depth=3,

    min_child_weight=2,

    n_estimators=200,

    subsample=0.7
)


ensemble = VotingClassifier(

    estimators=[

        ('svm', svm),

        ('xgb', xgb)
    ],

    voting='soft'
)

pipeline = Pipeline([

    ('scaler', StandardScaler()),

    ('ensemble', ensemble)
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

        target_names=['F', 'A']
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