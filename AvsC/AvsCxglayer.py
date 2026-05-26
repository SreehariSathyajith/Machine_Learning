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
    df['group'].isin(['A', 'C'])
]

print("\nDataset shape:")
print(df.shape)

print("\nClass counts:")
print(df['group'].value_counts())


ranksum_df = pd.read_csv(ranksum_path)


significant_features = ranksum_df[
    ranksum_df['A_vs_C_significant'] == True
]['feature'].tolist()

print("\n===================================")
print("SELECTED FEATURES")
print("===================================")

for feat in significant_features:
    print(feat)

print("\nTotal selected features:")
print(len(significant_features))


X = df[significant_features]


y = df['group']

y = y.map({
    'A': 1,
    'C': 0
})


X = X.replace(
    [np.inf, -np.inf],
    np.nan
)

X = X.fillna(
    X.mean()
)


corr_matrix = X.corr().abs()

upper = corr_matrix.where(
    np.triu(
        np.ones(corr_matrix.shape),
        k=1
    ).astype(bool)
)

to_drop = [

    column for column in upper.columns

    if any(
        upper[column] > 0.90
    )
]

X = X.drop(
    columns=to_drop
)

print("\nRemoved correlated features:")
print(len(to_drop))

print("\nRemaining features:")
print(X.shape[1])

print("\nRemaining feature names:")
print(X.columns.tolist())


cv = StratifiedKFold(

    n_splits=5,

    shuffle=True,

    random_state=42
)


xgb = XGBClassifier(

    objective='binary:logistic',

    eval_metric='logloss',

    random_state=42,


    colsample_bytree=1.0,

    gamma=0,

    learning_rate=0.1,

    max_depth=4,

    min_child_weight=1,

    n_estimators=100,

    subsample=0.8
)


pipeline = Pipeline([

    (
        'scaler',
        StandardScaler()
    ),

    (
        'xgb',
        xgb
    )
])


y_pred = cross_val_predict(

    pipeline,

    X,
    y,

    cv=cv
)


print("\n===================================")
print("CLASSIFICATION REPORT")
print("===================================")

print(

    classification_report(

        y,

        y_pred,

        target_names=['A', 'C']
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