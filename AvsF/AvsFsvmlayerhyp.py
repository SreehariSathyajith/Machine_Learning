

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

from sklearn.svm import SVC

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


ranksum_df = pd.read_csv(ranksum_path)

significant_features = ranksum_df[
    ranksum_df['A_vs_F_significant'] == True
]['feature'].tolist()

print("\nSelected Features:")
print(significant_features)


X = df[significant_features]

y = df['group'].map({
    'A': 0,
    'F': 1
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


svm = SVC()


pipeline = Pipeline([

    ('scaler', StandardScaler()),

    ('svm', svm)
])

param_grid = {

    'svm__kernel': [
        'linear',
        'rbf',
        'poly'
    ],

    'svm__C': [
        0.001,
        0.01,
        0.1,
        1,
        10,
        50,
        100
    ],

    'svm__gamma': [
        'scale',
        'auto',
        0.0001,
        0.001,
        0.01,
        0.1,
        1
    ],

    'svm__degree': [
        2,
        3,
        4
    ],

    'svm__coef0': [
        0.0,
        0.1,
        0.5,
        1.0
    ],

    'svm__class_weight': [
        None,
        'balanced'
    ],

    'svm__shrinking': [
        True,
        False
    ],

    'svm__tol': [
        1e-2,
        1e-3,
        1e-4
    ]
}


grid_search = GridSearchCV(

    estimator=pipeline,

    param_grid=param_grid,

    scoring='f1',

    cv=cv,

    n_jobs=-1,

    verbose=2
)


grid_search.fit(X, y)


print("\n===================================")
print("BEST PARAMETERS")
print("===================================")

print(
    grid_search.best_params_
)

print("\nBest F1 Score:")
print(
    grid_search.best_score_
)


best_model = grid_search.best_estimator_


y_pred = cross_val_predict(

    best_model,

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
        target_names=['A', 'F']
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