import pandas as pd
import numpy as np

from scipy.stats import ranksums


features_path = (
    r"C:\Users\sreeh\ML_fiiting\Layerwise_Features.csv"
)

participants_path = (
    r"C:\Users\sreeh\ML_fiiting\participants.tsv"
)

output_path = (
    r"C:\Users\sreeh\ML_fiiting\Ranksum_Layerwise_Features.csv"
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

print("\nMerged dataset shape:")
print(df.shape)

print("\nGroup counts:")
print(df['group'].value_counts())

A = df[df['group'] == 'A']

C = df[df['group'] == 'C']

F = df[df['group'] == 'F']


exclude_cols = [
    'subject',
    'group'
]

features = [
    col for col in df.columns
    if col not in exclude_cols
]

print("\nTotal features:")
print(len(features))


results = []

for feature in features:

    print(f"\nTesting: {feature}")


    A_vals = (
        A[feature]
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )

    C_vals = (
        C[feature]
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )

    F_vals = (
        F[feature]
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )


    if len(A_vals) > 1 and len(C_vals) > 1:

        stat_ac, p_ac = ranksums(
            A_vals,
            C_vals
        )

    else:

        stat_ac = np.nan
        p_ac = np.nan


    if len(A_vals) > 1 and len(F_vals) > 1:

        stat_af, p_af = ranksums(
            A_vals,
            F_vals
        )

    else:

        stat_af = np.nan
        p_af = np.nan


    if len(F_vals) > 1 and len(C_vals) > 1:

        stat_fc, p_fc = ranksums(
            F_vals,
            C_vals
        )

    else:

        stat_fc = np.nan
        p_fc = np.nan


    sig_ac = (
        p_ac < 0.05
        if not np.isnan(p_ac)
        else False
    )

    sig_af = (
        p_af < 0.09
        if not np.isnan(p_af)
        else False
    )

    sig_fc = (
        p_fc < 0.05
        if not np.isnan(p_fc)
        else False
    )


    results.append({

        "feature": feature,

        "A_vs_C_p": p_ac,
        "A_vs_F_p": p_af,
        "F_vs_C_p": p_fc,


        "A_vs_C_stat": stat_ac,
        "A_vs_F_stat": stat_af,
        "F_vs_C_stat": stat_fc,


        "A_vs_C_significant": sig_ac,
        "A_vs_F_significant": sig_af,
        "F_vs_C_significant": sig_fc
    })


results_df = pd.DataFrame(results)


results_df["min_p"] = results_df[
    [
        "A_vs_C_p",
        "A_vs_F_p",
        "F_vs_C_p"
    ]
].min(axis=1)

results_df = results_df.sort_values(
    by="min_p"
)

results_df.to_csv(
    output_path,
    index=False
)

print("\n========================================")
print("SIGNIFICANT FEATURES")
print("========================================")

significant_features = []

for _, row in results_df.iterrows():

    if (
        row['A_vs_C_significant']
        or row['A_vs_F_significant']
        or row['F_vs_C_significant']
    ):

        significant_features.append(
            row['feature']
        )

        print("\nFeature:", row['feature'])

        print(
            "A vs C p-value:",
            row['A_vs_C_p']
        )

        print(
            "A vs F p-value:",
            row['A_vs_F_p']
        )

        print(
            "F vs C p-value:",
            row['F_vs_C_p']
        )


print("\n========================================")
print("SUMMARY")
print("========================================")

print(
    "\nTotal significant features:",
    len(significant_features)
)

print("\nSignificant feature names:")

for feat in significant_features:

    print("-", feat)

print("\n✅ Rank-Sum testing completed!")

print("\nResults saved to:")
print(output_path)