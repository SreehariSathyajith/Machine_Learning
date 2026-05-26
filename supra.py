import os
import numpy as np
from scipy.io import loadmat

input_path = r'C:\Users\sreeh\ML_PROJECT\EEG_Bands_Output'
output_path = r'C:\Users\sreeh\ML_PROJECT\Supra_Adjacency'

os.makedirs(output_path, exist_ok=True)

bands = ['delta', 'theta', 'alpha', 'beta', 'gamma']

subjects = [d for d in os.listdir(input_path)
            if os.path.isdir(os.path.join(input_path, d))]

for subj in subjects:

    print(f"\nProcessing: {subj}")

    subj_path = os.path.join(input_path, subj)

    signals = []

    for band in bands:
        file_path = os.path.join(subj_path, f"{band}.mat")

        if not os.path.exists(file_path):
            print(f"⚠ Missing: {file_path}")
            continue

        data = loadmat(file_path)['filtered_data']  
        signals.append(data)

    if len(signals) != 5:
        print("⚠ Skipping subject (incomplete bands)")
        continue

    n_layers = len(signals)
    n_nodes = signals[0].shape[0]  # 19
    N = n_layers * n_nodes

    supra_adj = np.zeros((N, N))

    for i in range(n_layers):
        start = i * n_nodes
        end = start + n_nodes

        corr_mat = np.corrcoef(signals[i])

        # handle NaNs (important for stability)
        corr_mat = np.nan_to_num(corr_mat)

        supra_adj[start:end, start:end] = corr_mat

    for i in range(n_layers - 1):

        start1 = i * n_nodes
        start2 = (i + 1) * n_nodes

        for ch in range(n_nodes):

            sig1 = signals[i][ch, :]
            sig2 = signals[i + 1][ch, :]

            w = np.corrcoef(sig1, sig2)[0, 1]

            if np.isnan(w):
                w = 0

            supra_adj[start1 + ch, start2 + ch] = w
            supra_adj[start2 + ch, start1 + ch] = w

    save_path = os.path.join(output_path, f"{subj}_supra.csv")
    np.savetxt(save_path, supra_adj, delimiter=',')

    print(f"✅ Saved: {save_path}")

print("\n🎉 All subjects processed successfully!")