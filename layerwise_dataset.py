import os
import numpy as np
import pandas as pd
import networkx as nx

from networkx.algorithms.community import (
    greedy_modularity_communities
)

from networkx.algorithms.community.quality import (
    modularity
)

input_folder = r"C:\Users\sreeh\ML_fiiting\Supra_Adjacency"

output_file = (
    r"C:\Users\sreeh\ML_fiiting\Layerwise_Features.csv"
)

N = 19
L = 5
EPS = 1e-10

layer_names = [
    "delta",
    "theta",
    "alpha",
    "beta",
    "gamma"
]

results = []

def get_block(A, layer):

    start = layer * N
    end = (layer + 1) * N

    return A[start:end, start:end]

files = [
    f for f in os.listdir(input_folder)
    if f.endswith(".csv")
]

for file in files:

    print(f"\nProcessing: {file}")

    file_path = os.path.join(
        input_folder,
        file
    )

    subject = file.replace(
        "_supra.csv",
        ""
    )

    A = pd.read_csv(
        file_path,
        header=None
    ).values

    A = np.nan_to_num(A)

    subject_result = {
        "subject": subject
    }

    for l in range(L):

        layer_name = layer_names[l]

        layer = get_block(A, l)

        layer_no_diag = layer.copy()

        np.fill_diagonal(
            layer_no_diag,
            0
        )

        layer_abs = np.abs(layer_no_diag)

        n = layer.shape[0]

        mean_connectivity = (
            np.sum(layer_no_diag)
            /
            (n * (n - 1))
        )

        density = (
            np.count_nonzero(layer_no_diag)
            /
            (n * (n - 1))
        )

        total_strength = np.sum(layer_abs)

        upper_triangle = layer_abs[
            np.triu_indices(n, k=1)
        ]

        std_edge_weight = np.std(
            upper_triangle
        )

        max_edge_weight = np.max(
            upper_triangle
        )

        min_edge_weight = np.min(
            upper_triangle
        )

        weights = upper_triangle[
            upper_triangle > 0
        ]

        prob = (
            weights
            /
            (np.sum(weights) + EPS)
        )

        entropy = -np.sum(
            prob * np.log(prob + EPS)
        )

        G = nx.from_numpy_array(
            layer_abs
        )

        strengths = np.array([

            d for _, d in G.degree(
                weight='weight'
            )
        ])

        mean_strength = strengths.mean()

        std_strength = strengths.std()

        clustering_vals = np.array(list(

            nx.clustering(
                G,
                weight='weight'
            ).values()
        ))

        mean_clustering = (
            clustering_vals.mean()
        )

        std_clustering = (
            clustering_vals.std()
        )

        average_clustering = (
            nx.average_clustering(
                G,
                weight='weight'
            )
        )

        bet = np.array(list(

            nx.betweenness_centrality(
                G,
                weight='weight'
            ).values()
        ))

        betweenness_mean = bet.mean()

        betweenness_std = bet.std()

        eig = np.array(list(

            nx.eigenvector_centrality_numpy(
                G,
                weight='weight'
            ).values()
        ))

        eigenvector_mean = eig.mean()

        eigenvector_std = eig.std()

        A_dist = np.where(

            layer_abs > EPS,

            1 / layer_abs,

            0
        )

        G_dist = nx.from_numpy_array(
            A_dist
        )

        global_efficiency = (
            nx.global_efficiency(
                G_dist
            )
        )

        try:

            characteristic_path_length = (
                nx.average_shortest_path_length(
                    G_dist,
                    weight='weight'
                )
            )

        except:

            characteristic_path_length = np.nan

        communities = (
            greedy_modularity_communities(G)
        )

        modularity_value = modularity(

            G,
            communities,
            weight='weight'
        )

        transitivity = nx.transitivity(G)

        try:

            assortativity = (
                nx.degree_pearson_correlation_coefficient(
                    G,
                    weight='weight'
                )
            )

        except:

            assortativity = np.nan

        subject_result.update({

            f"{layer_name}_mean_connectivity":
                mean_connectivity,

            f"{layer_name}_density":
                density,

            f"{layer_name}_total_strength":
                total_strength,

            f"{layer_name}_std_edge_weight":
                std_edge_weight,

            f"{layer_name}_max_edge_weight":
                max_edge_weight,

            f"{layer_name}_min_edge_weight":
                min_edge_weight,

            f"{layer_name}_entropy":
                entropy,

            f"{layer_name}_mean_strength":
                mean_strength,

            f"{layer_name}_std_strength":
                std_strength,

            f"{layer_name}_mean_clustering":
                mean_clustering,

            f"{layer_name}_std_clustering":
                std_clustering,

            f"{layer_name}_average_clustering":
                average_clustering,

            f"{layer_name}_betweenness_mean":
                betweenness_mean,

            f"{layer_name}_betweenness_std":
                betweenness_std,

            f"{layer_name}_eigenvector_mean":
                eigenvector_mean,

            f"{layer_name}_eigenvector_std":
                eigenvector_std,

            f"{layer_name}_global_efficiency":
                global_efficiency,

            f"{layer_name}_path_length":
                characteristic_path_length,

            f"{layer_name}_modularity":
                modularity_value,

            f"{layer_name}_transitivity":
                transitivity,

            f"{layer_name}_assortativity":
                assortativity
        })

    results.append(subject_result)

df = pd.DataFrame(results)

df.to_csv(
    output_file,
    index=False
)

print("\n✅ Layerwise feature extraction completed!")

print(f"\nSaved to:\n{output_file}")