# task2
'''
Find an optimized value for eps.
Plot the elbow and extract the optimal value from the plot
Apply DBSCAN again with the new eps value and confirm visually that clusters are proper

https://www.analyticsvidhya.com/blog/2020/09/how-dbscan-clustering-works/
https://machinelearningknowledge.ai/tutorial-for-dbscan-clustering-in-python-sklearn/

For both the datasets
Report the optimal value of eps in the Readme to your github project
Add the elbow plots to your github project Readme
Add the cluster plots to your github project Readme
'''
from pathlib import Path
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import KDTree
from sklearn.cluster import DBSCAN

# Ground levels from task1 used to filter out ground points
GROUND_Z1 = 61.30881333333333
GROUND_Z2 = 61.2680366666667

# Load 3D point cloud datasets
pcd1 = np.load(r"C:\Users\davdel0404\Downloads\AI Lund\Assignement 5\Assignment_5\dataset1.npy")
pcd2 = np.load(r"C:\Users\davdel0404\Downloads\AI Lund\Assignement 5\Assignment_5\dataset1.npy")
# Filter: Keep only objects above ground level
pcd1_above = pcd1[pcd1[:, 2] > GROUND_Z1]
pcd2_above = pcd2[pcd2[:, 2] > GROUND_Z2]

print(f"[Task2] dataset1 above-ground shape: {pcd1_above.shape}")
print(f"[Task2] dataset2 above-ground shape: {pcd2_above.shape}")


def k_distance_elbow(points, k=5, tag="dataset1", save_plot=True):
    #Calculate K distance curve and find the elbow
    #to estimate the optimal epsilon

    points = np.asarray(points)
    tree = KDTree(points)
    dists, _ = tree.query(points, k=k+1)
    kth = np.sort(dists[:, k])

    # Find knee via maximun deviation from the line

    x = np.arange(len(kth), dtype=float)
    y = kth.astype(float)
    x0, y0 = x[0], y[0]
    x1, y1 = x[-1], y[-1]
    num = np.abs((y1 - y0) * x - (x1 - x0) * y + x1*y0 - y1*x0)
    den = np.hypot(y1 - y0, x1 - x0) or 1.0
    idx_opt = int(np.argmax(num / den))
    eps_opt = float(y[idx_opt])

    # Elbow plot: save and show
    if save_plot:
        plt.figure(figsize=(8, 5))
        plt.plot(kth, linewidth=1)
        plt.scatter([idx_opt], [eps_opt], s=40, c="red")
        plt.title(f"k-distance (k={k}) – elbow={eps_opt:.3f} — {tag}")
        plt.xlabel("Points (sorted)")
        plt.ylabel(f"{k}:th nearest neighbor distance")
        plt.tight_layout()
        out = Path("images") / f"kdist_elbow_{tag}.png"
        plt.savefig(out, dpi=150)
        plt.show()
        print(f"[Task2] Sparade elbow-plot: {out}")

    return eps_opt

def run_dbscan_and_plot(points, eps, min_samples=5, tag="dataset1"):
    # Run DBSCAN and save a 2D plot (using the found eps)
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(points)
    labels = clustering.labels_
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    print(f"[Task2] {tag}: eps={eps:.3f}, kluster={n_clusters}")

    plt.figure(figsize=(9, 9))
    plt.scatter(points[:, 0], points[:, 1], c=labels, s=2, cmap="Spectral")
    plt.title(f"DBSCAN (eps={eps:.3f}) – {tag}\nKluster: {n_clusters}")
    plt.xlabel("x"); plt.ylabel("y")
    plt.tight_layout()
    out = Path("images") / f"dbscan_clusters_{tag}.png"
    plt.savefig(out, dpi=150)
    plt.show()
    print(f"[Task2] Saved cluster-plot: {out}")
    return n_clusters

# Dataset 1 
eps1 = k_distance_elbow(pcd1_above, k=5, tag="dataset1", save_plot=True)
run_dbscan_and_plot(pcd1_above, eps=eps1, min_samples=5, tag="dataset1")

# Dataset 2 
eps2 = k_distance_elbow(pcd2_above, k=5, tag="dataset2", save_plot=True)
run_dbscan_and_plot(pcd2_above, eps=eps2, min_samples=5, tag="dataset2")

print(f"[Task2] Optimal eps – dataset1: {eps1:.4f}")
print(f"[Task2] Optimal eps – dataset2: {eps2:.4f}")
