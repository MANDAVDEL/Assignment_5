'''
Task 1 (3)
- Find the best value for the ground level
- One way: use a histogram (np.histogram)
- Update the function get_ground_level() with your changes

The README should include the ground level for both datasets and the histogram images.
'''
# %%
import os
import numpy as np
import matplotlib.pyplot as plt

def get_ground_level(pcd, nbins=300, low_prct=20, high_prct=80, save_plot=False, tag="dataset"):

    """
    Estimates the ground level (z) as the midpoint of the histogram bin with the most points within the lower part of the height distribution.

    pcd       : Nx3 point cloud (x, y, z)  
    nbins     : number of bins in the histogram  
    low_prct  : lowest percentile
    high_prct : upper percentile for trimming  
    save_plot : if True, save the histogram  
 
    """
    z = pcd[:, 2].astype(float)

    # Focus on lower elevations where the ground is likely located.

    low = np.percentile(z, low_prct)
    high = np.percentile(z, high_prct)
    z_window = z[(z >= low) & (z <= high)]
    if z_window.size == 0:
        z_window = z  # fallback om urvalet blev tomt

    # Histogram -> select the bin with the most points (mode)

    counts, edges = np.histogram(z_window, bins=nbins)
    k = int(np.argmax(counts))
    ground_level = 0.5 * (edges[k] + edges[k + 1])

    # save histogram
    if save_plot:
        os.makedirs("images", exist_ok=True)
        plt.figure(figsize=(8, 4))
        plt.hist(z, bins=nbins)
        plt.axvline(ground_level, linestyle="--")
        plt.title(f"Ground level (z) histogram – {tag}")
        plt.xlabel("z")
        plt.ylabel("antal punkter")
        plt.tight_layout()
        plt.savefig(f"images/ground_hist_{tag}.png", dpi=150)
        plt.show()

    return float(ground_level)

pcd1 = np.load("C:/Users/davdel0404/Downloads/AI Lund/Assignement 5/Lidar_assignment-1/dataset1.npy")
print("Dataset1 ranges:")
print("x:", np.min(pcd1[:,0]), "→", np.max(pcd1[:,0]))
print("y:", np.min(pcd1[:,1]), "→", np.max(pcd1[:,1]))
print("z:", np.min(pcd1[:,2]), "→", np.max(pcd1[:,2]))
g1 = get_ground_level(pcd1, save_plot=True, tag="dataset1")
print("Dataset1: Ground level =", g1)

pcd2 = np.load("C:/Users/davdel0404/Downloads/AI Lund/Assignement 5/Lidar_assignment-1/dataset2.npy")
print("Dataset2 ranges:")
print("x:", np.min(pcd2[:,0]), "→", np.max(pcd2[:,0]))
print("y:", np.min(pcd2[:,1]), "→", np.max(pcd2[:,1]))
print("z:", np.min(pcd2[:,2]), "→", np.max(pcd2[:,2]))
g2 = get_ground_level(pcd2, save_plot=True, tag="dataset2")
print("Dataset2: Ground level =", g2)
