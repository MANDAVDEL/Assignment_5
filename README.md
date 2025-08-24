Assignment_5

This repository contains task and code for "assignment 5"
It includes datasets, python scripts, results and images.
Task 1
Task1 is a mapp containing the code, results and histograms plots.
The steps to solved the task was: loads a 3D point cloud dataset,
estimates ground level by analyzing the distribution of z values (hights)
and saves histogram plots. 
How ground level is estimated?
1. extract z values
2. Trim the distribution, I tested to only keep the lower part of the height
   distribution (1-40 percentile) to th reduce the influence of tall objects.
3. Create a histogram of the selected z values.
   The bin with the highest number of point represents the most common ground height
   
   Dataset1 ranges:
x: 0.14499999998952262 → 99.26600000000326
y: 80.00899999961257 → 159.9989999998361
z: 60.57800000000003 → 79.769
    Dataset1: Ground level = 61.30881333333333
   
    Dataset2 ranges:
x: 0.0 → 69.80100000000675
y: 0.0030000004917383194 → 79.99799999967217
z: 59.762 → 79.80799999999999
    Dataset2: Ground level = 61.2680366666667
