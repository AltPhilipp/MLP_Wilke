"""
6-1_HierClust.py
"""

################ Preliminaries

import matplotlib
matplotlib.use('Qt5Agg') # Sets the backend for matplotlob.
                         # Qt5Agg needs PyQt5 installed.
                         # Also, make sure to set the backend before importing pyplot.
from matplotlib import pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import dendrogram
from scipy.cluster.hierarchy import single, complete


################ Create a synthetic data set

# Create synthetic dataset with 4 "blobs" (clusters)
#
#    - n_samples=1000. Specify that we want to generate 1000 data objects.
#    - centers=4. Specify that we want 4 cluster centers (and thus 4 clusters)
#    - n_features=2. Specify number of features.
#    - random_state=42. Set a seed for reproducability.
#
#    Note:
#    - We intentionally only create a 2-dimensional data set.
#    - This way we can visualize it in a scatterplot, and later visually check our clustering results.
#    - This is useful in the beginning to get a feeling of what's happening.

X, y = make_blobs(n_samples=1000, centers=4, n_features=2, random_state=42)

# Check the data structure in a scatterplot
plt.scatter(X[:, 0], X[:, 1])
plt.xlabel("Feature 0")
plt.ylabel("Feature 1")
plt.show()

################ Preprocessing

# Note:
#    - It is not necessary to do a train/test split, since our data set does not contain examples of good clusters!
#    - Thus, a "test" set would not help us to test anything.
#    - Instead, the only way to evaluate the clustering quality is to use, e.g., the Dunn Index or the Silouette Score,
#      to measure if the clustering matches the natural structure of the data well.


################ Fit (learn) a hierarchical clustering model

#   1. Initialize
#      Create an instance of the function AgglomerativeClustering() by specifying the following parameters:
#      - linkage="complete". Choose the compete linkage distance as a clsuter distance.
#      - n_clusters=4. Specify that we want 4 clusters as a result.
#
#      Note 1:
#      - We choose 4 clusters, because we know that the synthetic data set we created has 4 blobs.
#      - This way we can easily check if the algorithm works well in finding the "natural structure" (the 4 blobs).
#
#      Note 2:
#      - The function AgglomerativeClustering() applies a hierarchical clustering approach.
#      - Specifying that we want 4 cluster "cuts" the resulting dendogram at an appropriate level to get 4 clusters.
#      - Remember that the result is always an exclusive clustering (full and non-overlapping).
#
agg = AgglomerativeClustering(linkage="complete", n_clusters=4)

#   2. Fit and Predict
#
#      Note:
#      - Most traditional machine learning models in scikit-learn, such as linear regression and decision trees
#        have separate fit and predict methods. You first call fit(X_train, y_train) to train the model, then you
#        can call predict(X_test) to make predictions on new data.
#      - Yet, some models, particularly clustering algorithms provide a fit_predict method.
#        This method combines the fitting and predicting steps into one call.
#      - For clustering, "prediction" is not really a prediction, but rather means assigning to each point its
#        cluster label. You can think of it as adding an additional column (feature) "cluster label" to your data set.
#
assignment = agg.fit_predict(X)

################ Visually check the clustering result

# Add the cluster assignment as color to the scatter plot
#    - We see that our clustering indeed matches the 4 blobs nicely
#    - You can run the clustering again with a different number of clusters (e.g., n_clusters=4) and check the result!
plt.scatter(X[:, 0], X[:, 1], c=assignment)
plt.xlabel("Feature 0")
plt.ylabel("Feature 1")
plt.show()

################ Hyper-Parameter Tuning with the Silhouette Score

# Typically, the correct number of clusters in an application is not known: it is a hyperparameter.
#    - The following code fragment tests various hierarchical clusterings with different
#      numbers of clusters and calculates the silhouette score for each clustering.
#    - We see in the results that a clustering with 4 clusters indeed has the highest Silhouette Score.

# Find optimal number of clusters
for k in range(2, 8):
     agg = AgglomerativeClustering(linkage="complete", n_clusters=k)
     assignment = agg.fit_predict(X)
     score = silhouette_score(X, assignment)
     print("K =", k, "Silhouette Score =", score)

################ Plotting the Dendogram

# Visualizing a dendrogram is not (yet?) possible in Scikit-Learn.
#    - Instead, we can use the function dendrogram() from scipy.cluster.hierarchy.
#    - Yet, first we must re-create our hierarchical clustering using also a scipy function:
#      We can either use complete() or single(). Both are from scipy.cluster.hierarchy.

# We use the function complete(), because we want to use the complete-linkage distance (same as above).
#    - It returns an array that specifies the cluster distances.
linkage_array = complete(X)

# Now we can plot the dendrogram using the function dendogram():
dendrogram(linkage_array)
plt.xlabel("Data Object")
plt.ylabel("Cluster distance")
plt.show()

# We see we can cut the dendogram at hight (cluster distance) 10 to get 4 clusters.
#    - To visualize the 4 clusters better, we can use the parameter color_threshold:
#    - color_threshold=10 specifies that the subclusters below height 10 should be colored differently.
dendrogram(linkage_array, color_threshold=10)
plt.xlabel("Data Object")
plt.ylabel("Cluster distance")
plt.show()
