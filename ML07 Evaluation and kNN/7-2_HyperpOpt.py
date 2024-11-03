"""
7-2_HyperpOpt.py
"""


################ Preliminaries

import matplotlib
matplotlib.use('Qt5Agg') # Sets the backend for matplotlob.
                         # Qt5Agg needs PyQt5 installed.
                         # Also, make sure to set the backend before importing pyplot.
from matplotlib import pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier



################ Load the data

#       We again use the diabetes dataset.
df = pd.read_csv('diabetes.csv')
#       Separate input and target.
X = df.drop(columns=['Outcome'])
y = df['Outcome'].values



################ Hyperparameter optimization (finding the optimal k)

# In the last lab, we chose k=10 for the knn. Yet, we don't know if this is a good choice.
# Maybe another choice of k would give a higher Recognition Rate.
# To find the best k, we simply
#   - fit different versions of knn (with different values of k),
#   - evaluate them, and
#   - chose the model with the best evaluation result (highest Recognition Rate).
# To do the evaluation, we can either use VSA or CV:
#   1. VSA has the advantage of being quicker,
#   2. CV has the advantage of being more trustworthy.
# As an exercise, we try both!
# To not make the lab too big, we only look at the unweighted knn.


###### 1. Using VSA

#   Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=23, stratify = y)

#   Calculate the test recognition rates for different values of k
train_RR = []   # initialize the list of training Recoginition Rates
test_RR = []    # initialize the list of test Recoginition Rates
neighbors_settings = range(1, 200, 2) # k = 1, 3, 5, ..., 199
for n in neighbors_settings:
    knn = KNeighborsClassifier(n_neighbors=n)               # initialize
    knn_model = knn.fit(X_train, y_train)                   # fit
    train_RR.append(knn_model.score(X_train, y_train))      # predict, calculate, and record training recognition rates
    test_RR.append(knn_model.score(X_test, y_test))         # predict, calculate, and record test recognition rates
    print("k =", n, "Test RR:", knn.score(X_test, y_test))
#   Plot the result
plt.plot(neighbors_settings, train_RR, label="training RR")
plt.plot(neighbors_settings, test_RR, label="test RR")
plt.ylabel("Recognition Rate")
plt.xlabel("n_neighbors")
plt.legend()
plt.show()

#   Choose best k
max_test_RR = max(test_RR) # highest RR
best_idx = test_RR.index(max_test_RR) # index that holds highest RR
best_k = list(neighbors_settings)[best_idx] # 'neighbors_settings' is a 'range' object.
                                           # To access its values, we need to first convert it to a list using list().
print("For knn, the optimal number of neighbours is", best_k, ".")
print("The test recognition rate is", round(max_test_RR*100, 1),"%.")

#   Train best model
#       - We fit the best model to the whole sample X.
#       - It is now ready to be deployed to make predictions on new data.
knn_best = KNeighborsClassifier(n_neighbors=best_k)  # initialize
knn_best_model = knn.fit(X, y)                       # fit


###### 2. Using CV

#   Since we use CV, we don't need a train/test split.
#   Instead, we use the whole sample X.

#   Calculate the CV recognition rates for different values of k
cv_RR = []
neighbors_settings = range(1, 200, 2) # k = 1, 3, 5, ..., 199
for n in neighbors_settings:
    knn = KNeighborsClassifier(n_neighbors=n)               # initialize
    knn_model = knn.fit(X_train, y_train)                   # fit
    scores = cross_val_score(knn, X, y, cv=10)              # predict and calculate the recognition rates per fold
    cv_RR.append(scores.mean())                             # calculate (and record) the cv recognition rate
    print("k =", n, "cv RR:", scores.mean())
#   Plot the result
plt.plot(neighbors_settings, cv_RR, label="cv RR")
plt.ylabel("Cross Validated Recognition Rate")
plt.xlabel("n_neighbors")
plt.legend()
plt.show()

#   Choose best k
max_cv_RR = max(cv_RR) # highest RR
max_cv_idx = cv_RR.index(max_cv_RR) # index that holds highest RR
best_cv_k = list(neighbors_settings)[max_cv_idx] # 'neighbors_settings' is a 'range' object.
                                           # To access its values, we need to first convert it to a list using list().
print("For knn, the optimal number of neighbours is", best_cv_k, ".")
print("The cross-validated recognition rate is", round(max_cv_RR*100, 1),"%.")

#   Train best model
#       - We fit the best model to the whole sample X.
#       - It is now ready to be deployed to make predictions on new data.
knn_best = KNeighborsClassifier(n_neighbors=best_cv_k)  # initialize
knn_best_model = knn.fit(X, y)                          # fit



################ Note
#
#   The code for the CV-based approach is a bit shorter than for the VSA-based approach,
#   but the computational effort is higher by factor 10:
#       - For each k = 1, 3, 5, ..., 199, CV performs 10 train/test iterations.
#       - So, all in all, the above code trains and evaluates (199+1)/2 * 10 = 1'000 models.
#       - In contrast, the VSA-based approach only trains and evaluates (199+1)/2 = 100 models.
#       - This is not a problem here, because our data set is so small. But with bigger data sets,
#         this can easily lead to a long computing time.
#       - This is true for all kinds of classifiers, but is particularly problematic for kNN, because it is a
#         lazy learner. It's computing time increases linearly with the size of the training set.
