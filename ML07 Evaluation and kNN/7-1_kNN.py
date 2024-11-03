"""
7-1_kNN.py
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

#       We import the diabetes dataset (see github). It originally stems from Kaggle.
#       It holds data on diabetes tests. Our goal is to predict the column 'Outcome'.
#       Since 'diabetes' is a csv file, we can use read_csv() from pandas to load it as a data frame.
df = pd.read_csv('diabetes.csv')
print(df)
#       We need to separate input and target.
X = df.drop(columns=['Outcome']) # Use the method .drop for data frames to drop the target column ''
y = df['Outcome'].values


################ Fit (learn) a kNN model

##  Data preprocessing
#  - We want to use VSA (the Validation Set Approach) for the evaluation later.
#  - Thus, we must split our sample in training and test set before we fit the model.
#  - To do that, we can use the function train_test_split() from sklearn.model_selection:
#      - test_size=0.4 specifies that the split ratio for trin/test set is 60/40.
#      - random_state=23 sets the seed for the pseudo-randomizer to 23 (arbitrary, but fixed number).
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=23, stratify = y)

##  Fit the model (aka "learn" or "train" the model)
#   We fit the kNN model on the training set. To do that, we apply the initialize-fit-predict process.
#
#   1. Initialize
#       Create an instance of the KNeighborsClassifier() by specifying the following parameter:
#       - n_neighbors = 10. The 'n_neighbors' parameters specifies the number of neighbours k that kNN uses
#         for the majority vote. As a first attempt, we choose k=10.
knn = KNeighborsClassifier(n_neighbors=10)
#
#   2. Fit
knn_model = knn.fit(X_train, y_train)
#       Note:
#       - Normally, in this step, the learning algorithm derives a model from the training data that is independent
#         of the training data. E.g., the DecisionTreeClassifier() algorithm learns the tree structure from the
#         training data. The training data and the algorithm are not stored as part of the model object, because
#         they are not needed any more when we apply the model to make predictions.
#       - This is different for kNN, because kNN is a lazy learner. That means that we cannot separate training data,
#         learning algorithm and model.
#       - Thus, when we look into the model object, we see that the training set and the algorithm are stored in it:
#           - '_fit_X' is an attribute of the model object 'knn_model'. It stores the input features of the training data.
#           - '_y' is an attribute of the model object 'knn_model'. It stores the class values of the training data.
#           - 'n_neighbors' is an attribute of the model object 'knn_model'. It stores the number of neighbours used to
#                           make the majority vote for prediction. (This is all we need to know to do the majority vote!)
knn_model._fit_X
knn_model._y
knn_model.n_neighbors
#       - Thus, when we "fit" the kNN algorithm to the training data, we only store the training data
#         and the decision rule in an object that we call "the model object".
#
#   3. Predict
#       When we "apply the model" to the test set to make predictions, we use 'n_neighbors', '_fit_X' and '_y'
#       from the model object to do the majority vote.
y_pred = knn_model.predict(X_test)



################ Fit (learn) a weighted kNN model

##  Data preprocessing
#  - We reuse the train/test split from above.

##  Fit (learn/train) the model
#
#   1. Initialize
#       We can use the same algorithm KNeighborsClassifier() with an additional parameter:
#       - n_neighbors = 10.
#       - weights='distance'. The neighbours according are weighed according to their distance to the new object.
knn_w = KNeighborsClassifier(n_neighbors=10, weights='distance')
#
#   2. Fit
knn_w_model = knn_w.fit(X_train, y_train)
#
#   3. Predict
y_pred_w = knn_w_model.predict(X_test)



################ Compare the prediction performance of both models using VSA

# Above, we have trained both models on the training set.
# Now, we evaluate both models on the test set.
# To do that, we use the method score(). It internally preforms 3 steps:
#       1. Use the model stored in the model object (knn_model, knn_w_model) to predict the target values of the test data.
#       2. Compare the predictions with the true values.
#       3. From the comparison, calculate the Recognition Rate.
knn_model.score(X_test, y_test)     # Evaluate the knn model.
knn_w_model.score(X_test, y_test)   # Evlautes the weighted knn model.



################ Compare the prediction performance of both models using CV

#   We use CV with 10 folds ("10-fold cross-validation").
#   CV does the train/test splits automatically for each fold.
#   Thus, we don't need to do a train/test split manually. Instead, we apply CV to the whole sample X.
#   To use CV, we can use the function cross_val_score from sklearn.model_selection with the following parameters:
#       - the model to evaluate (knn_model or knn_w_model)
#       - the data set to use for CV (our whole sample X)
#       - the corresponding class values y
#       - the number of folds to use (cv=5).
knn_scores = cross_val_score(knn_model, X, y, cv=10)     # use the knn model
knn_w_scores = cross_val_score(knn_w_model, X, y, cv=10) # use the weighted knn model
#   The result is a list of Recognition Rates (one per fold):
print("Recognition Rate per fold", knn_scores)
print("Recognition Rate per fold", knn_w_scores)
#
#   To get the cross-validated Recognition Rate, we need to calculate the mean of the above scores:
print("Cross-validated Recognition Rate:", knn_scores.mean())
print("Cross-validated Recognition Rate:", knn_w_scores.mean())




