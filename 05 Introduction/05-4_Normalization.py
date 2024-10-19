"""

05-4_Normalization.py
Transforming the data so that all features have the same scale.

"""

import sklearn.datasets as ds
import sklearn.preprocessing as pre
import pandas as pd
import matplotlib.pyplot as plt

##### Load an integrated data set

# We load the iris dataset, which is included in scikit-learn.
#   Every dataset in scikit-learn has its own function for loading
#   E.g. the data set breast_cancer is loaded with the function load_breast_cancer().
#   We load the iris data set
iris = ds.load_iris()

# 'iris' is a bunch:
type(iris)
# A bunch is a subclass of a dictionary. Characteristics:
#   - Dictionary-like: It stores key-value pairs, similar to a dictionary.
#   - Attribute Access: You can access the values using dot notation (e.g., bunch.key)
#     in addition to the standard dictionary access (e.g., bunch['key']).

# Print all keys of the iris bunch
keys = iris.keys()
for name in keys:
    print(name)

# The key „data“ stores the input features as a numpy array:
print(iris.data)
type(iris.data)

# The key 'target' stores the target variable as a numpy array:
print(iris.target)
type(iris.target)

# The key 'feature_names'stores the names of the features as a list:
print(iris.feature_names)
type(iris.feature_names)

# We will use only the input features of iris. Let's store them in the variable X:
X = iris.data



##### 0-1-Normalization
#
# 0-1-Normalization is also called min-max-Normalization

# We use the MinMaxScaler() from the preprocessing module of sklearn.
#
#   To use it, we follow the "initialize-fit-transform" process.
#
#   1. Initialize
#       Create an instance of the MinMaxScaler().
#       We don't need to specify parameters here.
min_max = pre.MinMaxScaler()
#
#   2. Fit
#       Calculate the minimum and maximum value for each feature in X.
min_max.fit(X)
#
#   3. Transform
#       Scale the features in X accordingly.
X_min_max = min_max.transform(X)


##### Mu-Sigma method:

# We use the StandardScaler() from the preprocessing module of sklearn.
#
#   To use it, we follow the "initialize-fit-transform" process.
#
#   1. Initialize
#       Create an instance of the StandardScaler().
#       We don't need to specify parameters here.
mu_sigma = pre.StandardScaler()
#
#   2. Fit
#       Calculate Mu (the mean) and Sigma (the standard deviation) values for each feature in X.
mu_sigma.fit(X)
#
#   3. Transform
#       Scale the features in X accordingly.
X_mu_sigma = mu_sigma.transform(X)


##### Check if it worked

# Create a data frame
#   Data frames are easier to handle for data analysis than ndarrays.
#   We create a data frame from the original data X .
#   We take the columns labels from 'iris.feature_names'.
X_df = pd.DataFrame(X, columns=iris.feature_names)

# We do the same for the transformed data.
X_min_max_df = pd.DataFrame(X_min_max, columns=iris.feature_names)
X_mu_sigma_df = pd.DataFrame(X_mu_sigma, columns=iris.feature_names)

# We can now display the statistical summaries of the data sets:
X_df.describe()
X_min_max_df.describe() # min = 0 and max =1
X_mu_sigma_df.describe() # mean = 0

# We can also create a scatterplot to see the result visually
#
#   Plotting involves 2 steps:
#   1. Creating the plot in the background.
#   2. Rendering the plot to display it.
#
#   1. Creating the plot
#       The 'plot' method in Pandas plots directly from DataFrames.
#       - kind='scatter' creates a scatter plot.
#       We can check any pair of input features - let's look at 'petal length' vs 'petal width'
#       - x='petal length (cm)'
#       - y='petal width (cm)'
plot_X = X_df.plot(x='petal length (cm)', y='petal width (cm)', kind='scatter')
#       We set the axes limits of the plot to the same lengths so that we see the scales of the features
plot_X.set_xlim(0, 10)  # Set x-axis limits
plot_X.set_ylim(0, 10)  # Set y-axis limits
#
#   2. Render the plot
#       plt.show() is a function in Matplotlib that renders the plot.
#       block=True blocks the further execution of your script until the plot window is closed by the user.
plt.show(block=True)

# Now let's look at the scaled data:
plot_X = X_min_max_df.plot(x='petal length (cm)', y='petal width (cm)', kind='scatter')
plt.show(block=True)
plot_X = X_mu_sigma_df.plot(x='petal length (cm)', y='petal width (cm)', kind='scatter')
plt.show(block=True)




