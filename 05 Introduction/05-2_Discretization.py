"""

2-1_Discretization.py
Transforming numerical to categorical.

"""

import sklearn.datasets as ds
import sklearn.preprocessing as pre
import pandas as pd


##### Generate a synthetic data set

# ds.make_classification() generates a synthetic data for a classification task
#   - n_samples = 10 data objects
#   - n_features = 3 features
#   - n_redudandant = 0 (no features should be redundant)
#   - n_classes = 2 (the target variable should have 2 classes)
#   - The function returns the input features X and the target variable y.
X, y = ds.make_classification(n_samples=10, n_features=3, n_redundant=0, n_classes=2)

# Print the input feature X
print(X)
print(y)

# X and y are numpy arrays
type(X)
type(y)

# The underlying data types are numerical (float64)
X.dtype
y.dtype


##### Equal Width Binning

# We use pre.KBinsDiscretizer()
#
#   The "initialize-fit-transform" process
#       In scikit-learn, the process of data preprocessing often involves a sequence of steps
#       that can be summarized as "initialize-fit-transform". This sequence is commonly used
#       with various preprocessing classes, such as scalers, encoders, and transformers.
#
#   1. Initialize
#       Create an instance of the KBinsDiscretizer class by specifying the parameters:
#       - n_bins = 3 (we want to have 3 intervals/bin)
#       - strategy = 'uniform' (we want to use Equal Width Binning)
#       - encode = 'ordinal' (the interval identifiers are encoded as an integer values)
ewb = pre.KBinsDiscretizer(n_bins=3, strategy='uniform', encode='ordinal')
#
#   2. Fit
#       Calculate the bin edges based on the specified number of bins (3) and the strategy used ('uniform').
#       The bin edges are stored in the attribute 'bin_edges_' of 'ewb'.
ewb.fit(X)
print(ewb.bin_edges_)
#       For each of the 3 features, we get 4 bin edges (thus 3 bins)
#       Remember that we stored 3 features in X - ewb.fit() was applied to each of them separately.
#
#   3. Transform
#       Maps the numerical values stored in each feature in X to the corresponding bins.
X_ewb = ewb.transform(X)



##### Equal Frequency Binning
#   To use it, we must again follow the "initialize-fit-transform" process.
#
#   1. Initialize
#       Create an instance of the KBinsDiscretizer() class by specifying the parameters:
#       - n_bins = 3 (we want to have 3 intervals/bin)
#       - strategy = 'quantile' (we want Equal Frequency Binning now)
#       - encode = 'ordinal' (the interval identifiers are encoded as an integer values)
efb = pre.KBinsDiscretizer(n_bins=3, strategy='quantile', encode='ordinal')
#
#   2. Fit
#       Calculate the bin edges.
efb.fit(X)
print(efb.bin_edges_)

#   3. Transform
#       Map the values accordingly.
X_efb = efb.transform(X)




##### Are the results on a categorical scale?


# The KBinsDiscretizer encodes the bin identifiers as floats 0.,1.,2.:
type(X_ewb) # numpy.ndarray
X_ewb.dtype # dtype('float64')
# This means that the data type is still numerical.
# We *discretized* the numerical values, but we did not make them *categorical*!


##### How to make them categorical?

# If we want to make the discrete bin identifiers categorical, we can convert them to string:
X_ewb_cat = X_ewb.astype(str)

# If we want nicer strings, we can first convert to integer, then to string:
X_ewb_cat = X_ewb.astype(int).astype(str)

# If we need a date frame, convert using the method DataFrame from pandas:
X_ewb_cat_df = pd.DataFrame(X_ewb_cat)

# Double-check the data type:
type(X_ewb_cat_df) # It's indeed a data frame now
X_ewb_cat_df.info() # All features are categorical ('object')

# If we want to have different bin labels, we can replace the unique values of the categorical features:
    # Look up the old unique values:
X_ewb_cat_df[0].unique() # we only check feature 0 (we know that all are the same)
    # Define a dictionary for mapping old values to new values:
rename_mapping = {
    '2': 'High',
    '1': 'Meduim',
    '0': 'Low'
}
    # Use the pandas method replace() to rename them:
X_ewb_cat_df = X_ewb_cat_df.replace(rename_mapping)

# If we want to change the feature names also:
X_ewb_cat_df.columns = ["Temperature", "Income", "Mood"]

