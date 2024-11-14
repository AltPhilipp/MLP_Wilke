"""

2-1_Discretization.py
Transforming numerical to categorical.

"""

import sklearn.datasets as ds
import sklearn.preprocessing as pre
import pandas as pd


##### Generate a synthetic data set

# ds.make_classification() generates a synthetic data for a classification task
#   - n_samples = 10 data objects (rows)
#   - n_features = 3 features (columns)
#   - n_redudandant = 0 (no features should be redundant, e.g., shouldn't repeat a number)
#   - n_classes = 2 (the target variable should have 2 classes)
#   - The function returns the input features X and the target variable y as matrices.
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
#       Create an instance of the KBinsDiscretizer() class by specifying the parameters:
#       - n_bins = 3 (we want to have 3 intervals/bin)
#       - strategy = 'uniform' (we want to use Equal Width Binning)
#       - encode = 'ordinal' (the interval identifiers are encoded as integer values)
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
#       Maps the original numerical values to the corresponding bins.
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
# The KBins*Discretizer* only *discretizes* the numerical values, it does not make them categorical!

# Note:
#   - Discrete variables with a numerical data type can be both, categorical scale or numerical scale!
#   - Which of them applies depends on the semantic of the attributes, because the semantic tells us
#     which mathematical operations make sense.
#   - Example: 'age'
#       - The data type of 'age' (measured in years) is integer, thus numerical.
#       - The scale level of 'age' is ratio, because it makes sense to calculate age differences and fractions.
#         Thus, the scale level is also numerical.
#       - Both, data type and scale level are numerical.
#   - Example: 'age class (EWB)'
#       - Assume we have 10 age classes of equal length.
#       - Assume the classes are encoded as integers: 1 = [1,10], 2 = [11,20], 3 = [21-30], ..., 10=[91,100]
#       - The data type of 'age class (EWB)' is integer, thus numerical.
#       - The scale level of 'age class (EWB)' is interval, because it makes sense calculate differences of age classes.
#         Thus, the scale level is also numerical.
#       - Both, data type and scale level are numerical.
#   - Example: 'age class (EFB)'
#       - Assume we have 4 age classes of equal frequency and unequal length.
#       - Assume the classes are encoded as integers: 1 = [1,20], 2 = [21,25], 3 = [25-30], 4=[41-100].
#       - The data type of 'age class (EFB)' is integer, thus numerical.
#       - The scale level of 'age class (EFB)' is ordinal: it does NOT make sense to calculate differences of age classes,
#         but it makes sense to put them into the above order.
#         Thus, the scale level is categorical.
#       - The data type is numerical while the scale level is numerical.

# Important note:
#   - Algorithms cannot know the semantic of attributes. Thus, they infer the scale level from the data type.
#   - If scale level and data type don't match, you might get nonsensical results!


##### How to make sure an algorithm infers a categorical scale after discretization?

# To achieve this, we can convert the bin identifiers to string:
X_ewb_cat = X_ewb.astype(str)

# If we want nicer strings, we can first convert to integer, then to string:
X_ewb_cat = X_ewb.astype(int).astype(str)

# If we need a date frame, we can convert the ndarray to DataFrame using the method .DataFrame() from pandas:
X_ewb_cat_df = pd.DataFrame(X_ewb_cat)

# Did it work?
type(X_ewb_cat_df) # It's indeed a data frame now
X_ewb_cat_df.info() # All features are categorical ('object')

# We can assign new bin labels
    # Look up the old labels:
X_ewb_cat_df[0].unique() # we only check feature 0, because we know that all features have the same bin labels
    # Define a dictionary for mapping old values to new values:
rename_mapping = {
    '2': 'High',
    '1': 'Meduim',
    '0': 'Low'
}
    # Use the pandas method replace() to rename them:
X_ewb_cat_df = X_ewb_cat_df.replace(rename_mapping)

# We can also change the feature names
X_ewb_cat_df.columns = ["Temperature", "Income", "Mood"]

print(X_ewb_cat_df)