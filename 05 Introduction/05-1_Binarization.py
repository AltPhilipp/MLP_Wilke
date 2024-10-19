"""

05-1_Binarization.py
Transforming categorical to numerical.

"""

import pandas as pd


##### Import the census data set

#  Note: Please download 'census.data' from Moodle!

# We import the census data set using pd.read_csv()
#   - pd.read_csv() imports a .csv file as a data frame
#   - 'census.data' does not have a header that contains the column names. So we set header=None.
#   - With names = ... we specify the column names manually.
data = pd.read_csv('census.data', header=None, index_col=False,
                   names=['age', 'workclass', 'fnlwgt',
                            'education', 'education-num', 'marital- status', 'occupation',
                            'relationship', 'race', 'gender', 'capital-gain',
                            'capital-loss', 'hours-per-week', 'native-country', 'income'])

# Check that it is really a data frame
type(data)

# Select a subset of 4 variables
my_data = data[['age', 'workclass', 'gender', 'income']]



##### Check the data types of the 4 variables

# Display the structure of the data frame to see the data types
my_data.info()
    # - 'object' means categorical.
    # - in64 means numerical
    # Note: In Pandas, 'object' is a catch-all for any data type that doesn't fit into the standard numeric types.

# Display the summary statistics of the columns
my_data.describe()
    # Notice that only 'age' is displayed!
    # The reason is that 'describe()' summarizes only numerical variables by default.

#   To include a summary of the categorical variables, you must set the parameter include='all':
my_data.describe(include='all')
    # 'unique' shows the number of unique values of a categorical variable.
    # E.g., income has 2 unique values

# unique() prints the unique values of a categorical variable
my_data['workclass'].unique()
my_data['gender'].unique()
my_data['income'].unique()



##### Binarize the categorical variables
#   Remember: Binarization (One-Hot-Encoding) transforms categorical variables to numerical variables.
#   We use the pandas function 'get_dummies()' to do that. It leaves the numerical variables untouched
#   Remark: Binarized variables are often also called 'dummy variables'.
my_data_num = pd.get_dummies(my_data)

# Display the structure of the transformed data frame
my_data_num.info()
    # We see that we have many more variables now!
    # All variables are either integer or boolean.



##### Double-check what happened

# Inspect the new dummy variables again:
my_data_num.info()
    # 'age' is untouched
    # workclass: 9 dummies
    # gender: 2 dummies
    # income: 2 dummies

# This is indeed consistent with the unique values before transformation:
my_data.describe(include='all')
my_data['workclass'].unique()
my_data['gender'].unique()
my_data['income'].unique()
