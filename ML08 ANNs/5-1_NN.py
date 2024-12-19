"""
5-1_NN.py
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

# We import the diabetes (originally from Kaggle) using pandas
df = pd.read_csv('diabetes.csv')

# Separate input features from target variable
X = df.drop(columns=['Outcome'])
y = df['Outcome'].values

# Split the data to training and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify = y, test_size=0.4, random_state=23)

# We use the neural network classifier MLPClassifier.
# (MLP stands for Multi Layer Perceptron)
mlp = MLPClassifier() # initialize
mlp.fit(X_train, y_train) # fit
predictions = mlp.predict(X_test) # predict
print(predictions)
print("Accuracy:", mlp.score(X_test, y_test)) # evaluate (Recognition Rate)

# Remark:
#   We have instantiated this classifier without parameters!
#   The default values for some important parameters are the following:
#   - The learning rate can be changed via parameter learning_rate_init=0.01 (default value is 0.001).
#   - The activation function can be changed via the parameter activation='relu' (default value is 'relu').
#     The possible values are 'identity', 'logistic', 'tanh', 'relu'.
#   - The maximum number of epochs can be controlled via parameter max_iter=100 (default value is 200).
#   - The architecture of the network can be defined via the hidden_layer_sizes parameter.
#     By default, the architecture of the network is defined with a hidden layer with 100 neurons.
#     This can be changed, for example, as follows:
#           - A hidden layer with 10 neurons: hidden_layer_sizes=(10)
#           - Two hidden layers with 30 and 20 neurons: hidden_layer_sizes=(30, 20)
#   - With random_state=2, you can make your results reproducible.

