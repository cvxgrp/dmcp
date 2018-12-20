from sklearn import datasets
from sklearn.model_selection import train_test_split
import numpy as np
import cvxpy as cvx
import dmcp


# Get the iris dataset
iris = datasets.load_iris()
data_set = iris.data
test_labels = iris.target

# Number of classes
num_classes = len(set(iris.target))
num_examples = len(iris.target)

# Define Multivariate Normal Distribution optimization variables
mean_array = []
precision_array = []
for i in range(num_classes):
    mean_array.append(cvx.Variable((4,1)))
    precision_array.append(cvx.Variable((4,4), PSD=True))

# Define categorical distribution optimization variables
categorical_array = []
for i in range(num_examples):
    conditional_array = []
    for j in range(num_classes):
        conditional_array.append(cvx.Variable((1,1)))
    categorical_array.append(conditional_array)

# Create objective function


print(mean_array)
print(precision_array)
print(len(categorical_array))