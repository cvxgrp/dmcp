from sklearn import datasets
from sklearn.model_selection import train_test_split
import cvxpy as cvx
import dmcp


# Get the iris dataset
iris = datasets.load_iris()

X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.25, random_state=42)

print("Train")
for i in range(len(y_train)):
    print(X_train[i], y_train[i])

print("Test")
for i in range(len(y_test)):
    print(X_test[i], y_test[i])

print(len(X_train), len(X_test))
print(len(y_train), len(y_test))