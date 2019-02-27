__author__ = 'Xinyue'

import numpy as np
import cvxpy as cvx
import dmcp

n = 4
m = 3

theta = cvx.Variable(n)
theta.value = np.ones(n)/float(n)
a = cvx.Parameter(1)
x = cvx.Variable(m)
x.value = np.ones(m)/float(m)
P = cvx.Variable((m,m))
P.value = np.ones((m,m))
P0 = []
P0.append(np.matrix([[0.2,0.7,0.1],[0.3,0.3,0.4],[0.1,0.8,0.1]]))
P0.append(np.matrix([[0.25,0.35,0.4],[0.1,0.1,0.8],[0.45,0.05,0.5]]))
P0.append(np.matrix([[0.1,0.25,0.65],[0.2,0.1,0.7],[0.3,0.3,0.4]]))
P0.append(np.matrix([[0.23,0.67,0.1],[0.01,0.24,0.75],[0.2,0.45,0.35]]))

x0 = [0.25,0.3,0.45]
cost = cvx.norm(x-x0)
constr = [theta >= 0, cvx.sum(theta) == 1, x >= 0, cvx.sum(x) == 1, P*x == x]
right = 0
for i in range(n):
    right += theta[i]*np.transpose(P0[i])
constr += [P == right]
prob = cvx.Problem(cvx.Minimize(cost), constr)
prob.solve(method = 'bcd', ep = 1e-5)

print x.value
print theta.value