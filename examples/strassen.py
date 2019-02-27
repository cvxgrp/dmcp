from cvxpy import *
import dmcp
import matplotlib.pyplot as plt
import numpy as np

np.random.seed(0)
n = 2
m = 8
p = 1
A = []
B = []
C = []
for i in range(p):
    A.append(np.random.randn(n,n))
    B.append(np.random.randn(n,n))
    C.append(np.dot(A[-1],B[-1]))

psi = Variable(n*n,m)
phi = Variable(m,n*n)
theta = Variable(m,n*n)
"""
phi.value = np.matrix([[1,0,0,1],
             [0,1,0,1],
             [1,0,0,0],
             [0,0,0,1],
             [1,0,1,0],
             [-1,1,0,0],
             [0,0,1,-1],
             [0,0,0,0]])
theta.value = np.matrix([[1,0,0,1],
               [1,0,0,0],
               [0,0,1,-1],
               [-1,1,0,0],
               [0,0,0,1],
               [1,0,1,0],
               [0,1,0,1],
               [0,0,0,0]])
psi.value = np.matrix([[1,0,0,1,-1,0,1,0],
             [0,0,1,0,1,0,0,0],
             [0,1,0,1,0,0,0,0],
             [1,-1,1,0,0,1,0,0]])
"""
phi.value = np.matrix([[1,0,0,0],
                       [0,0,1,0],
                       [1,0,0,0],
                       [0,0,0,1],
                       [0,1,0,0],
                       [0,0,0,1],
                       [0,1,0,0],
                       [0,0,0,1]])
theta.value = np.matrix([[1,0,0,0],
                         [0,1,0,0],
                         [0,1,0,0],
                         [0,0,0,1],
                         [1,0,0,0],
                         [0,1,0,0],
                         [0,0,1,0],
                         [0,0,0,1]])
psi.value = np.matrix([[1,1,0,0,0,0,0,0],
                       [0,0,1,1,0,0,0,0],
                       [0,0,0,0,1,1,0,0],
                       [0,0,0,0,0,0,1,1]])
obj = Minimize((norm(psi,1)+norm(phi,1)+norm(theta,1)))
constr = []
for i in range(p):
    constr.append(vec(C[i]) == psi*multiply(phi*vec(A[i]), theta*vec(B[i])))
prob = Problem(obj, constr)

prob.solve(method = 'bcd', update = 'minimize', mu = 2, mu_max = 1e5, max_iter = 100)
print "phi = ", phi.value
print "theta = ", theta.value
print "psi = ", psi.value