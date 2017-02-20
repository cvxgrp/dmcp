__author__ = 'Xinyue'

from cvxpy import *
import dmcp
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import dft

np.random.seed(0)

n = 4
K = 2
#A = np.random.randn(n,n)
A = dft(n)

A_fac = []
for k in range(K):
    A_fac.append(Variable(n,n))
lambd = Parameter(1,sign = 'Positive')

cost =  0
#cost_new = 0
A_rec = np.eye(n)
for k in range(K):
 #   cost_new += norm(A_fac[k],1)
    cost += lambd*norm(A_fac[k],1)
    A_rec = A_rec*A_fac[k]
cost += square(norm(A-A_rec,"fro"))
#constr = [square(norm(A-A_rec,"fro")) <= lambd]
obj = Minimize(cost)
prob = Problem(obj)
#lambd_V = [0.06,0.08,0.1,0.15,0.2,0.22,0.24,0.26,0.28,0.3,0.31]#
lambd_V = np.linspace(0.01,1.8,20)
#lambd_V = 30*np.logspace(-3,-1,20)
err = []
card = []
for k in range(K):
    card.append([])

for i in range(0,len(lambd_V),1):
    lambd.value = lambd_V[i]
    prob.solve(method = 'bcd',ep = 1e-3, max_iter = 500, update = 'proximal')
    err.append(square(norm(A-A_rec,"fro")).value/square(norm(A,"fro")).value)
    print "relative error:", err[-1]
    for k in range(K):
        #card[k].append(norm(A_fac[k],1).value)
        max_k = norm(A_fac[k],"inf").value
        if max_k >= 1e-4:
            card[k].append(np.sum(np.abs(A_fac[k].value)/max_k>=0.01))
        else:
            card[k].append(0)
        print "number of non-zeros:",card[k][-1]
        print norm(A_fac[k], "inf").value
c = ['b-o','r:s','k-*','g-^','m->']
for k in range(K):
    plt.semilogy(card[k],err,c[k])
plt.xlabel("cardinality")
plt.ylabel("relative error")
#plt.legend(["$A_1$","$A_2$","$A_3$"])
plt.show()