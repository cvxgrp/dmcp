__author__ = 'Xinyue'

import cvxpy as cvx
import numpy as np
import matplotlib.pyplot as plt
import dmcp

m = 10
n = 20
T = 20
times = 1
alpha_value = np.logspace(-5,0,50)

Error_random = np.zeros((1,len(alpha_value)))
Cardinality_random = np.zeros((1,len(alpha_value)))

for t in range(times):
    X = np.random.randn(m,T)

    D = cvx.Variable((m,n))
    Y = cvx.Variable((n,T))
    alpha = cvx.Parameter(nonneg=True)
    cost = cvx.square(cvx.norm(D*Y-X,'fro'))/2+alpha*cvx.norm(Y,1)
    obj = cvx.Minimize(cost)
    prob = cvx.Problem(obj,[cvx.norm(D,'fro')<=1])

    err = []
    card = []
    err_random = []
    card_random = []
    for a_value in alpha_value:
        D.value = None
        Y.value = None
        alpha.value = a_value
        prob.solve(method = 'bcd', ep = 1e-1, rho = 2, max_iter = 100)
        err_random.append(cvx.norm(D*Y-X,'fro').value/cvx.norm(X,'fro').value)
        card_random.append(cvx.sum(cvx.abs(Y).value>=1e-3).value)
        print "======= solution ======="
        print "objective =", cost.value
    Error_random += np.array(err_random).flatten()/float(times)
    Cardinality_random += np.array(card_random).flatten()/float(times)

plt.plot(Cardinality_random, Error_random, 'b o')
plt.xlabel('Cardinality of $Y$')
plt.ylabel('$||D*Y-X||_F/||X||_F$')
plt.show()
