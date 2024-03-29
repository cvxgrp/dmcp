
DMCP
====
A multi-convex optimization problem is one in which the variables can be partitioned into sets, over each of which the problem is convex when the other variables are fixed.
It is generally a nonconvex problem.
DMCP package provides methods to verify multi-convexity and to find minimal sets of variables that have to be fixed for the problem to be convex, as well as an organized heuristic for multi-convex programming.
The full details of our approach are discussed in [the associated paper](http://stanford.edu/~boyd/papers/dmcp.html). DMCP is built on top of [CVXPY](http://www.cvxpy.org/), a domain-specific language for convex optimization embedded in Python.

Installation
------------
You should first install [CVXPY 1.0](http://www.cvxpy.org/). Then clone the repository and run ``python setup.py install`` inside.

DMCP rules
----------
Consider an optimization problem
```
minimize    f_0(x) 
subject to  f_i(x) <= 0, i = 1,...,m
            g_i(x) = 0, i = 1,...,p,
```
where variable ``x`` admits a partition of blocks of variables ``x = (x_1,...,x_N)``, and functions ``f_i`` for ``i = 0,...,m`` and ``g_i`` for ``i = 1,...,p`` are proper.
Given a set of DCP atomic functions and its extension of multi-convex atomic functions,
the problem can be specified as disciplined multi-convex programming (DMCP), if there are index sets ``F_1,..., F_K``, such that their intersection is empty, and for every ``k`` the problem with variables ``x_i`` for all ``i`` in set ``F_k`` fixed to any value can be specified as DCP with respect to the DCP atoms set.

Example
-------
The following code uses DMCP to approximately solve a simple multi-convex problem.
```
import cvxpy as cvx
import dmcp

x_1 = cvx.Variable()
x_2 = cvx.Variable()
x_3 = cvx.Variable()
x_4 = cvx.Variable()

objective = cvx.Minimize(cvx.abs(x_1*x_2+x_3*x_4))
constraint = [x_1+x_2+x_3+x_4 == 1]
myprob = cvx.Problem(objective, constraint)

print("minimal sets:", dmcp.find_minimal_sets(myprob))   # find all minimal sets
print("problem is DCP:", myprob.is_dcp())   # false
print("problem is DMCP:", dmcp.is_dmcp(myprob))  # true
result = myprob.solve(method = 'bcd')
```
The output of the above code is as follows.
```
minimal sets: [[1, 3], [1, 2], [0, 3], [0, 2]]
problem is DCP: False
problem is DMCP: True
maximum value of slack variables: 1.15081491391e-05
objective value: 1.74866042578e-05
```

The solutions obtained by DMCP can depend on the initial point.
The algorithm starts from the values of any variables that are already specified; 
for any that are not specified, random values are used.
It is suggested that users set reasonable initial values for all variables,
which can be done by manually setting the ``value`` field of the problem variables.
For example:
```
x_1.value = 1.2
x_2.value = -3
x_3.value = 4
x_4.value = 0.15
result = myprob.solve(method = 'bcd')
```
More examples can be found [here] (https://github.com/cvxgrp/dmcp/tree/master/examples).

Multi-convex atomic functions
-----------------------------
In order to allow multi-convex functions, we extend the atomic function set of ``CVXPY``.
The following atoms are allowed to have non-constant expressions in both arguments, while in the dictionary of ``CVXPY`` the first argument must be constant.
* multiplication: ``expression1 * expression2``
* elementwise multiplication: ``cvx.multiply(expression1, expression2)``

Functions and attributes
----------------
* ``is_dmcp(problem)`` returns a boolean indicating if an optimization problem satisfies DMCP rules.
* ``find_minimal_sets(problem)`` analyzes the problem and returns a list of minimal sets of (indexes of) variables.
The indexes are with respect to the list ``problem.variables()``, namely the variable corresponding to the index ``0`` is
``problem.variables()[0]``. If the problem is DCP, it returns an empty list. ``is_all = True`` will generate all minimal sets.
* ``fix(obj, fix_vars)`` returns a new expression or a new problem with the variables in the list ``fix_vars`` replaced with parameters of the same dimensions and signs. The ``obj`` can either be an expression or a problem.

Constructing and solving problems
---------------------------------
The components of the variable, the objective, and the constraints are constructed using standard CVXPY syntax. Once the user has constructed a problem object, they can apply the following solve method:
* ``problem.solve(method = 'bcd')`` applies the solving algorithm with proximal operators, and returns the number of iterations, and the maximum value of the slack variables. The solution to every variable is in its ``value`` field.
* ``problem.solve(method = 'bcd', update = 'minimize')`` applies the solving method without proximal operators.
* ``problem.solve(method = 'bcd', update = 'prox_linear')`` applies the solving method with prox-linear operators.

Additional arguments can be used to specify the parameters.

Solve method parameters:
* The ``solver`` parameter specifies what solver to use to solve convex subproblems.
* The ``max_iter`` parameter sets the maximum number of iterations in the algorithm. The default is 100.
* The ``mu`` parameter trades off satisfying the constraints and minimizing the objective. Larger ``mu`` favors satisfying the constraints. The default is 0.001.
* The ``rho`` parameter sets the rate at which ``mu`` increases inside the algorithm. The default is 1.2.
* The ``mu_max`` parameter upper bounds how large ``mu`` can get. The default is 1e4.
* The ``lambd`` parameter is the parameter in the proximal operator. The default is 10.

If the convex solver for the subproblems accepts any additional keyword arguments, 
then the user can set them in the ``problem.solve()`` function, and they will be passed to the convex solver.
