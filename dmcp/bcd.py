from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

__author__ = 'Xinyue'

import cvxpy as cvx
from dmcp import rand_initial
from dmcp import find_minimal_sets
from dmcp import fix
import numpy as np
from cvxpy.constraints.nonpos import NonPos
from cvxpy.constraints.zero import Zero
from cvxpy.constraints.psd import PSD

def is_dmcp(obj):
    """
    :param obj: an obj
    :return: a boolean indicating if the obj (Function, Expression, Variable) is DMCP
    """
    for var in obj.variables():
        fix_var = [avar for avar in obj.variables() if not avar.id == var.id]
        if not fix(obj,fix_var).is_dcp():
            return False
    return True


def bcd(prob, max_iter = 100, solver = 'SCS', mu = 5e-3, rho = 1.5, mu_max = 1e5, ep = 1e-3, lambd = 10, update = 'proximal'):
    """
    call the solving method
    :param prob: a problem
    :param max_iter: maximal number of iterations
    :param solver: DCP solver
    :param mu: initial value of parameter mu
    :param rho: increasing factor for mu
    :param mu_max: maximal value of mu
    :param ep: precision in convergence criterion
    :param lambd: parameter lambda
    :param update: update method
    :return: it: number of iterations; max_slack: maximum slack variable
    """
    # check if the problem is DMCP
    if not is_dmcp(prob):
        print("problem is not DMCP")
        return None
    # check if the problem is dcp
    if prob.is_dcp():
        print("problem is DCP")
        prob.solve()
    else:
        fix_sets = find_minimal_sets(prob)
        flag_ini = 0
        for var in prob.variables():
            if var.value is None: # check if initialization is needed
                flag_ini = 1
                rand_initial(prob)
                break
        # set update option
        if update == 'proximal':
            proximal = True
            linearize = False
        elif update == 'minimize':
            proximal = False
            linearize = False
        elif update == 'prox_linear':
            proximal = True
            linearize = True
        else:
            print("no such update method")
            return None
        result = _bcd(prob, fix_sets, max_iter, solver, mu, rho, mu_max, ep, lambd, linearize, proximal)
        # print result
        print("======= result =======")
        print("minimal sets:", fix_sets)
        if flag_ini:
            print("initial point not set by the user")
        print("number of iterations:", result[0]+1)
        print("maximum value of slack variables:", result[1])
        print("objective value:", prob.objective.value)
        print("Value of the variables: ", [var.value for var in prob.variables()])
        return result

def _bcd(prob, fix_sets, max_iter, solver, mu, rho, mu_max, ep, lambd, linear, proximal):
    """
    block coordinate descent
    :param prob: Problem
    :param max_iter: maximum number of iterations
    :param solver: solver used to solved each fixed problem
    :return: it: number of iterations; max_slack: maximum slack variable
    """
    obj_pre = np.inf
    for it in range(max_iter):
        np.random.shuffle(fix_sets)
        #print "======= iteration", it, "======="
        for subset in fix_sets:
            problem_variables = prob.variables()
            problem_variables.sort(key = lambda x:x.id)
            fix_var = [prob.variables()[idx] for idx in subset]
            # fix variables in fix_set
            fixed_p = fix(prob,fix_var)
            # linearize
            if linear:
                fixed_p_obj = cvx.linearize(fixed_p.objective.expr)
                fixed_p_constr = fixed_p.constraints
                if fixed_p.objective.NAME == 'minimize':
                    fixed_p = cvx.Problem(cvx.Minimize(fixed_p_obj), fixed_p_constr)
                else:
                    fixed_p = cvx.Problem(cvx.Maximize(fixed_p_obj), fixed_p_constr)
            # add slack variables
            fixed_p, var_slack = add_slack(fixed_p, mu)
            # proximal operator
            if proximal:
                fixed_p = proximal_op(fixed_p, var_slack, lambd)
            # solve
            fixed_p.solve(solver = solver)
            max_slack = 0
            if not var_slack == []:
                max_slack = np.max([np.max(cvx.abs(var).value) for var in var_slack])
                print("max abs slack =", max_slack, "mu =", mu, "original objective value =", prob.objective.args[0].value, "fixed objective value =",fixed_p.objective.args[0].value, "status=", fixed_p.status)
            else:
                print("original objective value =", prob.objective.args[0].value, "status=", fixed_p.status)
        mu = min(mu*rho, mu_max) # adaptive mu
        if np.linalg.norm(obj_pre - prob.objective.args[0].value) <= ep and max_slack<=ep: # quit
            return it, max_slack
        else:
            obj_pre = prob.objective.args[0].value
    return it, max_slack

def linearize(expr):
    """Returns the tangent approximation to the expression.
    Gives an elementwise lower (upper) bound for convex (concave)
    expressions. No guarantees for non-DCP expressions.
    Args:
        expr: An expression.
    Returns:
        An affine expression.
    """
    if expr.is_affine():
        return expr
    else:
        tangent = expr.value
        if tangent is None:
            raise ValueError(
        "Cannot linearize non-affine expression with missing variable values."
            )
        grad_map = expr.grad
        for var in expr.variables():
            if var.is_matrix():
                flattened = np.transpose(grad_map[var])*cvx.vec(var - var.value)
                tangent = tangent + cvx.reshape(flattened, *expr.shape)
            else:
                if var.shape[1] == 1:
                    tangent = tangent + np.transpose(grad_map[var])*(var - var.value)
                else:
                    tangent = tangent + (var - var.value)*grad_map[var]
        return tangent

def add_slack(prob, mu):
    """
    Add a slack variable to each constraint.
    For leq constraint, the slack variable is non-negative, and is on the right-hand side
    :param prob: a problem
    :param mu: weight of slack variables
    :return: a new problem with slack vars added, and the list of slack vars
    """
    var_slack = []
    new_constr = []
    for constr in prob.constraints:
        constr_shape = constr.expr.shape
        if isinstance(constr, NonPos):
            var_slack.append(cvx.Variable(constr_shape, nonneg=True)) # NonNegative slack var
            left = constr.expr
            right = var_slack[-1]
            new_constr.append(left<=right)
        elif isinstance(constr, PSD):
            var_slack.append(cvx.Variable((), nonneg=True)) # NonNegative slack var
            left = constr.expr + var_slack[-1]*np.eye(constr_shape[0])
            new_constr.append(left>>0)
        else: # equality constraint
            var_slack.append(cvx.Variable(constr_shape))
            left = constr.expr
            right = var_slack[-1]
            new_constr.append(left==right)
    new_cost = prob.objective.args[0]
    if prob.objective.NAME == 'minimize':
        for var in var_slack:
            new_cost  =  new_cost + cvx.norm(var,1)*mu
        new_prob = cvx.Problem(cvx.Minimize(new_cost), new_constr)
    else: # maximize
        for var in var_slack:
            new_cost  =  new_cost - cvx.norm(var,1)*mu
        new_prob = cvx.Problem(cvx.Maximize(new_cost), new_constr)
    return new_prob, var_slack

def proximal_op(prob, var_slack, lambd):
    """
    proximal operator of the objective
    :param prob: problem
    :param var_slack: list of slack variables
    :param lambd: proximal operator parameter
    :return: a problem with proximal operator
    """
    new_cost = prob.objective.expr
    new_constr = prob.constraints
    slack_id = [var.id for var in var_slack]

    #Add proximal variables
    prob_variables = prob.variables()
    prob_variables.sort(key = lambda x:x.id)
    for var in prob_variables:
        # add quadratic terms for all variables that are not slacks
        if not var.id in slack_id:
            if prob.objective.NAME == 'minimize':
                new_cost = new_cost + cvx.square(cvx.norm(var - var.value,'fro'))/2/lambd
            else:
                new_cost = new_cost - cvx.square(cvx.norm(var - var.value,'fro'))/2/lambd

    # Define proximal problem
    if prob.objective.NAME == 'minimize':
        new_prob = cvx.Problem(cvx.Minimize(new_cost), new_constr)
    else: # maximize
        new_prob = cvx.Problem(cvx.Maximize(new_cost), new_constr)
    return new_prob

cvx.Problem.register_solve("bcd", bcd)