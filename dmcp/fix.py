from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

__author__ = 'Xinyue'
import cvxpy as cvx
from cvxpy.expressions.expression import Expression
from cvxpy.problems.problem import Problem
from cvxpy.expressions.variable import Variable
from cvxpy.expressions.constants.parameter import Parameter
from cvxpy.constraints.nonpos import NonPos
from cvxpy.constraints.zero import Zero
from cvxpy.constraints.psd import PSD

def fix(obj, fix_vars):
    """
    Fix the given variables in the object
    :param obj: a problem or an expression
    :param fix_var: a list of variables
    :return: a problem or an expression
    """
    if isinstance(obj,Expression):
        return fix_expr(obj,fix_vars)
    elif isinstance(obj,Problem):
        return fix_prob(obj,fix_vars)
    else:
        print("wrong type to fix")

def fix_prob(prob, fix_var):
    """Fix the given variables in the problem.

        Parameters
        ----------
        expr : Problem
        fix_var : List
            Variables to be fixed.

        Returns
        -------
        Problem
        """
    new_cost = fix_expr(prob.objective.expr, fix_var)
    if prob.objective.NAME == 'minimize':
        new_obj = cvx.Minimize(new_cost)
    else:
        new_obj = cvx.Maximize(new_cost)
    new_constr = []
    for con in prob.constraints:
        fix_con = fix_expr(con.expr, fix_var)
        if isinstance(con, NonPos):
            new_constr.append(fix_con <= 0)
        elif isinstance(con, PSD):
            new_constr.append(fix_con >> 0)
        else:
            new_constr.append(fix_con == 0)
    new_prob = Problem(new_obj, new_constr)
    return new_prob


def fix_expr(expr, fix_var):
    """Fix the given variables in the expression.

        Parameters
        ----------
        expr : Expression
        fix_var : List
            Variables to be fixed.

        Returns
        -------
        Expression
    """
    fix_var_id = [var.id for var in fix_var]
    if isinstance(expr, Variable) and expr.id in fix_var_id:
        if expr.sign == "POSITIVE":
            para = cvx.Parameter(shape = expr.shape, nonneg=True)
            para.value = abs(expr).value
        elif expr.sign == "NEGATIVE":
            para = cvx.Parameter(shape = expr.shape, nonpost=True)
            para.value = -abs(expr).value
        else:
            para = cvx.Parameter(shape = expr.shape)
            para.value = expr.value
        return para
    elif len(expr.args) == 0:
        return expr
    else:
        new_args = []
        for arg in expr.args:
            new_args.append(fix(arg, fix_var))
        return expr.copy(new_args)
