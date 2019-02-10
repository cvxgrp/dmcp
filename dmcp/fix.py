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

    # Create list of parameters
    variable_list = obj.variables()
    variable_list.sort(key = lambda x:x.id)
    param_list = []
    for var in variable_list:
        if var.sign == 'NONNEGATIVE':
            para = cvx.Parameter(shape = var.shape, nonneg=True)
            if var.value is not None:
                para.value = abs(var.value)
            para.id = var.id
            param_list.append(para)
        elif var.sign == 'NONPOSITIVE':
            para = cvx.Parameter(shape = var.shape, nonpos=True)
            if var.value is not None:
                para.value = -abs(var.value)
            para.id = var.id
            param_list.append(para)
        elif var.attributes['PSD'] == True:
            para = cvx.Parameter(shape = var.shape, PSD=True)
            if var.value is not None:
                para.value = var.value
            para.id = var.id
            param_list.append(para)
        else:
            para = cvx.Parameter(shape = var.shape)
            para.id = var.id
            param_list.append(para)
    
    param_list.sort(key = lambda x:x.id)
    if isinstance(obj,Expression):
        return fix_expr(obj,fix_vars, param_list)
    elif isinstance(obj,Problem):
        return fix_prob(obj,fix_vars, param_list)
    else:
        print("wrong type to fix")

def fix_prob(prob, fix_var, param_list):
    """Fix the given variables in the problem.

        Parameters
        ----------
        expr : Problem
        fix_var : List
            Variables to be fixed.
        params: : List
            List of parameters to replace variables from fix_var

        Returns
        -------
        Problem
        """
    new_cost = fix_expr(prob.objective.expr, fix_var, param_list)
    if prob.objective.NAME == 'minimize':
        new_obj = cvx.Minimize(new_cost)
    else:
        new_obj = cvx.Maximize(new_cost)
    new_constr = []
    for con in prob.constraints:
        fix_con = fix_expr(con.expr, fix_var, param_list)
        if isinstance(con, NonPos):
            new_constr.append(fix_con <= 0)
        elif isinstance(con, PSD):
            new_constr.append(fix_con >> 0)
        else:
            new_constr.append(fix_con == 0)
    new_prob = Problem(new_obj, new_constr)
    return new_prob

def fix_expr(expr, fix_var, param_list):
    """Fix the given variables in the expression.

        Parameters
        ----------
        expr : Expression
        fix_var : List
            Variables to be fixed.
        params : List
            List of parameters to replace variables from fix_var

        Returns
        -------
        Expression
    """
    fix_var_id = [var.id for var in fix_var]
    fix_var_id.sort()
    if isinstance(expr, Variable) and expr.id in fix_var_id:
        param = next((temp_param for temp_param in param_list if temp_param.id == expr.id), None)
        return param
    elif len(expr.args) == 0:
        return expr
    else:
        new_args = []
        for arg in expr.args:
            new_args.append(fix_expr(arg, fix_var, param_list))
        return expr.copy(args=new_args)
