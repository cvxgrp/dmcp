from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

__author__ = 'Xinyue'

from dmcp.fix import fix
from dmcp.utils import is_atom_multiconvex
import numpy as np
from cvxpy.expressions.leaf import Leaf
from cvxpy.expressions.variable import Variable
import cvxpy as cvx

def find_minimal_sets(prob, is_all = False):
    """
    find minimal sets to fix
    :param prob: a problem
    :param is_all: to find all minimal sets or not
    :return: result: the list of minimal sets,
    each is a set of indexes of variables in prob.variables()
    """
    if prob.is_dcp():
        return []
    maxsets = find_MIS(prob, is_all)
    for sets in maxsets:
        print("Hi", [var.id for var in sets])
    result = []
    Vars = prob.variables()
    for maxset in maxsets:
        maxset_id = [var.id for var in maxset]
        maxset_id.sort()
        fix_id = [var.id for var in Vars if var.id not in maxset_id]
        V = [var.id for var in Vars]
        V.sort()
        fix_idx = [V.index(varid) for varid in fix_id]
        result.append(fix_idx)
    return result

def find_MIS(prob, is_all):
    """
    find maximal independent sets of a graph until all vertices are included
    :param prob: a problem
    :param is_all: to find all minimal sets or not
    :return: a list of maximal independent sets
    """
    if prob.is_dcp():
        return [prob.variables()]
    # graph of conflict vars
    V = prob.variables()
    node_num = len(V)
    g = np.zeros((node_num,node_num)) # table of edges
    varid = [var.id for var in V]
    varid.sort()
    stack, g = search_conflict_l(prob.objective.expr,[],varid,g)
    for con in prob.constraints:
        stack, g = search_conflict_l(-con.expr,[],varid,g)
    # find all independent sets of the conflict graph
    i_subsets = find_all_iset(V,g)

    # sort all independent sets
    subsets_len = [len(subset) for subset in i_subsets]
    sort_idx = np.argsort(subsets_len) # sort the subsets by card
    
    # check all sorted sets
    result = []
    U = [] # union of all collected vars

    # collecting from a subset with the largest cardinality
    for count in range(1,len(sort_idx)+1):
        flag = 1
        for subs in result:
            if is_subset(i_subsets[sort_idx[-count]], subs): # the current one is a subset of a previously collected one
                flag = 0
                break
        if flag:
            set_id = [var.id for var in i_subsets[sort_idx[-count]]]
            fix_set = [var for var in V if var.id not in set_id]
            if fix(prob, fix_set).is_dcp():
                result.append(i_subsets[sort_idx[-count]])
                U = union(U, i_subsets[sort_idx[-count]])
        
        # break if the collected vars cover all vars
        if not is_all and is_subset(V,U):
            break
    return result

def find_all_iset(V,g):
    """
    find all independent subsets, including the empty set
    :param V: vertex set
    :param g: graph
    :return: a list of independent subsets
    """
    subsets = find_all_subsets(V)
    result = []
    V_id = [var.id for var in V]
    V_id.sort()
    for subset in subsets:
        subset_id = [var.id for var in subset]
        subset_id.sort()
        subset_ind = [V_id.index(i) for i in subset_id]
        var_set_ind = [i for i in range(len(V_id))]
        set_complement = list(set(var_set_ind).difference(set(subset_ind)))
        if is_independent(set_complement, g) and not set_complement == []:
            result.append(subset)
    return result

def is_independent(s,g):
    """
    if a subset of vertices is independent on a graph
    :param s: a subset of vertices represented by indices
    :param g: graph
    :return: boolean
    """
    if sum([g[i,j] for i in s for j in s]) == 0:
        return True
    else:
        return False

def find_all_subsets(s):
    """
    find all subsets of a set, except for the empty set
    :param s: a set represented by a list
    :return: a list of subsets
    """
    subsets = []
    N = np.power(2,len(s))
    for n in range(N-1): # each number represent a subset
        subset = [] # the subset corresponding to n
        binary_ind = np.binary_repr(n+1) # binary
        for idx in range(1,len(binary_ind)+1): # each bit of the binary number
            if binary_ind[-idx] == '1': # '1' means to add the element corresponding to that bit
                subset.append(s[-idx])
        subsets.append(subset)
    return subsets

def search_conflict_l(expr,stack,V,t):
    '''
    search conflict variables in an expression using lists
    :param expr: an expression
    :param stack: stack of lists
    :param V: a list of id numbers of variables
    :param t: graph corresponding to the variables that can't be optimized together
    :return:
    '''
    if isinstance(expr,Leaf):
        if isinstance(expr,Variable):
            stack.append([expr.id])
        else:
            stack.append([])
    else:
        args_num = 0 # number of arguments
        for arg in expr.args:
            stack,t = search_conflict_l(arg,stack,V,t)
            args_num += 1
        if not is_atom_multiconvex(expr):        # at a convex node
            while args_num>1:
                stack[-2] = stack[-1] + stack[-2] # merge lists of its arguments
                args_num -= 1
                stack = stack[0:-1]
        else:                                     # at a multi-convex node (with two arguments)
            stack[-1] = list(set(stack[-1]))      # remove duplicates
            stack[-2] = list(set(stack[-2]))
            for i in range(len(V)):               # write conflict graph
                if V[i] in stack[-1]:
                    for j in range(len(V)):
                         if V[j] in stack[-2]:
                             t[i,j] = 1
                             t[j,i] = 1
            stack[-2] = stack[-1] + stack[-2]    # merge lists
            stack = stack[0:-1]
    return stack,t

def union(set1, set2):
    """
    the union of set1 and set2
    :param set1: a list of vars
    :param set2: a list of vars
    :return: a list of vars with no duplicates
    """
    return list(set(set1+set2))

def is_subset(var_set1, var_set2):
    """
    Checks if var_set2 is a subset of var_set1
    :param var_set1: a list of variables
    :param var_set2: a list of variables
    :return: a boolean indicating if var_set1 is a subset of var_set2
    """
    if var_set2 == []:
        return False
    if var_set1 == []:
        return True
    var_set1 = list(set(var_set1)) #remove duplicates
    var_set2 = list(set(var_set2))
    if len(var_set2) == len(list(set(var_set1+ var_set2))):
        return True
    else:
        return False

def search_conflict(expr,t,varid):
    """
    search conflict variables in an expression
    :param expr: expression
    :param t: a table recording the conflict pairs
    :param varid: id of all vars in table t
    :return: table t
    """
    for arg in expr.args:
        t = search_conflict(arg,t,varid)
    if is_atom_multiconvex(expr) and not expr.args[0].is_constant() and not expr.args[1].is_constant():
        id1 = [var.id for var in expr.args[0].variables()] # var ids in left child node
        id2 = [var.id for var in expr.args[1].variables()]
        index1 = [varid.index(vi) for vi in id1] # table index in left child node
        index2 = [varid.index(vi) for vi in id2]
        for i in index1:
            for j in index2:
                t[i,j] = 1
                t[j,i] = 1
    return t

def is_intersect(set1, set2):
    """
    if the intersection of set1 and set2 is nonempty
    :param set1: a list of vars
    :param set2: a list of vars
    :return: boolean
    """
    set1 = list(set(set1))
    set2 = list(set(set2))
    if len(set1)+len(set2) == len(list(set(set1+set2))):
        return False
    else:
        return True

def find_maxset_prob(prob,vars,current=[]):
    """
    Analyze a problem to find maximal subsets of variables,
    so that the problem is dcp restricting on each subset
    :param prob: Problem
    :return: a list of subsets of Variables, or None
    """
    if prob.is_dcp():
        return [prob.variables()]
    result = []
    next_level = []
    for var in vars:
        vars_active = erase(vars,var) # active variables
        if vars_active == []:  # an empty list indicates that the problem is not multi-convex
            return None
        # if the set of active vars is not a subset of the current result
        if all([not is_subset(vars_active, current_set) for current_set in current]):
            vars_active_id = [var.id for var in vars_active]
            fix_vars_temp = [var for var in prob.variables() if not var.id in vars_active_id]
            if fix(prob,fix_vars_temp).is_dcp() == True:
                result.append(vars_active) # find a subset
                current.append(vars_active)
            else:
                next_level.append(vars_active) # to be decomposed in the next level
    for set in next_level:
        result_temp = find_maxset_prob(prob,set,current)
        if result_temp is None:
            return None
        else:
            for set in result_temp:
                result.append(set)
    return result

def find_dcp_maxset(expr,vars,current=[]):
    """
    find maximal subsets of variables, so that expr is a dcp expression within each subset
    :param expr: an expression
    :param vars: variables that are not fixed
    :param current: current list of subsets
    :return: a list of subsets of variables and each subset is a list, or None
    """
    if expr.is_dcp():
        return [expr.variables()]
    result = []
    next_level = []
    for var in vars:
        vars_active = erase(vars,var) # active variables
        if vars_active == []:  # an empty list indicates that the expression is not multi-dcp
            return None
        # if the set of active vars is not a subset of the current result
        if all([not is_subset(vars_active, current_set) for current_set in current]):
            vars_active_id = [var.id for var in vars_active]
            fix_vars_temp = [var for var in expr.variables() if not var.id in vars_active_id]
            if fix(expr,fix_vars_temp).is_dcp() == True:
                result.append(vars_active) # find a subset
                current.append(vars_active)
            else:
                next_level.append(vars_active) # to be decomposed in the next level
    for set in next_level:
        result_temp = find_dcp_maxset(expr,set,current)
        if result_temp is None:
            return None
        else:
            for set in result_temp:
                result.append(set)
    return result

def find_dcp_set(expr, vars):
    """
    find subsets of variables, so that expr is a dcp expression within each subset
    :param expr:
    :param vars: variables that are not fixed
    :return: a list of lists of variables, or None
    """
    if vars == []:  # an empty list indicates that the expression is not multi-dcp
        return None
    vars_id = [var.id for var in vars]
    fix_vars = [var for var in expr.variables() if not var.id in vars_id]
    if fix(expr,fix_vars).is_dcp() == True:
        return [vars]
    else:
        result = []
        for var in vars: # erase each variable from vars
            vars_temp = erase(vars,var) # active variables
            result_temp = find_dcp_set(expr,vars_temp)
            if result_temp is None:
                return None
            for var_set in result_temp:
                result.append(var_set)
        return result

def erase(vars,var):
    """
    erase var from a set of variables vars
    :param vars: a non-empty set of variables
    :param var: the variable to be erased from the set
    :return: a set of variables
    """
    return [v for v in vars if v != var]