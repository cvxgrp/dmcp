from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cvxpy as cvx
from cvxpy.atoms.affine.binary_operators import multiply
from cvxpy.atoms.affine.binary_operators import MulExpression

def is_atom_multiconvex(obj):
    '''
        Checks of operator node in expression tree is a multi-convex atom.
        Input:
            obj:        expression
        output:
            boolean:    if atom is multi-convex
    '''
    if isinstance(obj, MulExpression):
        return True
    elif isinstance(obj, multiply):
        return True
    else:
        return False