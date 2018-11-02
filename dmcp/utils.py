from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cvxpy as cvx
from cvxpy.atoms.affine.binary_operators import multiply
from cvxpy.atoms.affine.binary_operators import MulExpression

def is_atom_multiconvex(obj):
    if isinstance(obj, MulExpression):
        return True
    elif isinstance(obj, multiply):
        return True
    else:
        return False