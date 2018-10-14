from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from dmcp.test.base_test import BaseTest
import numpy as np
import cvxpy as cvx
from dmcp.fix import fix

class fixTestCases(BaseTest):    
    def test_fixExpr(self, obj, fix_vars):
        '''
        Tests whether or not the fix variable function works for expressions.
        '''
        #Define variables
        x1 = cvx.Variable()
        x2 = cv2.Variable()
        x3 = cv2.Variable()
        x4 = cv2.Variable()

        #Define expression
        expr = cvx.abs(x1*x2 + x3*x4)

        #Define fixed list
        fix_vars = [x1, x3]

        #Fix variables and get list of parameters
        new_expr = fix(expr, fix_vars))
        list_params = new_expr.parameters()

        #Assertion test
        self.assertEqual(len(list_params), len(fix_vars))

    def test_fixProb(self, prob, fix_vars):
        '''
        Tests whether or not the fix variable function works for problems/
        '''
        #Define variables
        x1 = cvx.Variable()
        x2 = cv2.Variable()
        x3 = cv2.Variable()
        x4 = cv2.Variable()
        
        #Define problem
        obj = cvx.Minimize(cvx.abs(x1*x2 + x3*x4))
        constr = [x1*x2 + x3*x4 == 1]
        prob = cvx.Problem(obj, constr)

        #Define fixed list
        fix_vars = [x1, x3]

        #Fix variables and get list of parameters
        new_prob = fix(prob, fix_vars))
        list_params = new_prob.parameters()

        #Assertion test
        self.assertEqual(len(list_params), len(fix_vars))