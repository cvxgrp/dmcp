from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from dmcp.test.base_test import BaseTest
import numpy as np
import cvxpy as cvx
from dmcp.fix import fix

class fixTestCases(BaseTest):  
    def setUp(self):
        '''
        Used to setup all the parameters of the tests.
        '''  
        #Define variables
        self.x1 = cvx.Variable(nonpos = True)
        self.x2 = cvx.Variable()
        self.x3 = cvx.Variable(nonneg = True)
        self.x4 = cvx.Variable()

        #Define fixed list
        self.fix_vars = [self.x1, self.x3]

    def test_fixExpr(self):
        '''
        Tests whether or not the fix variable function works for expressions.
        '''
        # Define expression
        expr = cvx.abs(self.x1 + self.x1 + self.x2 + self.x3 + self.x4)

        # Fix variables and get list of parameters
        new_expr = fix(expr, self.fix_vars)
        list_params = new_expr.parameters()

        # Assertion test
        # 2 variables since fix_vars has 2 elements
        self.assertEqual(len(list_params), 2)

    def test_fixProb(self):
        '''
        Tests whether or not the fix variable function works for problems/
        '''
        # Define problem
        obj = cvx.Minimize(cvx.abs(self.x1*self.x2 + self.x3*self.x4))
        constr = [self.x1 + self.x2 + self.x3 + self.x4 == 1]
        prob = cvx.Problem(obj, constr)

        # Fix variables and get list of parameters
        new_prob = fix(prob, self.fix_vars)
        list_params = new_prob.parameters()

        # Assertion test
        # 4 variables since x1 and x3 for both objective and constraints are fixed
        self.assertEqual(len(list_params), 2)

    def test_fixPSD(self):
        '''
        Tests whether or not the fix variable function works for problems/
        '''
        # Define problem
        x = cvx.Variable((4,4))
        obj = cvx.Minimize(cvx.norm(x))
        constr = [x >> 0]
        prob = cvx.Problem(obj, constr)

        # Fix variables and get list of parameters
        new_prob = fix(prob, [x])
        list_params = new_prob.parameters()

        # Assertion test
        # 2 variables since x is symmetric positive semi-definite
        self.assertEqual(len(list_params), 1)