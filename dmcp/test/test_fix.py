from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from dmcp.test.base_test import BaseTest
import numpy as np
import cvxpy as cvx
import dmcp

class fixTestCases(BaseTest):
    def setUp(self):
        '''
        Used to setup all the parameters of the tests.
        '''
    
    def test_fixObj(self, obj, fix_vars):
        '''
        Tests whether or not the fix variable function works for objects.
        '''

    def test_fixExpr(self, expr, fix_vars):
        '''
        Tests whether or not the fix variable function works for expressions.
        '''

    def test_fixProb(self, prob, fix_vars):
        '''
        Tests whether or not the fix variable function works for problems/
        '''