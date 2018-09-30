from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from dmcp.test.base_test import BaseTest
import numpy as np
import cvxpy as cvx
import dmcp.bcd as bcd

class bcdTestCases(BaseTest):
    def setUp(self):
        '''
        Used to setup all the parameters of the tests.
        '''
    
    def test_dmcp(self):
        '''
        Checks if a given problem (prob) is dmcp.
        '''
        x1 = cvx.Variable()
        x2 = cvx.Variable()
        x3 = cvx.Variable()
        x4 = cvx.Variable()

        obj = cvx.Minimize(cvx.abs(x1*x2 + x3*x4))
        constr = [x1*x2 + x3*x4 == 1]
        prob = cvx.Problem(obj, constr)
        
        assert bcd.is_dmcp(prob) == True

    def test_linearize(self, expr):
        '''
        Test the linearize function.
        '''
        z = cvx.Variable((1,5))
        expr = cvx.square(z)
        z.value = np.reshape(np.array([1,2,3,4,5]), (1,5))
        lin = linearize(expr)
        self.assertEqual(lin.shape, (1,5))
        self.assertItemsAlmostEqual(lin.value, [1,4,9,16,25])

    def test_slack(self, prob):
        '''
        Checks if the add slack function works.
        '''

    def test_proximal(self, prob):
        '''
        Checks if proximal objective function works.
        '''

    def test_block(self, prob):
        '''
        Checks if the block coordinate descent algorithm works.
        '''

    def test_bcd(self, prob):
        '''
        Checks if the solution method to solve the DMCP problem works.
        '''