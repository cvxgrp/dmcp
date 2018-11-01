from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from dmcp.test.base_test import BaseTest
import numpy as np
import cvxpy as cvx
import dmcp.initial as initial

class initialTestCases(BaseTest):
    def setUp(self):
        '''
        Used to setup all the parameters of the tests.
        '''
        #Define variables
        x1 = cvx.Variable()
        x2 = cvx.Variable()
        x3 = cvx.Variable()
        x4 = cvx.Variable()
        
        #Define problem
        obj = cvx.Minimize(cvx.abs(x1*x2 + x3*x4))
        constr = [x1 + x2 + x3 + x4 == 1]
        self.prob = cvx.Problem(obj, constr)
    
    def test_randInitial(self):
        '''
        Test whether or not the rand_initial function initializes the variables
        '''
        #Create random initialization
        initial.rand_initial(self.prob)

        #Check if problem variables are initialized
        values = [var.value for var in self.prob.variables()]
        self.assertEqual(np.any(values is None), False)
    
    def test_randInitialProject(self):
        '''
        Test whether ot not the rand_initial_project function intializaes the variables
        with projects in place.
        '''
        initial.rand_initial_proj(self.prob)

        #Check if problem variables are initialized
        values = [var.value for var in self.prob.variables()]
        self.assertEqual(np.any(values is None), False)