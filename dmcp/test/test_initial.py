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
    
    def test_randInitial(self, prob):
        '''
        Test whether or not the rand_initial function initializes the variables
        '''
    
    def test_randInitialProject(self, prob):
        '''
        Test whether ot not the rand_initial_project function intializaes the variables
        with projects in place.
        '''