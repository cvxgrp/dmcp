from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import unittest
import numpy as np
import cvxpy as cvx
import dmcp
class bcdTest(unittest.TestCase):
    def setUp(self):
        '''
        Used to setup all the parameters of the tests.
        '''
    
    def test_dmcp(self, prob):
        '''
        Checks if a given problem (prob) is dmcp.
        '''

    def test_linearize(self, expr):
        '''
        Checks if the linearize function works on a given DMCP expression.
        '''

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