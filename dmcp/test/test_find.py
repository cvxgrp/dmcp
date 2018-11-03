from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from dmcp.test.base_test import BaseTest
import numpy as np
import cvxpy as cvx
import dmcp.find_set as find_set

class findMinimalTestCases(BaseTest):
    def setUp(self):
        '''
        Used to setup all the parameters of the tests.
        '''
        # Define variables
        x1 = cvx.Variable()
        x2 = cvx.Variable()
        x3 = cvx.Variable()
        x4 = cvx.Variable()

        print(x1.id, x2.id, x3.id, x4.id)

        # Define problem
        obj = cvx.Minimize(cvx.abs(x1*x2 + x3*x4))
        constr = [x1 + x2 + x3 + x4 == 1]
        self.prob = cvx.Problem(obj, constr)
    
    def test_findMinimalSets(self):
        '''
        Tests the function that finds minimal sets.
        '''
        # Get minimal sets
        outputSets = find_set.find_minimal_sets(self.prob)
        # Assert the set set of minimal sets
        outputMinimal = {frozenset(i) for i in outputSets}
        print(outputSets)
        print(outputMinimal)
        self.assertEqual(outputMinimal, {frozenset([1, 3]), frozenset([1, 2]), frozenset([0, 3])})

    def test_findAllSets(self):
        '''
        Tests whether or not the function finds all minimal sets
        '''
        # Get minimal sets
        outputSets = find_set.find_minimal_sets(self.prob, is_all=True)

        # Assert the set set of minimal sets
        outputMinimal = {frozenset(i) for i in outputSets}
        print(outputSets)
        print(outputMinimal)
        self.assertEqual(outputMinimal, {frozenset([2,1]), frozenset([3,1]), frozenset([2,0]), frozenset([3,0])})