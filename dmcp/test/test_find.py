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
        self.assertEqual(outputSets, [[2,1], [3,1], [2,0], [3,0]])

    def test_findAllSets(self):
        '''
        Tests whether or not the function finds all minimal sets
        '''
        # Get minimal sets
        outputSets = find_set.find_minimal_sets(self.prob, is_all=True)

        # Assert the set set of minimal sets
        self.assertEqual(outputSets, [[2,1], [3,1], [2,0], [3,0]])
    """
    def test_findMIS(self):
        '''
        Tests the function that finds maximals independent sets of a graph until all vertices are included.
        '''

    def test_findAllIset(self):
        '''
        Tests the function that finds all independent subsets, including the empty set.
        '''

    def test_isIndependent(self):
        '''
        Tests the function that checks if a subset of vertices is independent on a graph.
        '''
    
    def test_findAllSubsets(self):
        '''
        Tests the function that finds all subsets of a set, except for the empty set.
        '''
    
    def test_searchConflictL(self):
        '''
        Tests the function that searches conflict variables in an expression using lists.
        '''

    def test_searchConflict(self):
        '''
        Tests the function that searches conflict variables in an expression.
        '''

    def test_isIntersect(self):
        '''
        Tests the function that checks if the intersection of two sets, set1 and set2, is nonempty.
        '''

    def test_union(self):
        '''
        Tests the function that takes the union of two sets set1 and set2.
        '''

    def test_findMaxsetProb(self):
        '''
        Tests the function that analyzes a problem to find maximal subsets of variables, 
        so that the problem is dcp restricting on each subset.
        '''

    def test_findDCPMaxset(self):
        '''
        Tests the function that finds maximal subsets of variables, so that expr is a
        dcp expression within each subset.
        '''

    def test_findDCPSet(self):
        '''
        Tests the function that finds subsets of variables, so that expr is
        a dcp expression within each subset.
        '''

    def test_isSubset(self):
        '''
        Tests the function that checks if a set is a subset of another set.
        '''

    def test_erase(self):
        '''
        Tests the function that erase variable from a set of variables.
        '''
    """