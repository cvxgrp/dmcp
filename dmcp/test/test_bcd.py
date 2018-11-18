from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from dmcp.test.base_test import BaseTest
import numpy as np
import cvxpy as cvx
import dmcp.bcd as bcd

class bcdTestCases(BaseTest):    
    def test_dmcp(self):
        '''
        Checks if a given problem (prob) is dmcp.
        '''
        #Define variables
        x1 = cvx.Variable()
        x2 = cvx.Variable()
        x3 = cvx.Variable()
        x4 = cvx.Variable()

        #Define problem
        obj = cvx.Minimize(cvx.abs(x1*x2 + x3*x4))
        constr = [x1 + x2 + x3 + x4 == 1]
        prob = cvx.Problem(obj, constr)
        
        #Assertion test
        self.assertEqual(bcd.is_dmcp(prob), True)

    def test_bcdProximal(self):
        '''
        Checks if the solution method to solve the DMCP problem works to a simple problem.
        '''
        #Define variables
        x1 = cvx.Variable()
        x2 = cvx.Variable()
        x3 = cvx.Variable()
        x4 = cvx.Variable()

        #Define problem
        obj = cvx.Minimize(cvx.abs(x1*x2 + x3*x4))
        constr = [x1 + x2 + x3 + x4 == 1]
        prob = cvx.Problem(obj, constr)

        #Solve problem
        result = prob.solve(method = 'bcd', update = 'proximal', solver=cvx.MOSEK, max_iter=100)

        #Assertion test
        epsilon = 20
        assert np.abs(prob.objective.value) < epsilon
        assert np.abs(result[1]) < epsilon

    def test_bcdMinimize(self):
        '''
        Checks if the solution method to solve the DMCP problem works to a simple problem.
        '''
        #Define variables
        x1 = cvx.Variable()
        x2 = cvx.Variable()
        x3 = cvx.Variable()
        x4 = cvx.Variable()

        #Define problem
        obj = cvx.Minimize(cvx.abs(x1*x2 + x3*x4))
        constr = [x1 + x2 + x3 + x4 == 1]
        prob = cvx.Problem(obj, constr)

        #Solve problem
        result = prob.solve(method = 'bcd', update = 'minimize', solver=cvx.MOSEK, max_iter=100)

        #Assertion test
        epsilon = 20
        assert np.abs(prob.objective.value) < epsilon
        assert np.abs(result[1]) < epsilon

    def test_bcdProxLinear(self):
        '''
        Checks if the solution method to solve the DMCP problem works to a simple problem.
        '''
        #Define variables
        x1 = cvx.Variable()
        x2 = cvx.Variable()
        x3 = cvx.Variable()
        x4 = cvx.Variable()

        #Define problem
        obj = cvx.Minimize(cvx.abs(x1*x2 + x3*x4))
        constr = [x1 + x2 + x3 + x4 == 1]
        prob = cvx.Problem(obj, constr)

        #Solve problem
        result = prob.solve(method = 'bcd', update = 'prox_linear', solver=cvx.MOSEK, max_iter=100)

        #Assertion test
        epsilon = 20
        assert np.abs(prob.objective.value) < epsilon
        assert np.abs(result[1]) < epsilon

    def test_linearize(self):
        '''
        Test the linearize function.
        '''
        #Define expression
        z = cvx.Variable((1,5))
        expr = cvx.square(z)

        #Initialize variable value
        z.value = np.reshape(np.array([1,2,3,4,5]), (1,5))

        #Linearize
        lin = bcd.linearize(expr)

        #Assertion tests
        self.assertEqual(lin.shape, (1,5))
        self.assertItemsAlmostEqual(lin.value, [1,4,9,16,25])

    def test_slack(self):
        '''
        Checks if the add slack function works.
        '''
        #Define variables
        x1 = cvx.Variable()
        x2 = cvx.Variable()
        x3 = cvx.Variable()
        x4 = cvx.Variable()
        x = cvx.Variable((4,4))

        #Define slack variable inputs
        mu = 5e-3
        slack = cvx.Variable()

        #Define problem
        obj = cvx.Minimize(cvx.abs(x1*x2 + x3*x4))
        constr = [x1 + x2 + x3 + x4 == 1]
        prob = cvx.Problem(obj, constr)

        #Get slacked problem
        outputProb, slackList = bcd.add_slack(prob, mu)

        #Define ground truth for testing
        objTest = cvx.Minimize(cvx.abs(x1*x2 + x3*x4) + mu*cvx.abs(slack))
        constrTest = [x1 + x2 + x3 + x4 - 1 == slack]
        probTest = cvx.Problem(objTest, constrTest)        

        #Define Initialization for Testing
        x1.value = 1
        x2.value = 1
        x3.value = 0
        x4.value = 0
        slack.value = 1/(5e-3)
        slackList[0].value = 1/(5e-3)

        #Assertion Tests
        self.assertEqual(len(slackList), 1)
        self.assertAlmostEqual(outputProb.objective.value, probTest.objective.value)
        self.assertAlmostEqual(outputProb.constraints[0].violation(), probTest.constraints[0].violation())

    def test_proximal(self):
        '''
        Checks if proximal objective function works.
        '''
        #Define variables
        x1 = cvx.Variable()
        x2 = cvx.Variable()
        x3 = cvx.Variable()
        x4 = cvx.Variable()

        #Define slack variable inputs
        lambd = 1/2
        slack = cvx.Variable()

        #Define initialization
        x1.value = 1
        x2.value = 1
        x3.value = 0
        x4.value = 0
        slack.value = 1/(5e-3)

        #Define problem
        obj = cvx.Minimize(cvx.abs(x1*x2 + x3*x4))
        constr = [x1 + x2 + x3 + x4 == 1]
        prob = cvx.Problem(obj, constr)

        #Get proximal problem
        outputProb = bcd.proximal_op(prob, [slack], lambd)

        #Define ground truth test for proximal objective
        objTest = cvx.Minimize(cvx.abs(x1*x2 + x3*x4) 
                                + cvx.square(cvx.norm(x1 - x1.value, 'fro'))/2/lambd
                                + cvx.square(cvx.norm(x2 - x2.value, 'fro'))/2/lambd
                                + cvx.square(cvx.norm(x3 - x3.value, 'fro'))/2/lambd
                                + cvx.square(cvx.norm(x4 - x4.value, 'fro'))/2/lambd)

        #Assertion Test
        self.assertAlmostEqual(outputProb.objective.value, objTest.value)
        self.assertEqual(outputProb.objective.expr.name(), objTest.expr.name())