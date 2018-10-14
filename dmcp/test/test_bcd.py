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
        x = cvx.Variable((4,1))

        #Define problem
        obj = cvx.Minimize(cvx.abs(x[0]*x[1] + x[2]*x[3]))
        constr = [x[0]*x[1] + x[2]*x[3] == 1]
        prob = cvx.Problem(obj, constr)
        
        #Assertion test
        self.assertEqual(bcd.is_dmcp(prob), True)

    def test_bcdProximal(self):
        '''
        Checks if the solution method to solve the DMCP problem works to a simple problem.
        '''
        #Define variables
        x = cvx.Variable((4,1))

        #Define problem
        obj = cvx.Minimize(cvx.abs(x[0]*x[1] + x[2]*x[3]))
        constr = [x[0]*x[1] + x[2]*x[3] == 1]
        prob = cvx.Problem(obj, constr)

        #Solve problem
        prob.solve(method = 'bcd', update = 'proximal')

        #Assertion test
        self.assertAlmostEqual(prob.objective.value, 0, places = 3)
        self.assertAlmostEqual(prob.constraints[0].violation(), 0)

    def test_bcdMinimize(self):
        '''
        Checks if the solution method to solve the DMCP problem works to a simple problem.
        '''
        #Define variables
        x = cvx.Variable((4,1))

        #Define problem
        obj = cvx.Minimize(cvx.abs(x[0]*x[1] + x[2]*x[3]))
        constr = [x[0]*x[1] + x[2]*x[3] == 1]
        prob = cvx.Problem(obj, constr)

        #Solve problem
        prob.solve(method = 'bcd', update = 'minimize')

        #Assertion test
        self.assertAlmostEqual(prob.objective.value, 0, places = 3)
        self.assertAlmostEqual(prob.constraints[0].violation(), 0)

    def test_bcdProxLinear(self):
        '''
        Checks if the solution method to solve the DMCP problem works to a simple problem.
        '''
        #Define variables
        x = cvx.Variable((4,1))

        #Define problem
        obj = cvx.Minimize(cvx.abs(x[0]*x[1] + x[2]*x[3]))
        constr = [x[0]*x[1] + x[2]*x[3] == 1]
        prob = cvx.Problem(obj, constr)

        #Solve problem
        prob.solve(method = 'prox_linear')

        #Assertion test
        self.assertAlmostEqual(prob.objective.value, 0, places = 3)
        self.assertAlmostEqual(prob.constraints[0].violation(), 0)

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
        x = cvx.Variable((4,1))

        #Define slack variable inputs
        mu = 5e-3
        slack = cvx.Variable()

        #Define problem
        obj = cvx.Minimize(cvx.abs(x[0]*x[1] + x[2]*x[3]))
        constr = [x[0]*x[1] + x[2]*x[3] == 1]
        prob = cvx.Problem(obj, constr)

        #Get slacked problem
        outputProb, slackList = bcd.add_slack(prob, mu)

        #Define ground truth for testing
        objTest = cvx.Minimize(cvx.abs(x[0]*x[1] + x[2]*x[3]) + mu*cvx.abs(slack))
        constrTest = [x[0]*x[1] + x[2]*x[3] - 1 == slack]
        probTest = cvx.Problem(objTest, constrTest)        

        #Define Initialization for Testing
        x.value = [1,1,0,0]
        slack.value = 1/(5e-3)

        #Assertion Tests
        self.assertEqual(len(slackList), 1)
        self.assertAlmostEqual(outputProb.objective.value, objTest.value)
        self.assertAlmostEqual(outputProb.constraints[0].violation(), probTest.constraints[0].violation())

    def test_proximal(self):
        '''
        Checks if proximal objective function works.
        '''
        #Define variables
        x = cvx.Variable((4,1))

        #Define slack variable inputs
        lambd = 1/2
        slack = cvx.Variable()

        #Define initialization
        x.value = [1,1,0,0]
        slack.value = 1/(5e-3)

        #Define problem
        obj = cvx.Minimize(cvx.abs(x[0]*x[1] + x[2]*x[3]))
        constr = [x[0]*x[1] + x[2]*x[3] == 1]
        prob = cvx.Problem(obj, constr)

        #Get proximal problem
        outputProb = bcd.proximal_op(prob, [slack], lambd)

        #Define ground truth test for proximal objective
        objTest = cvx.Minimize(cvx.abs(x[0]*x[1] + x[2]*x[3]) + (1/(2*lambd))*cvx.square(cvx.norm(x - x.value, 'fro')))

        #Assertion Test
        self.assertAlmostEqual(outputProb.objective.value, objTest.value)