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

        #Define variables
        x1 = cvx.Variable()
        x2 = cvx.Variable()
        x3 = cvx.Variable()
        x4 = cvx.Variable()

        #Define problem
        obj = cvx.Minimize(cvx.abs(x1*x2 + x3*x4))
        constr = [x1*x2 + x3*x4 == 1]
        prob = cvx.Problem(obj, constr)
        
        #Assertion test
        self.assertEqual(bcd.is_dmcp(prob), True)

    def test_bcdSimple(self, prob):
        '''
        Checks if the solution method to solve the DMCP problem works to a simple problem.
        '''

    def test_bcdComplex(self, prob):
        '''
        Checks if the solution method to solve the DMCP problem works to a more complex problem.
        '''

    def test_bcdProximal(self, prob):
        '''
        Checks if the solution method to solve the DMCP problem works to a problem using proximal BCD.
        '''

    def test_block(self, prob):
        '''
        Checks if the block coordinate descent algorithm works.
        '''

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

        #Define slack variable inputs
        mu = 5e-3
        slack = cvx.Variable()

        #Define problem
        obj = cvx.Minimize(cvx.abs(x1*x2 + x3*x4))
        constr = [x1*x2 + x3*x4 == 1]
        prob = cvx.Problem(obj, constr)

        #Get slacked problem
        outputProb, slackList = bcd.add_slack(prob, mu)

        #Define test slacked problem
        objTest = cvx.Minimize(cvx.abs(x1*x2 + x3*x4) + mu*cvx.abs(slack))
        constrTest = [x1*x2 + x3*x4 - 1 == slack]
        probTest = cvx.Problem(objTest, constrTest)

        #Assertion Tests
        self.assertEqual(len(slackList), 1)
        self.assertEqual(outputProb, probTest)

    def test_proximal(self):
        '''
        Checks if proximal objective function works.
        '''

        #Define variables
        x = cvx.Variable(4,1)

        #Define initialization
        x.value = [1,1,0,0]

        #Define slack variable inputs
        lambd = 10
        slack = cvx.Variable()

        #Define problem
        obj = cvx.Minimize(cvx.abs(x[0]*x[1] + x[2]*x[3]))
        constr = [x[0]*x[1] + x[2]*x[3] == 1]
        prob = cvx.Problem(obj, constr)

        #Get proximal problem
        outputProb = bcd.proximal_op(prob, [slack], lambd)

        #Define test proximal slacked objective problem
        objTest = cvx.Minimize(cvx.abs(x[0]*x[1] + x[2]*x[3]) + (1/(2*lambd))*cvx.square(cvx.norm(x - x.value, 'fro')))
        constrTest = [x[0]*x[1] + x[2]*x[3] == 1]
        probTest = cvx.Problem(objTest, constrTest)

        #Assertion Test
        self.assertEqual(prob, probTest)