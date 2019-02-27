__author__ = 'Xinyue'

from cvxpy import *
import dmcp
from dmcp.fix import fix
from dmcp.find_set import find_minimal_sets
from dmcp.bcd import is_dmcp

alpha = Variable(nonneg=True)
alpha.value = 1
#x = NonNegative(1)
#y = NonNegative(1)

x = Variable()
y = Variable()
z = Variable(nonneg=True)
w = Variable()

x.value = 2.0
y.value = 1
z.value  = 0.6
w.value = 5.0

expr = y*x
print fix(expr,[y])

prob = Problem(Minimize(alpha), [square(x) +1 <= sqrt(x+0.5)*alpha])
#prob = Problem(Minimize(inv_pos(sqrt(y+0.5))*(square(x) +1)), [x==y])
#prob = Problem(Minimize(alpha), [square(x)+1 <= log(x+2)*alpha])
#prob = Problem(Minimize(inv_pos(log(y+2))*(square(x)+1)), [x==y])
#prob = Problem(Minimize(inv_pos(x+1)*exp(y)), [x==y])
#prob = Problem(Minimize(alpha), [exp(x) <= (x+1)*alpha])
#prob = Problem(Minimize(z*(abs(x)+abs(y))))
print fix(prob,[y])

print prob.is_dcp()
print is_dmcp(prob)
print find_minimal_sets(prob)
prob.solve(method = 'bcd', ep = 1e-4, rho = 1.1)
#print "======= solution ======="
#for var in prob.variables():
#    print var.name(), "=", var.value
#print "objective = ", prob.objective.args[0].value

# bisection
print "==== bisection method ===="
upper = Parameter(nonneg=True)
lower = Parameter(nonneg=True)
prob = Problem(Minimize(0), [square(x) +1<=(upper+lower)*sqrt(x+0.5)/float(2)])
upper.value = 1000
lower.value = 0
flag = 1
while lower.value +1e-3 <= upper.value:
    prob.solve()
    if x.value == None:
        lower.value = (upper+lower).value/float(2)
    else:
        upper.value = (upper+lower).value/float(2)
print "upper = ", upper.value
print "lower = ", lower.value
