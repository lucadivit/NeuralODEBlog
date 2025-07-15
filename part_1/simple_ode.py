from sympy import symbols, Function, dsolve, Eq, Derivative, pprint


print("---- Symbolic Solution ----")
print()

t = symbols('t')
y = Function('y')

dy_dt = Derivative(y(t), t)
ode = Eq(dy_dt, 3 * y(t))

ode_solution = dsolve(ode, y(t))
print("-- ODE --")
pprint(ode)
print("-- Solution --")
pprint(ode_solution)

print()
print("---- IVP ----")
print()

cauchy_solution = dsolve(ode, y(t), ics={y(0): 2})
print("-- ODE --")
pprint(ode)
print("-- Solution --")
pprint(cauchy_solution)