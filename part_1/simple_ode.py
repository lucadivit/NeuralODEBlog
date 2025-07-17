from sympy import symbols, Function, dsolve, Eq, Derivative, pprint

t = symbols('t')
y = Function('y')

dy_dt = Derivative(y(t), t)
ode = Eq(dy_dt, 3 * y(t))

ode_solution = dsolve(ode, y(t))
print("-- ODE --")
pprint(ode)