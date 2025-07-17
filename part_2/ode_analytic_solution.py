from sympy import symbols, Function, dsolve, Eq, Derivative, pprint
from support import plot_direction_and_phase_1d
import numpy as np

print("---- ODE ----")
print()

t = symbols('t')
y = Function('y')

dy_dt = Derivative(y(t), t)
ode = Eq(dy_dt, 3 * y(t))

ode_solution = dsolve(ode, y(t))
print("-- Equation --")
pprint(ode)
print("-- Solution --")
pprint(ode_solution)

plot_direction_and_phase_1d(
    f = lambda t, y: 3*y,
    initial_conditions = list(np.arange(-0.5, 0.5 + 0.1, 0.1)),
    fn="substitution_plot",
    t_range=(0, 1)
)



print()
print("---- IVP ----")
print()

ode = Eq(dy_dt, 2 * y(t) * t + t)
ode_solution = dsolve(ode, y(t))
cauchy_solution = dsolve(ode, y(t), ics={y(0): 0})
print("-- Equation --")
pprint(ode)
print("-- Analytic Solution --")
pprint(ode_solution)
print("-- Cauchy Solution --")
pprint(cauchy_solution)

plot_direction_and_phase_1d(
    f = lambda t, y: 2*y*t + t,
    initial_conditions = [-2, -1, 0, 1, 2],
    fn="integrating_factor_plot",
    t_range=(-2, 2)
)