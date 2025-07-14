from sympy import symbols, integrate

print("---- Symbolic Solution ----")
t = symbols('t')
v = 2*t

undefined_integral = integrate(v, t)
print("v'(t) = s(t) = ", undefined_integral)

instant = 2
defined_integral = integrate(v, (t, 0, instant))
print(f"v'({instant}) = s({instant}) = ", defined_integral)

print("---- Infinite Riemann Sum ----")

# Defining v(t) function
def v(t):
    return 2*t

# n -> oo
def riemann_sum(f, a, b, n=10000):
    dx = (a - b) / n
    sum_areas = 0
    for i in range(n):
        # area = f(x_i) * dx
        sum_areas += f(i * dx) * dx
    return sum_areas

# v' = s evaluation
print(f"v'({instant}) = s({instant}) = ", riemann_sum(f=v, a=2, b=0))