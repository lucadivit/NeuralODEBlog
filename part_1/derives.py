from sympy import symbols, diff


print("---- Symbolic Solution ----")
t = symbols('t')

# Defining function s(t) = -t^2 + 10t
f = -t**2 + 10*t

# Computing first order derivative
f_prime = diff(f, t)
print("s'(t) = v(t) = ", f_prime)

# s and s' evaluation
for instant in [0, 5, 7]:
    print(f"s'({instant}) = v({instant}) = ", f_prime.subs(t, instant))

print("---- Difference Quotient Solution ----")

# Defining s(t) function
def s(t):
    return -t**2 + 10*t

# h -> oo
def difference_quotient(f, t, h=1e-5):
    delta = f(t + h) - f(t)
    value = delta / h
    return value

# s and s' evaluation
for instant in [0, 5, 7]:
    print(f"s'({instant}) = v({instant}) = ", difference_quotient(f=s, t=instant))

