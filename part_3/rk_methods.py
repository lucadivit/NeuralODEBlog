import numpy as np
import matplotlib.pyplot as plt


def plot_phase_overlay(solutions_dict, fn="phase_comparison"):

    plt.figure(figsize=(5, 3))
    for i, label in enumerate(sorted(solutions_dict.keys())):
        sol = solutions_dict[label]
        x = [yi[0] for yi in sol]
        y = [yi[1] for yi in sol]
        plt.plot(x, y, label=label)

    plt.xlabel("x", fontsize=10)
    plt.ylabel("y", fontsize=10)
    plt.title("Phase Comparison", fontsize=14)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(fn + ".png")
    plt.show()

def plot_time_series_overlay(times_dict, solutions_dict, fn="time_series_comparison"):
    fig, axs = plt.subplots(2, 1, figsize=(5, 3), sharex=True)

    for label in sorted(solutions_dict.keys()):
        t = times_dict[label]
        sol = solutions_dict[label]
        x = [yi[0] for yi in sol]
        y = [yi[1] for yi in sol]

        axs[0].plot(t, x, label=label)
        axs[1].plot(t, y, label=label)

    axs[0].set_ylabel("x(t)", fontsize=10)
    axs[1].set_ylabel("y(t)", fontsize=10)
    axs[1].set_xlabel("t", fontsize=10)

    axs[0].set_title("x(t), y(t) Time Evolution", fontsize=14)
    axs[0].grid(True)
    axs[1].grid(True)

    axs[0].legend()
    plt.tight_layout()
    plt.savefig(fn + ".png")
    plt.show()

def euler_solver(f, y0, t0, tn, step_sizes):
    times = {}
    solutions = {}

    for h in step_sizes:
        print(f"Starting Euler Solver with h={h}")
        n_steps = int((tn - t0) / h) + 1
        t = [t0]
        y = [np.array(y0, dtype=float)]

        print(f"\n[h = {h}]")
        for n in range(1, n_steps):
            t_prev = t[-1]
            y_prev = y[-1]

            t_new = t_prev + h
            y_new = y_prev + h * f(t_prev, y_prev)

            t.append(round(t_new, 3))
            y.append(np.round(y_new, 3))

            print()
            print(f"n = {n}, t_{n} = {t_new:.3f}")
            print(f"y({t_new:.1f}) = {y_prev} + {h} * f({t_prev:.1f}, {y_prev}) = {y_new}")

        key = f"Euler(h={h})"
        times[key] = t
        solutions[key] = y

    return times, solutions


def heun_solver(f, y0, t0, tn, step_sizes):
    times = {}
    solutions = {}

    for h in step_sizes:
        print(f"Starting Heun Solver with h={h}")
        n_steps = int((tn - t0) / h) + 1
        t = [t0]
        y = [np.array(y0, dtype=float)]

        print(f"\n[h = {h}]")
        for n in range(1, n_steps):
            t_prev = t[-1]
            y_prev = y[-1]

            t_new = t_prev + h
            k1 = f(t_prev, y_prev)
            k2 = f(t_new, y_prev + h * k1)
            y_new = y_prev + (h / 2) * (k1 + k2)

            t.append(round(t_new, 3))
            y.append(np.round(y_new, 3))

            print()
            print(f"n = {n}, t_{n} = {t_prev:.3f}, t_{n + 1} = {t_prev:.3f}")
            print(f"k1 = f({y_prev}, {t_prev:.1f}) = {k1}")
            print(f"k2 = f({y_prev} + {h}*k1, {t_new:.1f}) = {k2}")
            print(f"y({t_new:.1f}) = {y_prev} + {h}/2 * (k1 + k2) = {y_new}")

        key = f"Heun(h={h})"
        times[key] = t
        solutions[key] = y

    return times, solutions

def rk4_solver(f, y0, t0, tn, step_sizes):
    times = {}
    solutions = {}

    for h in step_sizes:
        print(f"Starting RK4 Solver with h={h}")
        n_steps = int((tn - t0) / h) + 1
        t = [t0]
        y = [np.array(y0, dtype=float)]

        print(f"\n[h = {h}]")
        for n in range(1, n_steps):
            t_prev = t[-1]
            y_prev = y[-1]
            t_new = t_prev + h

            k1 = f(t_prev, y_prev)
            k2 = f(t_prev + h / 2, y_prev + h / 2 * k1)
            k3 = f(t_prev + h / 2, y_prev + h / 2 * k2)
            k4 = f(t_new, y_prev + h * k3)

            y_new = y_prev + (h / 6) * (k1 + 2*k2 + 2*k3 + k4)

            t.append(round(t_new, 3))
            y.append(np.round(y_new, 3))

            print()
            print(f"n = {n}, t_{n} = {t_prev:.3f}, t_{n + 1} = {t_new:.3f}")
            print(f"k1 = f({t_prev:.3f}, {y_prev}) = {k1}")
            print(f"k2 = f({t_prev + h/2:.3f}, {y_prev + h/2 * k1}) = {k2}")
            print(f"k3 = f({t_prev + h/2:.3f}, {y_prev + h/2 * k2}) = {k3}")
            print(f"k4 = f({t_new:.3f}, {y_prev + h * k3}) = {k4}")
            print(f"y({t_new:.1f}) = {y_prev} + h/6 * (k1 + 2k2 + 2k3 + k4) = {y_new}")

        key = f"RK4(h={h})"
        times[key] = t
        solutions[key] = y

    return times, solutions

if __name__ == "__main__":

    # ODE Definition
    def lotka_volterra(t, y):
        alpha = 1.1
        beta = 0.4
        delta = 0.1
        gamma = 0.4
        x, y_ = y
        dxdt = alpha * x - beta * x * y_
        dydt = delta * x * y_ - gamma * y_
        return np.array([dxdt, dydt])

    # Start/End Time, Integration Steps, Initial Condition
    t0 = 0.0
    tn = 2.
    step_sizes = [0.1]
    y0 = (10, 5)

    times, solutions = {}, {}

    t, s = euler_solver(f=lotka_volterra, y0=y0, t0=t0, tn=tn, step_sizes=step_sizes)
    solutions = {**solutions, **s}
    times = {**times, **t}

    t, s = heun_solver(f=lotka_volterra, y0=y0, t0=t0, tn=tn, step_sizes=step_sizes)
    solutions = {**solutions, **s}
    times = {**times, **t}

    t, s = rk4_solver(f=lotka_volterra, y0=y0, t0=t0, tn=tn, step_sizes=step_sizes)
    solutions = {**solutions, **s}
    times = {**times, **t}

    plot_phase_overlay(solutions_dict=solutions, fn=f"phase_comparison_t{int(tn)}")
    plot_time_series_overlay(solutions_dict=solutions, times_dict=times, fn=f"time_series_comparison_t{int(tn)}")