import math
import matplotlib.pyplot as plt

def plot_solutions(times_dict, solutions_dict, fn="comparison", plot_range=None):
    plt.figure(figsize=(5, 3))

    for label in sorted(solutions_dict.keys()):
        t = times_dict[label]
        y = solutions_dict[label]

        if plot_range is not None:
            t_min, t_max = plot_range
            t_filtered = []
            y_filtered = []
            for ti, yi in zip(t, y):
                if t_min <= ti <= t_max:
                    t_filtered.append(ti)
                    y_filtered.append(yi)
        else:
            t_filtered = t
            y_filtered = y

        plt.plot(t_filtered, y_filtered, '-', label=label)

    plt.xlabel("t", fontsize=10)
    plt.ylabel("y(t)", fontsize=10)
    plt.title("Numeric VS Analytic Solution", fontsize=14)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(fn + ".png")
    plt.show()

def euler_solver(f, y0, t0, tn, step_sizes):
    times = {}
    solutions = {}

    for h in step_sizes:
        n_steps = int((tn - t0) / h) + 1
        t = [t0]
        y = [y0]

        print(f"\n[h = {h}]")
        for n in range(1, n_steps):
            t_prev = t[-1]
            y_prev = y[-1]

            t_new = t_prev + h
            y_new = y_prev + h * f(t_prev, y_prev)

            t.append(round(t_new, 3))
            y.append(round(y_new, 3))

            print()
            print(f"n = {n}, t_{n} = {t_new:.3f}, y(t_{n}) = y(t_{n - 1}) + h * f(t_{n - 1}, y_{n - 1})")
            print(f"y({t_new:.1f}) = {y_prev:.3f} + {h} * f({t_prev:.1f}, {y_prev:.3f}) = {y_new:.3f}")

        key = f"Euler(h={h})"
        times[key] = t
        solutions[key] = y

    return times, solutions

if __name__ == "__main__":
    # ODE Definition
    def decay(t, y):
        k = 1.0
        return -k * y

    # Start/End Time, Integration Steps, initial condition
    t0 = 0.0
    tn = 2.0
    step_sizes = [0.1, 0.2, 0.5]
    y0 = 1.0

    times, solutions = euler_solver(f=decay, y0=y0, t0=t0, tn=tn, step_sizes=step_sizes)

    label = "Soluzione esatta"
    times[label] = times["Euler(h=0.1)"]
    solutions[label] = [math.exp(-t) for t in times[label]]

    plot_solutions(times, solutions, plot_range=(0., 1))