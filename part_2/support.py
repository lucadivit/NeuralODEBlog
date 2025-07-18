import numpy as np
import matplotlib.pyplot as plt
import inspect
from scipy.integrate import solve_ivp
import matplotlib.lines as mlines


def plot_direction_and_phase_1d(f, t_range=(0, 3), y_range=(-2, 2),
                                t_steps=20, y_steps=20,
                                initial_conditions=None, fn="fig"):
    t_min, t_max = t_range
    y_min, y_max = y_range

    # Mesh Direction Field
    t_vals = np.linspace(t_min, t_max, t_steps)
    y_vals = np.linspace(y_min, y_max, y_steps)
    T, Y = np.meshgrid(t_vals, y_vals)
    DT = np.ones_like(T)
    DY = f(T, Y)
    norm = np.sqrt(DT ** 2 + DY ** 2)
    DT /= norm
    DY /= norm

    plt.figure(figsize=(10, 6))
    plt.quiver(T, Y, DT, DY, angles='xy', color='black', alpha=0.6, label="Campo di Direzione")

    if initial_conditions is not None:
        for y0 in initial_conditions:
            sol = solve_ivp(f, t_range, [y0], t_eval=np.linspace(t_min, t_max, 300))
            plt.plot(sol.t, sol.y[0], label=fr"$y(0) = {y0:.2f}$")

    # Layout
    title = inspect.getsource(f).strip().split("lambda")[-1].split(":")[-1].split(",")[0]
    title = f"y' = {title}"
    plt.title(f"{title}", fontsize=16)
    plt.xlabel("t", fontsize=14)
    plt.ylabel("y", fontsize=14)
    plt.xlim(t_min, t_max)
    plt.ylim(y_min, y_max)
    plt.grid(True)
    if initial_conditions:
        handles, labels = plt.gca().get_legend_handles_labels()
        if len(labels) > 3:
            ellipsis_handle = mlines.Line2D([], [], color='gray', linestyle='--')
            hands = handles[:3] + [ellipsis_handle] + [handles[-1]]
            labs = labels[:3] + ['y(t) = ...'] + [labels[-1]]
            plt.legend(hands, labs, fontsize=14)
        else:
            plt.legend(fontsize=14)
    plt.tight_layout()
    plt.savefig(fn + ".svg")
    plt.show()