import numpy as np, matplotlib.pyplot as plt, torch
from .Data import DatasetProvider
from collections import OrderedDict

class HarmonicOscillator(DatasetProvider):
    """
    Oscillatore armonico smorzato in forma di stato:
        x'' + gamma * x' + omega^2 * x = 0
         __
       | x1' = x2
       | x2' = - omega^2 * x1 - gamma * x2
       | x1(0) = a
       | x2(0) = b
       | __

    Restituisce un DataObject con
    """

    def __init__(self, n_train: int = 80, noise_std: float = 0.01, t_train_range: tuple[float, float] = (0.0, 10.0),
                 t_extra_range: tuple[float, float] = (10.0, 15.0), n_extra: int = 200, n_resamp: int = 200, omega: float = 1.0,
                 gamma: float = 0.1, initial_values: tuple[float, float] = (2.0, 0.0)) -> None:
        super().__init__(initial_values=initial_values, n_train=n_train, noise_std=noise_std,
                         t_train_range=t_train_range, t_extra_range=t_extra_range, n_extra=n_extra,
                         n_resamp=n_resamp)

        self.omega = float(omega)
        self.gamma = float(gamma)

    @staticmethod
    def get_dim() -> int:
        return 2

    @staticmethod
    def get_name() -> str:
        return "HarmonicOscillator"

    @staticmethod
    def get_method() -> str:
        return "rk4"

    def dynamics(self, t: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        x1, x2 = state[0], state[1]
        x1dt = x2
        x2dt = -(self.omega ** 2) * x1 - self.gamma * x2
        return torch.stack([x1dt, x2dt]).to(self.dtype)

    def plot(self, data_obj: dict, show_noised: bool = True, show_true: bool = True,
             show_resamp: bool = True, show_extra: bool = True, figsize: tuple[int, int] = (10, 8),
             save: bool = True):

        if not show_true and not show_resamp and not show_extra and not show_noised:
            raise Exception("At Almost One Of These Flags, Must Be True 'show_true', 'show_resamp', 'show_extra', 'show_noised'")

        t_train = data_obj["t_train"]
        x_train_true = data_obj["x_train_true"]
        x_train_noised = data_obj["x_train_noised"]

        t_test_resamp = data_obj["t_test_resamp"]
        x_test_resamp_true = data_obj["x_test_resamp_true"]
        x_test_resamp_noised = data_obj["x_test_resamp_noised"]

        t_test_extra = data_obj["t_test_extra"]
        x_test_extra_true = data_obj["x_test_extra_true"]

        fig, axs = plt.subplots(2, 2, figsize=figsize)
        ax1, ax2, ax3, ax4 = axs[0, 0], axs[1, 0], axs[0, 1], axs[1, 1]

        # --- x1(t) ---
        if show_true:
            ax1.plot(t_train, x_train_true[:, 0], 'o', label="train true x1", color='blue')
            if show_noised:
                ax1.plot(t_train, x_train_noised[:, 0], 'o', label="train noised x1", color='cyan', alpha=0.6)
        if show_resamp:
            ax1.plot(t_test_resamp, x_test_resamp_true[:, 0], 'x', label="resamp true x1", color='red')
            if show_noised:
                ax1.plot(t_test_resamp, x_test_resamp_noised[:, 0], 'x', label="resamp noised x1", color='purple')
        if show_extra:
            ax1.plot(t_test_extra, x_test_extra_true[:, 0], '-', label="extra true x1", color='orange')
        ax1.set_xlabel("time t")
        ax1.set_ylabel("x1")

        # --- x2(t) ---
        if show_true:
            ax2.plot(t_train, x_train_true[:, 1], 'o', label="train true x2", color='blue')
            if show_noised:
                ax2.plot(t_train, x_train_noised[:, 1], 'o', label="train noised x2", color='cyan', alpha=0.6)
        if show_resamp:
            ax2.plot(t_test_resamp, x_test_resamp_true[:, 1], 'x', label="resamp true x2", color='red')
            if show_noised:
                ax2.plot(t_test_resamp, x_test_resamp_noised[:, 1], 'x', label="resamp noised x2", color='purple')
        if show_extra:
            ax2.plot(t_test_extra, x_test_extra_true[:, 1], '-', label="extra true x2", color='orange')
        ax2.set_xlabel("time t")
        ax2.set_ylabel("x2")

        # --- Phase space: x2 vs x1 ---
        if show_true:
            ax3.plot(x_train_true[:, 0], x_train_true[:, 1], 'o', label="train true", color='blue')
            if show_noised:
                ax3.plot(x_train_noised[:, 0], x_train_noised[:, 1], 'o', label="train noised", color='cyan', alpha=0.6)
        if show_resamp:
            ax3.plot(x_test_resamp_true[:, 0], x_test_resamp_true[:, 1], 'x', label="resamp true", color='red')
            if show_noised:
                ax3.plot(x_test_resamp_noised[:, 0], x_test_resamp_noised[:, 1], 'x', label="resamp noised", color='purple')
        if show_extra:
            ax3.plot(x_test_extra_true[:, 0], x_test_extra_true[:, 1], '-', label="extra true", color='orange')
        ax3.set_xlabel("x1")
        ax3.set_ylabel("x2")
        ax3.set_title("Phase space")

        fig.delaxes(ax4)

        handles, labels = [], []
        h, l = ax1.get_legend_handles_labels()
        handles += h
        labels += l
        by_label = OrderedDict(zip(labels, handles))

        fig.legend(by_label.values(), by_label.keys(), loc="lower right", ncol=1, frameon=False)

        plt.tight_layout()
        if save:
            plt.savefig(f"{self.get_name()}.svg", format='svg')
        else:
            plt.show()

    @staticmethod
    def plot_predictions(predictions: np.ndarray, targets: np.ndarray, t: np.ndarray, save: bool = True, fn: str = None):
        fig, axs = plt.subplots(2, 2, figsize=(10, 6))
        ax1, ax2, ax3, ax4 = axs[0, 0], axs[1, 0], axs[0, 1], axs[1, 1]

        ax1.plot(t, targets[:, 0], label="true")
        ax1.plot(t, predictions[:, 0], "--", label="predictions")
        ax1.set_xlabel("time t")
        ax1.set_ylabel("x1")

        ax2.plot(t, targets[:, 1], label="true")
        ax2.plot(t, predictions[:, 1], "--", label="predictions")
        ax2.set_xlabel("time t")
        ax2.set_ylabel("x2")

        ax3.plot(targets[:, 0], targets[:, 1], label="true")
        ax3.plot(predictions[:, 0], predictions[:, 1], "--", label="predictions")
        ax3.set_xlabel("x1")
        ax3.set_ylabel("x2")
        ax3.set_title("Phase space")

        fig.delaxes(ax4)

        handles, labels = [], []
        h, l = ax1.get_legend_handles_labels()
        handles += h
        labels += l
        by_label = OrderedDict(zip(labels, handles))

        fig.legend(by_label.values(), by_label.keys(), loc="lower right", ncol=1, frameon=False)

        plt.tight_layout()

        if save:
            if fn is None:
                fn = f"{HarmonicOscillator.get_name()}_predictions.svg"
            plt.savefig(fn, format='svg')
        else:
            plt.show()