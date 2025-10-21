import numpy as np, matplotlib.pyplot as plt, torch
from .Data import DatasetProvider
from collections import OrderedDict
from typing import Dict

class HarmonicOscillator(DatasetProvider):

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
        return "dopri5"

    def dynamics(self, t: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        x1, x2 = state[0], state[1]
        x1dt = x2
        x2dt = -(self.omega ** 2) * x1 - self.gamma * x2
        return torch.stack([x1dt, x2dt]).to(self.dtype)

    def plot(self, data_obj: dict, show_noised: bool = True, show_true: bool = True,
             show_resamp: bool = True, show_extra: bool = True, interpolate: bool = False,
             figsize: tuple[int, int] = (10, 8), save: bool = True):

        if not show_true and not show_resamp and not show_extra and not show_noised:
            raise Exception("At least one of 'show_true', 'show_resamp', 'show_extra', or 'show_noised' must be True")

        # --- unpack dei dati ---
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

        # --- funzione helper per gestire interpolazione / marker ---
        def plot_data(ax, t, x, label, color, marker, alpha=1.0):
            """Disegna punti o linee a seconda del flag interpolate."""
            if interpolate:
                ax.plot(t, x, '-', label=label, color=color, alpha=alpha)
            else:
                ax.plot(t, x, marker, label=label, color=color, alpha=alpha)

        # --- x1(t) ---
        if show_true:
            plot_data(ax1, t_train, x_train_true[:, 0], "train true x1", 'blue', 'o')
            if show_noised:
                plot_data(ax1, t_train, x_train_noised[:, 0], "train noised x1", 'cyan', 'o', alpha=0.6)
        if show_resamp:
            plot_data(ax1, t_test_resamp, x_test_resamp_true[:, 0], "resamp true x1", 'red', 'x')
            if show_noised:
                plot_data(ax1, t_test_resamp, x_test_resamp_noised[:, 0], "resamp noised x1", 'purple', 'x')
        if show_extra:
            plot_data(ax1, t_test_extra, x_test_extra_true[:, 0], "extra true x1", 'orange', '-', alpha=1.0)
        ax1.set_xlabel("time t")
        ax1.set_ylabel("x1")

        # --- x2(t) ---
        if show_true:
            plot_data(ax2, t_train, x_train_true[:, 1], "train true x2", 'blue', 'o')
            if show_noised:
                plot_data(ax2, t_train, x_train_noised[:, 1], "train noised x2", 'cyan', 'o', alpha=0.6)
        if show_resamp:
            plot_data(ax2, t_test_resamp, x_test_resamp_true[:, 1], "resamp true x2", 'red', 'x')
            if show_noised:
                plot_data(ax2, t_test_resamp, x_test_resamp_noised[:, 1], "resamp noised x2", 'purple', 'x')
        if show_extra:
            plot_data(ax2, t_test_extra, x_test_extra_true[:, 1], "extra true x2", 'orange', '-', alpha=1.0)
        ax2.set_xlabel("time t")
        ax2.set_ylabel("x2")

        # --- Phase space: x2 vs x1 ---
        def plot_phase(ax, x, label, color, marker, alpha=1.0):
            """Disegna la traiettoria nello spazio delle fasi."""
            if interpolate:
                ax.plot(x[:, 0], x[:, 1], '-', label=label, color=color, alpha=alpha)
            else:
                ax.plot(x[:, 0], x[:, 1], marker, label=label, color=color, alpha=alpha)

        if show_true:
            plot_phase(ax3, x_train_true, "train true", 'blue', 'o')
            if show_noised:
                plot_phase(ax3, x_train_noised, "train noised", 'cyan', 'o', alpha=0.6)
        if show_resamp:
            plot_phase(ax3, x_test_resamp_true, "resamp true", 'red', 'x')
            if show_noised:
                plot_phase(ax3, x_test_resamp_noised, "resamp noised", 'purple', 'x')
        if show_extra:
            plot_phase(ax3, x_test_extra_true, "extra true", 'orange', '-', alpha=1.0)
        ax3.set_xlabel("x1")
        ax3.set_ylabel("x2")
        ax3.set_title("Phase space")

        fig.delaxes(ax4)

        # --- legenda unica ---
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

    @staticmethod
    def plot_dataset(loaders: Dict):

        style_noised = {"color": "darkorange", "alpha": 0.7}
        style_true = {"color": "navy", "alpha": 0.9}

        def collect_split(split_key: str):
            if split_key not in loaders:
                return None
            dl = loaders[split_key]["dataset"]
            xs_i, xs_tgt, ts_i, ts_tgt = [], [], [], []
            for batch in dl:
                x_i, x_tgt, _, t_i, t_tgt = batch
                xs_i.append(x_i.detach().cpu())
                xs_tgt.append(x_tgt.detach().cpu())
                ts_i.append(t_i.detach().cpu())
                ts_tgt.append(t_tgt.detach().cpu())
            if not xs_i:
                return None
            x_i = torch.cat(xs_i, dim=0).numpy()
            x_tgt = torch.cat(xs_tgt, dim=0).numpy()
            t_i = torch.cat(ts_i, dim=0).numpy()
            t_tgt = torch.cat(ts_tgt, dim=0).numpy()
            return x_i, x_tgt, t_i, t_tgt

        splits_data = {}
        for key in ("train", "resamp"):
            data = collect_split(key)
            if data is not None:
                splits_data[key] = data

        any_key = next(iter(splits_data.keys()))
        D = splits_data[any_key][0].shape[-1]

        for split_key, (x_i, x_tgt, t_i, t_tgt) in splits_data.items():
            fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(10, 10), sharex=False)
            fig.suptitle(f"{split_key.upper()} dataset", fontsize=13)

            ax0 = axes[0]
            order_i = np.argsort(t_i)
            order_tgt = np.argsort(t_tgt)
            ax0.plot(t_i[order_i], x_i[order_i, 0], linestyle="-", label="noised", **style_noised)
            ax0.plot(t_tgt[order_tgt], x_tgt[order_tgt, 0], linestyle="-", label="true", **style_true)
            ax0.set_ylabel("x1")
            ax0.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
            ax0.legend(loc="best", fontsize=9, frameon=True)

            if D >= 2:
                ax1 = axes[1]
                order_i = np.argsort(t_i)
                order_tgt = np.argsort(t_tgt)
                ax1.plot(t_i[order_i], x_i[order_i, 1], linestyle="-", **style_noised)
                ax1.plot(t_tgt[order_tgt], x_tgt[order_tgt, 1], linestyle="-", **style_true)
                ax1.set_ylabel("x2")
                ax1.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
            else:
                axes[1].axis("off")

            ax2 = axes[2]
            if D >= 2:
                ax2.plot(x_i[:, 0], x_i[:, 1], linestyle="-", label="noised", **style_noised)
                ax2.plot(x_tgt[:, 0], x_tgt[:, 1], linestyle="-", label="true", **style_true)
                ax2.set_xlabel("x1")
                ax2.set_ylabel("x2")
                ax2.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
                ax2.legend(loc="best", fontsize=9, frameon=True)
            else:
                ax2.axis("off")

            fig.tight_layout(rect=(0, 0, 1, 0.95))
            fig.savefig(f"dataset_{HarmonicOscillator.get_name()}_{split_key}.svg", format="svg")
            plt.close(fig)
