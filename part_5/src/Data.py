from typing import TypedDict, Optional, Callable, Tuple, List
import numpy as np, torch, os, cv2
from torch.utils.data import TensorDataset, DataLoader
from torchdiffeq import odeint

class DataObject(TypedDict):
    # TRAIN

    # tempi irregolari usati per il training (es. [0..6])
    t_train: np.ndarray
    # osservazioni rumorose --> usati in fase di train
    x_train_noised: np.ndarray
    # osservazioni senza rumore --> usati per valutazione
    x_train_true: np.ndarray


    # TEST

    # tempi di test fuori dal range di training (es. [6..20])
    t_test_extra: np.ndarray
    # osservazioni non rumorose dei tempi di test --> usati per valutare la capacità di generalizzare.
    x_test_extra_true: np.ndarray

    # tempi irregolari nello stesso range del training ([0..6])
    t_test_resamp: np.ndarray
    # osservazioni tempi irregolari --> usati per valutare robustezza in istanti diversi da quelli visti
    x_test_resamp_true: np.ndarray
    x_test_resamp_noised: np.ndarray

class Loader(TypedDict):
    dataset: DataLoader

class Loaders(TypedDict):
    train: Loader
    resamp: Loader
    extra: Loader

class DatasetProvider:

    def __init__(self, initial_values: tuple[float, float], dtype=torch.float64, r_tol: float = 1e-6, a_tol: float = 1e-8, n_train: int = 80, noise_std: float = 0.01,
                 t_train_range: tuple[float, float] = (0.0, 6.0), t_extra_range: tuple[float, float] = (6.0, 10.0), n_extra: int = 200, n_resamp: int = 200):
        self.dtype = dtype
        self.r_tol = r_tol
        self.a_tol = a_tol
        self.n_train = n_train
        self.noise_std = noise_std
        self.t_train_range = t_train_range
        self.t_extra_range = t_extra_range
        self.n_extra = n_extra
        self.n_resamp = n_resamp
        self.x0_t = torch.tensor(initial_values, dtype=self.dtype)

    def plot(self, data_obj: dict, show_noised: bool = True, show_true: bool = True,
             show_resamp: bool = True, show_extra: bool = True, figsize: tuple[int, int] = (8, 6),
             save: bool = True):
        raise NotImplementedError

    @staticmethod
    def plot_predictions(predictions: np.ndarray, targets: np.ndarray, t: torch.tensor, save: bool = True):
        raise NotImplementedError

    @staticmethod
    def get_dim() -> int:
        raise NotImplementedError

    @staticmethod
    def get_name() -> str:
        raise NotImplementedError

    def dynamics(self, t: torch.Tensor, state: torch.Tensor) -> str:
        raise NotImplementedError

    @staticmethod
    def get_method() -> str:
        raise NotImplementedError

    def get_rtol(self) -> float:
        return self.r_tol

    def get_atol(self) -> float:
        return self.a_tol

    def _create_dict(self, t_train: np.ndarray, x_train_noised: np.ndarray, x_train_true: np.ndarray,
                     t_test_extra: np.ndarray, x_test_extra_true: np.ndarray,
                     t_test_resamp: np.ndarray, x_test_resamp_true: np.ndarray, x_test_resamp_noised: np.ndarray) -> DataObject:
        return {
            "t_train": t_train,
            "x_train_noised": x_train_noised,
            "x_train_true": x_train_true,
            "t_test_extra": t_test_extra,
            "x_test_extra_true": x_test_extra_true,
            "t_test_resamp": t_test_resamp,
            "x_test_resamp_true": x_test_resamp_true,
            "x_test_resamp_noised": x_test_resamp_noised
        }

    def __call__(self, method: str, dynamics: Callable, seed: int = 0) -> DataObject:
        rng = np.random.default_rng(seed)

        # contiene una lista. Mi permette di avere una foto per ogni istante temporale che passo ma tutti a partenza t=0
        t_train = np.sort(rng.uniform(*self.t_train_range, size=self.n_train))
        x_train_true = self._solve_at(t_grid_np=t_train, method=method, dynamics=dynamics)
        x_train_noised = x_train_true + self.noise_std * rng.standard_normal(x_train_true.shape)

        t_test_extra = np.linspace(*self.t_extra_range, self.n_extra)
        x_test_extra_true = self._solve_at(t_grid_np=t_test_extra, method=method, dynamics=dynamics)

        t_test_resamp = np.sort(rng.uniform(*self.t_train_range, size=self.n_resamp))
        x_test_resamp_true = self._solve_at(t_grid_np=t_test_resamp, method=method, dynamics=dynamics)
        x_test_resamp_noised = x_test_resamp_true + self.noise_std * rng.standard_normal(x_test_resamp_true.shape)
        return self._create_dict(t_train, x_train_noised, x_train_true, t_test_extra, x_test_extra_true, t_test_resamp, x_test_resamp_true, x_test_resamp_noised)

    def _solve_at(self, t_grid_np: np.ndarray, method: str, dynamics: Callable) -> np.ndarray:
        t_full = np.concatenate([[0.0], np.atleast_1d(t_grid_np)])
        t_full = np.unique(t_full)

        t_tensor = torch.tensor(t_full, dtype=self.dtype)
        sol = odeint(dynamics, self.x0_t, t_tensor, method=method, rtol=self.r_tol, atol=self.a_tol)
        sol_np = sol.detach().cpu().numpy()

        mask = np.isin(t_full, t_grid_np)
        return sol_np[mask]

    def _build_multi_horizon_transitions(self, t: np.ndarray, x_in: np.ndarray, x_true: np.ndarray, K_max: int = 5, device: str = "cpu"
    ) -> TensorDataset:
        if t.size < 2:
            raise ValueError("Servono almeno 2 campioni temporali.")
        order = np.argsort(t)
        t_sorted = t[order]
        x_in_sorted = x_in[order]
        x_true_sorted = x_true[order]

        N = len(t_sorted)
        starts = np.arange(N - 1, dtype=np.int64)
        ks = np.random.randint(1, K_max + 1, size=starts.shape[0])
        ends = np.minimum(starts + ks, N - 1)

        x_i_np = x_in_sorted[starts]
        x_tgt_np = x_true_sorted[ends]
        t_i_np = t_sorted[starts]
        t_tgt_np = t_sorted[ends]
        dt_np = (t_tgt_np - t_i_np)[:, None]

        x_i = torch.tensor(x_i_np, dtype=self.dtype, device=device)
        x_tgt = torch.tensor(x_tgt_np, dtype=self.dtype, device=device)
        dt = torch.tensor(dt_np, dtype=self.dtype, device=device)
        t_i = torch.tensor(t_i_np, dtype=self.dtype, device=device)
        t_tgt = torch.tensor(t_tgt_np, dtype=self.dtype, device=device)

        return TensorDataset(x_i, x_tgt, dt, t_i, t_tgt)

    def _build_pairwise_transitions(self, t: np.ndarray, x_in: np.ndarray, x_out: np.ndarray, device: str = "cpu") -> TensorDataset:
        if t.size < 2:
            raise ValueError("Servono almeno 2 campioni temporali per creare transizioni i->i+1.")

        order = np.argsort(t)
        t_sorted = t[order]
        x_in_sorted = x_in[order]
        x_out_sorted = x_out[order]

        x_i_np = x_in_sorted[:-1]
        x_next_np = x_out_sorted[1:]
        t_i_np = t_sorted[:-1]
        t_next_np = t_sorted[1:]
        dt_np = (t_next_np - t_i_np)[:, None]

        x_i = torch.tensor(x_i_np, dtype=self.dtype, device=device)
        x_next = torch.tensor(x_next_np, dtype=self.dtype, device=device)
        t_i = torch.tensor(t_i_np, dtype=self.dtype, device=device)
        t_next = torch.tensor(t_next_np, dtype=self.dtype, device=device)
        dt = torch.tensor(dt_np, dtype=self.dtype, device=device)

        dataset = TensorDataset(x_i, x_next, dt, t_i, t_next)
        return dataset

    def prepare_data(self, data: DataObject, device: str = "cpu",
                     batch_size: Optional[int] = None, shuffle: bool = True,
                     use_noise: bool = True, train_K_max: Optional[int] = None) -> Loaders:
        out = {}

        if train_K_max is None:
            ds_train = self._build_pairwise_transitions(data["t_train"], data["x_train_noised" if use_noise else "x_train_true"],
                                                        data["x_train_true"], device=device)
        else:
            ds_train = self._build_multi_horizon_transitions(data["t_train"], data["x_train_noised" if use_noise else "x_train_true"],
                                                             data["x_train_true"], K_max=train_K_max, device=device)
        out["train"] = {"dataset": DataLoader(ds_train, batch_size=batch_size, shuffle=shuffle)}

        ds_resamp = self._build_pairwise_transitions(data["t_test_resamp"], data["x_test_resamp_noised" if use_noise else "x_test_resamp_true"], data["x_test_resamp_true"], device=device)
        out["resamp"] = {"dataset": DataLoader(ds_resamp, batch_size=batch_size, shuffle=False)}

        ds_extra = self._build_pairwise_transitions(data["t_test_extra"], data["x_test_extra_true"], data["x_test_extra_true"], device=device)
        out["extra"] = {"dataset": DataLoader(ds_extra, batch_size=batch_size, shuffle=False)}

        return out

    @staticmethod
    def save(loaders: Loaders, path: str):
        obj = {}
        for key, val in loaders.items():
            ds: TensorDataset = val["dataset"].dataset
            obj[key] = [t.detach().cpu() for t in ds.tensors]  # salva solo i tensori
        torch.save(obj, path)

    @staticmethod
    def load(path: str, batch_size: Optional[int] = None) -> Loaders:
        obj = torch.load(path, map_location="cpu")
        out = {}
        for key, tensors in obj.items():
            ds = TensorDataset(*tensors)
            dl = DataLoader(ds, batch_size=batch_size)
            out[key] = {"dataset": dl}
        return out

    @staticmethod
    def create_video_from_images(image_paths: List[str], fn: str, fps: int = 12, size: Optional[Tuple[int, int]] = None, sort_paths: bool = True) -> None:
        if not image_paths:
            raise ValueError("La lista delle immagini è vuota.")

        if sort_paths:
            image_paths = sorted(image_paths)

        first = cv2.imread(image_paths[0], cv2.IMREAD_UNCHANGED)
        if first is None:
            raise ValueError(f"Impossibile leggere l'immagine: {image_paths[0]}")

        if first.shape[-1] == 4:
            first = cv2.cvtColor(first, cv2.COLOR_BGRA2BGR)

        h0, w0 = first.shape[:2]
        if size is None:
            size = (w0, h0)

        writer = cv2.VideoWriter(fn, -1, fps, size)

        if not writer.isOpened():
            raise RuntimeError("Impossibile aprire il VideoWriter. Verifica codec/estensione file.")

        try:
            for p in image_paths:
                img = cv2.imread(p, cv2.IMREAD_UNCHANGED)
                if img is None:
                    raise ValueError(f"Impossibile leggere l'immagine: {p}")

                if img.ndim == 3 and img.shape[-1] == 4:
                    img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

                if (img.shape[1], img.shape[0]) != size:
                    img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)

                writer.write(img)
        finally:
            writer.release()

    @staticmethod
    def print_dataset_with_times(ds: DataLoader):
        x_i, x_next, dt, t_i, t_next = ds.dataset.tensors
        t_i = t_i.squeeze(-1) if t_i.ndim > 1 else t_i
        t_next = t_next.squeeze(-1) if t_next.ndim > 1 else t_next

        print(f"{'idx':<5} {'t_i':<8} {'x_i':<20} {'t_next':<8} {'x_next':<20} {'dt':<8}")
        print("-" * 80)
        for idx in range(len(t_i)):
            xi_str = np.array2string(x_i[idx].cpu().numpy(), precision=3, suppress_small=True, max_line_width=20)
            xnext_str = np.array2string(x_next[idx].cpu().numpy(), precision=3, suppress_small=True, max_line_width=20)

            print(f"{idx:<5} "
                  f"{float(t_i[idx]):<8.3f} {xi_str:<20} "
                  f"{float(t_next[idx]):<8.3f} {xnext_str:<20} "
                  f"{float(dt[idx]):<8.3f}")






