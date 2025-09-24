import time, torch, torch.nn as nn, numpy as np, os, matplotlib.pyplot as plt
from torchdiffeq import odeint_adjoint as odeint
from torch.optim import Adam

model_folder = "models"
plot_folder = "plots"

optimizers = {
    "mlp": Adam,
    "node": Adam,
}

def resume_model(model, resume, device, opt):
    start_epoch, best_val = 0, float("inf")
    if resume:
        try:
            start_epoch, best_val = load_checkpoint(model, opt, resume, device)
            print(f"Loaded Model {resume}")
        except Exception as e:
            print(e)
            start_epoch, best_val = 0, float("inf")
    return start_epoch, best_val

def get_model_paths(model):
    path = f"{model_folder}/{model.name}/"
    os.makedirs(path, exist_ok=True)
    best_path = f"{path}best.pt"
    last_path = f"{path}last.pt"
    return best_path, last_path

def save_rollout_plot(t, pred, tgt, path, fn):
    os.makedirs(path, exist_ok=True)
    fig, axs = plt.subplots(1, 3, figsize=(12, 3.6))

    axs[0].plot(t, tgt[:, 0], label="true")
    axs[0].plot(t, pred[:, 0], "--", label="pred")
    axs[0].set_xlabel("t"); axs[0].set_ylabel("x1")

    axs[1].plot(t, tgt[:, 1], label="true")
    axs[1].plot(t, pred[:, 1], "--", label="pred")
    axs[1].set_xlabel("t"); axs[1].set_ylabel("x2")

    axs[2].plot(tgt[:, 0], tgt[:, 1], label="true")
    axs[2].plot(pred[:, 0], pred[:, 1], "--", label="pred")
    axs[2].set_xlabel("x1"); axs[2].set_ylabel("x2"); axs[2].set_title("phase")

    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper left", ncol=2, frameon=False)

    epoch = fn.split("_")[-1].split(".")[0]
    fig.suptitle(f"Epoch {epoch}", fontsize=14)

    plt.tight_layout()
    plt.savefig(path + fn)
    plt.close(fig)

def load_checkpoint(model, optimizer, path, device):
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    optimizer.load_state_dict(ckpt["optimizer_state"])

    for state in optimizer.state.values():
        for k, v in state.items():
            if torch.is_tensor(v):
                state[k] = v.to(device)

    start_epoch = ckpt.get("epoch", -1) + 1
    best_val = ckpt.get("val_loss", float("inf"))
    print(f"[resume] from {path}: start_epoch={start_epoch}, best_val={best_val:.6f}")
    return start_epoch, best_val

### MLP ###

@torch.no_grad()
def eval_mlp(loader, model, path, fn, device="cpu", per_element=True):
    model.eval().to(device)
    ds = loader.dataset
    x_i, x_next, _dt, t_i, t_next = ds.tensors
    t_i, t_next = t_i.squeeze(-1), t_next.squeeze(-1)
    t_grid = torch.cat([t_i[:1], t_next], 0).to(device)
    x = x_i[0].to(device).unsqueeze(0)
    traj = [x.squeeze(0).cpu().numpy()]
    for k in range(t_grid.size(0)-1):
        dt = (t_grid[k+1]-t_grid[k]).view(1,1).to(device).to(x.dtype)
        x = model(x, dt)
        traj.append(x.squeeze(0).cpu().numpy())
    pred = np.stack(traj, 0)
    tgt  = torch.cat([x_i[:1], x_next], 0).cpu().numpy()
    save_rollout_plot(t_grid.detach().cpu().numpy(), pred, tgt, path, fn)
    denom = tgt.size if per_element else tgt.shape[0]
    return float(np.sum((pred - tgt)**2) / max(denom, 1))

def train_mlp(model, train_loader, resamp_loader, device: str ="cpu", epochs: int = 50, lr: float = 1e-3, resume: str = None):
    if epochs == 0: return
    model.to(device)
    crit = nn.MSELoss()
    opt = optimizers["mlp"](**{"params": model.parameters(), "lr": lr})
    start_epoch, best_val = resume_model(model=model, opt=opt, resume=resume, device=device)

    best_path, last_path = get_model_paths(model)

    t0 = time.perf_counter()
    for i in range(start_epoch, start_epoch + epochs):
        model.train()
        epoch_loss = 0.0
        for x_i, x_next, dt, t_i, t_next in train_loader:
            x_i, x_next, dt = x_i.to(device), x_next.to(device), dt.to(device)
            opt.zero_grad()
            loss = crit(model(x_i, dt), x_next)
            loss.backward()
            opt.step()
            epoch_loss += loss.item() * (x_i.shape[0] if x_i.ndim > 1 else 1)

        train_avg = epoch_loss / len(train_loader.dataset)
        val_avg = eval_mlp(loader=resamp_loader, model=model, device=device,
                           path=f"{plot_folder}/{model.name}/", fn=f"epoch_{i:04d}.png")

        chk = {"epoch": i, "model_state": model.state_dict(), "optimizer_state": opt.state_dict(), "val_loss": val_avg, "train_loss": train_avg,}

        torch.save(chk, last_path)

        if val_avg < best_val:
            best_val = val_avg
            torch.save(chk, best_path)
            print(f"[ckpt] new best at epoch {i}: train={train_avg:.6f}, resamp={val_avg:.6f}")

        print(f"Epoch {i:03d} | train {train_avg:.6f} | resamp {val_avg:.6f}")

    print(f"MLP Train Done in {time.perf_counter() - t0:.3f}s")
    resume_model(model=model, opt=opt, resume=f"{model_folder}/{model.name}/best.pt", device=device)
    return model

@torch.no_grad()
def predict_mlp(model, loader, device="cpu") -> tuple[np.ndarray, np.ndarray]:
    model.eval().to(device)
    ds = loader.dataset
    x_i, x_next, _dt, t_i, t_next = ds.tensors
    t_i, t_next = t_i.squeeze(-1), t_next.squeeze(-1)
    t_grid = torch.cat([t_i[:1], t_next], 0).to(device)
    x = x_i[0].to(device).unsqueeze(0)
    traj = [x.squeeze(0).cpu().numpy()]
    for k in range(t_grid.size(0)-1):
        dt = (t_grid[k+1]-t_grid[k]).view(1,1).to(device).to(x.dtype)
        x = model(x, dt)
        traj.append(x.squeeze(0).cpu().numpy())
    pred = np.stack(traj, 0)
    tgt  = torch.cat([x_i[:1], x_next], 0).cpu().numpy()
    return pred, tgt

### NODE ###

@torch.no_grad()
def eval_node(model, loader, rtol: float, atol: float, path, fn, device="cpu", per_element=True):
    model.eval().to(device)
    ds = loader.dataset
    x_i, x_next, _dt, t_i, t_next = ds.tensors
    t_i    = t_i.squeeze(-1) if t_i.ndim > 1 else t_i
    t_next = t_next.squeeze(-1) if t_next.ndim > 1 else t_next
    t_grid = torch.cat([t_i[:1], t_next], dim=0).to(device)

    x0 = x_i[0].to(device)
    traj = odeint(model, x0, t_grid.to(dtype=x0.dtype), method=model.method, rtol=rtol, atol=atol).detach().cpu().numpy()
    tgt  = torch.cat([x_i[:1], x_next], dim=0).cpu().numpy()

    save_rollout_plot(t_grid.detach().cpu().numpy(), traj, tgt, path, fn)

    denom = tgt.size if per_element else tgt.shape[0]
    return float(np.sum((traj - tgt)**2) / max(denom, 1))

def train_node(model, train_loader, resamp_loader, rtol: float, atol: float, device: str = "cpu",
               epochs: int = 50, lr: float = 1e-3, resume: str = None):
    if epochs == 0:
        return

    opts = {"step_size": getattr(model, "h", 0.01)} if getattr(model, "method", None) in ("rk4", "euler", "midpoint") else None

    model.to(device)
    crit = nn.MSELoss()
    opt = optimizers["node"](**{"params": model.parameters(), "lr": lr})
    start_epoch, best_val = resume_model(model=model, opt=opt, resume=resume, device=device)
    best_path, last_path = get_model_paths(model)

    t0 = time.perf_counter()
    for epoch in range(start_epoch, start_epoch + epochs):
        model.train()
        epoch_loss = 0.0

        for x_i, x_next, _dt, t_i, t_next in train_loader:
            x_i, x_next = x_i.to(device), x_next.to(device)
            t_i    = t_i.to(device).squeeze(-1)
            t_next = t_next.to(device).squeeze(-1)

            opt.zero_grad()
            preds = []
            for j in range(x_i.size(0)):
                tspan = torch.stack((t_i[j], t_next[j])).to(device=x_i.device, dtype=x_i.dtype)
                x_T   = odeint(model, x_i[j], tspan, method=model.method, rtol=rtol, atol=atol, options=opts,
                               adjoint_method=model.method, adjoint_rtol=rtol, adjoint_atol=atol, adjoint_options=opts)[-1]
                preds.append(x_T)
            y = torch.stack(preds, 0)
            loss = crit(y, x_next)
            loss.backward()
            opt.step()

            epoch_loss += loss.item() * (x_i.shape[0] if x_i.ndim > 1 else 1)

        train_avg = epoch_loss / len(train_loader.dataset)
        val_avg   = eval_node(model=model, loader=resamp_loader, rtol=rtol, atol=atol, device=device,
                              path=f"{plot_folder}/{model.name}/", fn=f"epoch_{epoch:04d}.png")

        chk = {"epoch": epoch, "model_state": model.state_dict(), "optimizer_state": opt.state_dict(), "val_loss": val_avg, "train_loss": train_avg,}

        torch.save(chk, last_path)

        if val_avg < best_val:
            best_val = val_avg
            torch.save(chk, best_path)
            print(f"[ckpt] new best at epoch {epoch}: train={train_avg:.6f}, resamp={val_avg:.6f}")

        print(f"Epoch {epoch:03d} | train {train_avg:.6f} | resamp {val_avg:.6f}")
    print(f"NODE Train Done in {time.perf_counter() - t0:.3f}s")
    resume_model(model=model, opt=opt, resume=f"{model_folder}/{model.name}/best.pt", device=device)
    return model

@torch.no_grad()
def predict_node(model, loader, rtol: float, atol: float, device: str = "cpu") -> tuple[np.ndarray, np.ndarray]:
    model.eval().to(device)
    ds = loader.dataset
    x_i, x_next, _dt, t_i, t_next = ds.tensors
    t_i    = t_i.squeeze(-1) if t_i.ndim > 1 else t_i
    t_next = t_next.squeeze(-1) if t_next.ndim > 1 else t_next
    t_grid = torch.cat([t_i[:1], t_next], dim=0).to(device)

    x0 = x_i[0].to(device)
    traj = odeint(model, x0, t_grid.to(dtype=x0.dtype), method=model.method, rtol=rtol, atol=atol).detach().cpu().numpy()
    tgt  = torch.cat([x_i[:1], x_next], dim=0).cpu().numpy()
    return traj, tgt

