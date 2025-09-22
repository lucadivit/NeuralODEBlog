import os.path, torch, shutil, glob

from src.HarmonicOscillatorData import HarmonicOscillator
from src.LotkaVolterraData import LotkaVolterra
from src.models import harmonic_oscillator_nets, lotka_volterra_nets
from src.train_pred import train_mlp, predict_mlp, train_node, predict_node, plot_folder, model_folder, resume_model, optimizers

BATCH_SIZE = 16
EPOCHS = 100
LR = 1e-3
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using Device {device}")
reset = False

problem = "harmonic-oscillator"
if problem == "lotka-volterra":
    DataCreator = LotkaVolterra
    params = {
        "n_train": 350,
        "noise_std": 0.01,
        "t_train_range": (0.0, 12.0),
        "t_extra_range": (12.0, 18.0),
        "n_extra": 300,
        "n_resamp": 200,
    }
    mlp, node = lotka_volterra_nets()
elif problem == "harmonic-oscillator":
    DataCreator = HarmonicOscillator
    params = {
        "n_train": 350,
        "noise_std": 0.01,
        "t_train_range": (0.0, 12.0),
        "t_extra_range": (12.0, 18.0),
        "n_extra": 200,
        "n_resamp": 200,
    }
    mlp, node = harmonic_oscillator_nets()
else:
    exit(1)

data_fn = f"{DataCreator.get_name()}.pt"
mlp_pred_fn = f"{DataCreator.get_name()}_predictions_mlp.svg"
node_pred_fn = f"{DataCreator.get_name()}_predictions_node.svg"

if reset:
    shutil.rmtree(model_folder, ignore_errors=True)
    shutil.rmtree(plot_folder, ignore_errors=True)
    if os.path.isfile(data_fn): os.remove(data_fn)
    if os.path.isfile(mlp_pred_fn): os.remove(mlp_pred_fn)
    if os.path.isfile(node_pred_fn): os.remove(node_pred_fn)

data_creator = DataCreator(**params)
if not os.path.isfile(f"{DataCreator.get_name()}.pt"):
    print(f"Creating Dataset for {DataCreator.get_name()}")
    data = data_creator(method=DataCreator.get_method(), dynamics=data_creator.dynamics)
    data = data_creator.prepare_data(data, batch_size=BATCH_SIZE, device=device, train_K_max=5)
    DataCreator.save(data, data_fn)
else:
    print(f"Loading Dataset for {DataCreator.get_name()}")
    data = DataCreator.load(data_fn, batch_size=BATCH_SIZE)

train_loader = data["train"]["dataset"]
resamp_loader = data["resamp"]["dataset"]
extra_loader = data["extra"]["dataset"]

ds = data["extra"]["dataset"].dataset
t_grid = torch.cat([ds.tensors[3][:1], ds.tensors[4]], dim=0).squeeze(-1).cpu().numpy()

mlp = mlp.to(device).to(dtype=data_creator.dtype)
node = node.to(device).to(dtype=data_creator.dtype)

# MLP
best_mlp = train_mlp(mlp, train_loader, resamp_loader, epochs=EPOCHS, lr=LR, device=device, resume=f"{model_folder}/{mlp.name}/last.pt")
preds_mlp, tgts_mlp = predict_mlp(model=best_mlp, loader=extra_loader, device=device)
DataCreator.plot_predictions(preds_mlp, tgts_mlp, t_grid, fn=mlp_pred_fn)
DataCreator.create_video_from_images(image_paths=glob.glob(f"{plot_folder}/{mlp.name}/*.png"), fn=f"{mlp.name}.mp4", fps=5)

# NODE
best_node = train_node(node, train_loader, resamp_loader, epochs=EPOCHS, lr=LR, device=device, resume=f"{model_folder}/{node.name}/last.pt",
                       rtol=data_creator.get_rtol(), atol=data_creator.get_atol())
preds_node, tgts_node = predict_node(model=best_node, loader=extra_loader, device=device, rtol=data_creator.get_rtol(), atol=data_creator.get_atol())
DataCreator.plot_predictions(preds_node, tgts_node, t_grid, fn=node_pred_fn)
DataCreator.create_video_from_images(image_paths=glob.glob(f"{plot_folder}/{node.name}/*.png"), fn=f"{node.name}.mp4", fps=5)

