#!/usr/bin/env python3
"""
nn_mlp_estimator.py

Train a feedforward MLP to map (u,v,r,Fx,Fy,Nz) -> (u_dot,v_dot,r_dot).
Outputs:
  - Final train/val loss
  - Saved NN weights (parameter estimates)
  - Residuals (mean prediction error per DOF)
  - Training settings log
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path
import torch
from torch import nn
import json

# ===== USER SETTINGS =====
DATA_DIR   = "./../khai-data-processing/processed-data"
PATTERN    = "*.csv"
VAL_SPLIT  = 0.2
SEED       = 0

EPOCHS     = 2000
LR         = 1e-3
HIDDEN_DIM = 64

# Thruster geometry (meters)
A_SPACING  = 0.40
B_SPACING  = 0.90

# Output directories
TRAINING_DIR = "./nn2-training-settings"
RESULTS_DIR  = "./nn2-parameter-estimates"

os.makedirs(TRAINING_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
# ==========================

######### MAKE DATASET #########

# Build B once (geometry constants)
B = np.array([
    [1.0,  1.0,  0.0,  0.0],
    [0.0,  0.0,  1.0,  1.0],
    [A_SPACING/2.0, -A_SPACING/2.0, B_SPACING/2.0, -B_SPACING/2.0],
], dtype=float)

# Find files
base = Path(DATA_DIR).resolve()
files = sorted(base.rglob(PATTERN))
if not files:
    raise FileNotFoundError(f"No files matched {PATTERN} under {base}")

all_V, all_Vdot, all_Tau = [], [], []

for f in files:
    df = pd.read_csv(f)

    if "time_s" in df.columns:
        t = df["time_s"].to_numpy(float)
    elif "time" in df.columns:
        t = df["time"].to_numpy(float)
    else:
        raise ValueError(f"{f} missing time column")

    v = df[["u","v","r"]].to_numpy(float)
    thr = df[["f1","f2","f3","f4"]].to_numpy(float)

    if len(df) < 3:
        continue

    # central diff for vdot
    dt = (t[2:] - t[:-2])[:,None]
    dv = v[2:] - v[:-2]
    vdot = dv / dt

    # align mid samples and compute tau
    v_mid   = v[1:-1]
    thr_mid = thr[1:-1]
    tau_mid = thr_mid @ B.T   # (N,4) @ (4,3) -> (N,3)

    all_V.append(v_mid)
    all_Vdot.append(vdot)
    all_Tau.append(tau_mid)

V    = np.vstack(all_V)     # (N,3)
Vdot = np.vstack(all_Vdot)  # (N,3)
Tau  = np.vstack(all_Tau)   # (N,3)

# Split train/val
N = V.shape[0]
rng = np.random.default_rng(SEED)
idx = np.arange(N)
rng.shuffle(idx)
split = int((1.0 - VAL_SPLIT) * N)
tr, va = idx[:split], idx[split:]

V_tr, Vdot_tr, Tau_tr = V[tr], Vdot[tr], Tau[tr]
V_va, Vdot_va, Tau_va = V[va], Vdot[va], Tau[va]

print("total samples:", N)
print("train:", V_tr.shape[0], "val:", V_va.shape[0])

######### BUILD MLP MODEL #########

# Training tensors
X_tr = np.hstack([V_tr, Tau_tr])   # (N,6)
X_va = np.hstack([V_va, Tau_va])   # (N,6)

X_tr_t = torch.from_numpy(X_tr).double()
Y_tr_t = torch.from_numpy(Vdot_tr).double()
X_va_t = torch.from_numpy(X_va).double()
Y_va_t = torch.from_numpy(Vdot_va).double()

class MLPDynamics(nn.Module):
    def __init__(self, input_dim=6, hidden_dim=64, output_dim=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    def forward(self, x):
        return self.net(x)

device = torch.device("cpu")
torch.set_default_dtype(torch.double)

model = MLPDynamics(input_dim=6, hidden_dim=HIDDEN_DIM, output_dim=3).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
loss_fn = nn.MSELoss()

######### TRAINING LOOP #########

for epoch in range(EPOCHS):
    model.train()
    pred = model(X_tr_t)
    loss = loss_fn(pred, Y_tr_t)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch+1) % 200 == 0:
        val_loss = loss_fn(model(X_va_t), Y_va_t).item()
        print(f"Epoch {epoch+1}/{EPOCHS}, train={loss.item():.6e}, val={val_loss:.6e}")

######### FINAL EVAL #########

with torch.no_grad():
    train_loss = loss_fn(model(X_tr_t), Y_tr_t).item()
    val_loss   = loss_fn(model(X_va_t), Y_va_t).item()

    pred_tr = model(X_tr_t)
    pred_va = model(X_va_t)

    res_train = torch.mean(pred_tr - Y_tr_t, dim=0).cpu().numpy()
    res_val   = torch.mean(pred_va - Y_va_t, dim=0).cpu().numpy()

print("\n=== Final Losses (MLP) ===")
print(f"train_loss = {train_loss:.6e}")
print(f"  val_loss = {val_loss:.6e}")

print("\n=== Mean Residuals (pred - exp) ===")
print(f"train: surge={res_train[0]:.3e}, sway={res_train[1]:.3e}, yaw={res_train[2]:.3e}")
print(f"  val: surge={res_val[0]:.3e}, sway={res_val[1]:.3e}, yaw={res_val[2]:.3e}")

######### PRINT FINAL PARAMETERS #########

print("\n=== Final Network Parameters (weights & biases) ===")
for name, param in model.named_parameters():
    print(f"{name}: shape={tuple(param.shape)}")
    print(param.detach().cpu().numpy())
    print()


######### SAVE OUTPUTS #########

# 1. Save training settings
with open(os.path.join(TRAINING_DIR, "train_settings_mlp1.json"), "w") as f:
    json.dump({
        "epochs": EPOCHS,
        "lr": LR,
        "hidden_dim": HIDDEN_DIM,
        "val_split": VAL_SPLIT,
        "seed": SEED,
    }, f, indent=2)

# 2. Save final model weights, losses, residuals
weights = {name: p.detach().cpu().numpy().tolist() for name,p in model.named_parameters()}
out = {
    "weights": weights,
    "train_loss": train_loss,
    "val_loss": val_loss,
    "residuals": {
        "train": {"surge": float(res_train[0]), "sway": float(res_train[1]), "yaw": float(res_train[2])},
        "val":   {"surge": float(res_val[0]),   "sway": float(res_val[1]),   "yaw": float(res_val[2])},
    }
}
with open(os.path.join(RESULTS_DIR, "mlp_results1.json"), "w") as f:
    json.dump(out, f, indent=2)

print(f"\nSaved training settings to {TRAINING_DIR}")
print(f"Saved estimates/loss/residuals to {RESULTS_DIR}")
