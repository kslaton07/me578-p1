#!/usr/bin/env python3

import os
import numpy as np
import pandas as pd
from pathlib import Path
import torch
from torch import nn
import json

#less epoch -> better

# ===== USER SETTINGS =====
DATA_DIR   = "./../khai-data-processing/processed-data"
PATTERN    = "*.csv"
VAL_SPLIT  = 0.2
SEED       = 0
EPOCHS     = 10
LR         = 0.05
WEIGHTS    = (1.0, 1.0, 1.0)   # relative weighting of surge/sway/yaw in loss
SAVE_JSON  = "./nn1-parameter-estimates/lambda-estimates6.json"
# =========================

###### SAVE TRAINING PARAMS FOR REFERENCE #########
with open("./nn1-training-settings/train_setting6.txt", "w") as f:
    f.write(f"epochs   = {EPOCHS}\n")
    f.write(f"lr       = {LR}\n")
    f.write(f"weights  = {WEIGHTS}\n")

######### MAKE DATASET #########

# Build B once (geometry constants)
a = 0.40   # transverse thruster spacing [m]
b = 0.90   # longitudinal thruster spacing [m]
B = np.array([
    [1.0,  1.0,  0.0,  0.0],
    [0.0,  0.0,  1.0,  1.0],
    [a/2.0, -a/2.0,  b/2.0, -b/2.0],
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

    # central diff
    dt = (t[2:] - t[:-2])[:,None]
    dv = v[2:] - v[:-2]
    vdot = dv / dt

    # align mid samples and compute tau = B·thr
    v_mid   = v[1:-1]
    thr_mid = thr[1:-1]
    tau_mid = thr_mid @ B.T   # (N,4) @ (4,3) -> (N,3)

    all_V.append(v_mid)
    all_Vdot.append(vdot)
    all_Tau.append(tau_mid)

V    = np.vstack(all_V)
Vdot = np.vstack(all_Vdot)
Tau  = np.vstack(all_Tau)

# Split
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

######### MAKE MODEL #########

device = torch.device("cpu")
torch.set_default_dtype(torch.double)

V_tr_t     = torch.from_numpy(V_tr).to(device)
Vdot_tr_t  = torch.from_numpy(Vdot_tr).to(device)
Tau_tr_t   = torch.from_numpy(Tau_tr).to(device)

V_va_t     = torch.from_numpy(V_va).to(device)
Vdot_va_t  = torch.from_numpy(Vdot_va).to(device)
Tau_va_t   = torch.from_numpy(Tau_va).to(device)

# --- Define model ---
class LambdaNet(nn.Module):
    def __init__(self):
        super().__init__()
        init_vals = {
            "m11": 15.0, "m22": 15.0, "m33": 15.0,
            "Xu":  5.0,  "Yv":  5.0,  "Nr":  5.0,
        }
        self.params = nn.ParameterDict({
            k: nn.Parameter(torch.tensor(v, dtype=torch.double))
            for k, v in init_vals.items()
        })
        self.softplus = nn.Softplus(beta=1.0)  # ensures positive params
        self.eps = 1e-6

    def forward(self):
        sp = self.softplus
        return {
            "m11": sp(self.params["m11"]) + self.eps,
            "m22": sp(self.params["m22"]) + self.eps,
            "m33": sp(self.params["m33"]) + self.eps,
            "Xu":  sp(self.params["Xu"])  + self.eps,
            "Yv":  sp(self.params["Yv"])  + self.eps,
            "Nr":  sp(self.params["Nr"])  + self.eps,
        }

    @torch.no_grad()
    def pretty(self):
        return {k: float(v.item()) for k, v in self().items()}

def vdot_model(params, v, tau):
    u, vv, r = v[:, 0], v[:, 1], v[:, 2]
    m11, m22, m33 = params["m11"], params["m22"], params["m33"]
    Xu, Yv, Nr    = params["Xu"], params["Yv"], params["Nr"]

    # C(v)v
    Cv1 = -m22 * vv * r
    Cv2 =  m11 * u  * r
    Cv3 = (m22 - m11) * u * vv

    # D(v)v
    Dv1 = Xu * u
    Dv2 = Yv * vv
    Dv3 = Nr * r

    r1 = tau[:, 0] - (Cv1 + Dv1)
    r2 = tau[:, 1] - (Cv2 + Dv2)
    r3 = tau[:, 2] - (Cv3 + Dv3)

    vdot1 = r1 / m11
    vdot2 = r2 / m22
    vdot3 = r3 / m33
    return torch.stack([vdot1, vdot2, vdot3], dim=1)

# --- Training setup ---
model = LambdaNet().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
W = torch.tensor(WEIGHTS, dtype=torch.double, device=device)

def loss_fn(pred, target):
    err = pred - target
    return torch.mean(torch.sum(W * err * err, dim=1))

# --- Training loop ---
for epoch in range(EPOCHS):
    optimizer.zero_grad(set_to_none=True)
    params = model()
    vdot_hat = vdot_model(params, V_tr_t, Tau_tr_t)
    loss = loss_fn(vdot_hat, Vdot_tr_t)
    loss.backward()
    optimizer.step()
    if (epoch+1) % 200 == 0:
        print(f"Epoch {epoch+1}/{EPOCHS}, train_loss={loss.item():.6e}")

# --- Final evaluation ---
with torch.no_grad():
    params = model()
    train_loss = loss_fn(vdot_model(params, V_tr_t, Tau_tr_t), Vdot_tr_t).item()
    val_loss   = loss_fn(vdot_model(params, V_va_t, Tau_va_t), Vdot_va_t).item()
    lam = model.pretty()

print("\n=== Estimated Parameters (λ) ===")
for k, v in lam.items():
    print(f"{k:>4s} = {v:.6f}")

print("\n=== Final Losses ===")
print(f"train_loss = {train_loss:.6e}")
print(f"  val_loss = {val_loss:.6e}")

# --- Residual errors (per DOF) ---
with torch.no_grad():
    vdot_tr_pred = vdot_model(model(), V_tr_t, Tau_tr_t)
    vdot_va_pred = vdot_model(model(), V_va_t, Tau_va_t)

    res_train = torch.mean(vdot_tr_pred - Vdot_tr_t, dim=0).cpu().numpy()
    res_val   = torch.mean(vdot_va_pred - Vdot_va_t, dim=0).cpu().numpy()

print("\n=== Mean Residuals (vdot_model - vdot_exp) ===")
print(f"train: surge={res_train[0]:.3e}, sway={res_train[1]:.3e}, yaw={res_train[2]:.3e}")
print(f"  val: surge={res_val[0]:.3e}, sway={res_val[1]:.3e}, yaw={res_val[2]:.3e}")

# --- Save results ---
out = {
    "lambda": lam,
    "train_loss": train_loss,
    "val_loss": val_loss,
    "residuals": {
        "train": {"surge": float(res_train[0]),
                  "sway":  float(res_train[1]),
                  "yaw":   float(res_train[2])},
        "val":   {"surge": float(res_val[0]),
                  "sway":  float(res_val[1]),
                  "yaw":   float(res_val[2])},
    }
}
with open(SAVE_JSON, "w") as f:
    json.dump(out, f, indent=2)
print(f"\nSaved estimates and residuals to {SAVE_JSON}")
