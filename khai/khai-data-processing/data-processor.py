# roboat_runtime.py
# Dependencies: pandas, numpy

from pathlib import Path
import ast
import numpy as np
import pandas as pd

# --- Quaternion (w,x,y,z) -> yaw (rad) ---
def quat_to_yaw(w, x, y, z):
    num = 2.0 * (w * z + x * y)
    den = 1.0 - 2.0 * (y * y + z * z)
    return np.arctan2(num, den)

def get_roboat_runtime_data(location: str, odom_filename: str):
    """
    MATLAB-consistent core: process ONE odometry file + its matching force file.

    Returns:
      time (N,) seconds since first thruster sample
      trajectory (N,6): [x, y, psi, u, v, r] in frozen local frame
      thrusters (N,4): [f1, f2, f3, f4]
    """
    loc = Path(location)
    odom_path = loc / odom_filename

    date_name = odom_filename.split("-odometry-navsat.csv")[0]
    force_path = loc / f"{date_name}-command_force.csv"

    if not odom_path.exists():
        raise FileNotFoundError(f"Odometry file not found: {odom_path}")
    if not force_path.exists():
        raise FileNotFoundError(f"Force file not found: {force_path}")

    # ---------- Read ODOMETRY ----------
    odo = pd.read_csv(odom_path, low_memory=False)

    cx = ".pose.pose.position.x"
    cy = ".pose.pose.position.y"
    qx = ".pose.pose.orientation.x"
    qy = ".pose.pose.orientation.y"
    qz = ".pose.pose.orientation.z"
    qw = ".pose.pose.orientation.w"
    ux = ".twist.twist.linear.x"
    uy = ".twist.twist.linear.y"
    rz = ".twist.twist.angular.z"

    x  = pd.to_numeric(odo[cx], errors="coerce").to_numpy()
    y  = pd.to_numeric(odo[cy], errors="coerce").to_numpy()
    q_x = pd.to_numeric(odo[qx], errors="coerce").to_numpy()
    q_y = pd.to_numeric(odo[qy], errors="coerce").to_numpy()
    q_z = pd.to_numeric(odo[qz], errors="coerce").to_numpy()
    q_w = pd.to_numeric(odo[qw], errors="coerce").to_numpy()
    u  = pd.to_numeric(odo[ux], errors="coerce").to_numpy()
    v  = pd.to_numeric(odo[uy], errors="coerce").to_numpy()
    r  = pd.to_numeric(odo[rz], errors="coerce").to_numpy()

    psi = quat_to_yaw(q_w, q_x, q_y, q_z)
    odom_time = pd.to_datetime(odo["time"], format="%Y/%m/%d/%H:%M:%S.%f")

    odometry_data = np.column_stack([x, y, psi, u, v, r])  # (M,6)

    # ---------- Read THRUSTERS ----------
    force = pd.read_csv(force_path, low_memory=False)
    thrust_time = pd.to_datetime(force["time"], format="%Y/%m/%d/%H:%M:%S.%f")

    if ".data" not in force.columns:
        raise KeyError(f"Expected '.data' column in {force_path.name}")
    tuples = force[".data"].astype(str).apply(lambda s: ast.literal_eval(s))
    thrusters_all = np.vstack(tuples.to_list()).astype(float)  # (K,4)

    # ---------- Sync ----------
    if len(odom_time) == 0 or len(thrust_time) == 0:
        raise ValueError("Empty odometry or thruster time series.")

    odo_ns = odom_time.view("int64").to_numpy()
    thr_ns = thrust_time.view("int64").to_numpy()

    sync_idx = int(np.searchsorted(odo_ns, thr_ns[0], side="left"))
    if sync_idx >= len(odo_ns):
        raise ValueError("Odometry starts after all thruster samples; cannot sync.")

    init_inertial = odometry_data[sync_idx, :]
    psi0 = init_inertial[2]

    idxs = np.searchsorted(odo_ns, thr_ns, side="left")
    valid = idxs < len(odometry_data)
    idxs = idxs[valid]
    thr_ns = thr_ns[valid]
    thrusters = thrusters_all[valid, :]

    odo_sel = odometry_data[idxs, :]  # (N,6)

    # ---------- Frozen local frame ----------
    pos = odo_sel[:, 0:3]              # [x, y, psi]
    pos0 = init_inertial[0:3]
    pos_rel = pos - pos0

    c, s = np.cos(psi0), np.sin(psi0)
    R = np.array([[ c,  s, 0.0],
                  [-s,  c, 0.0],
                  [0.0, 0.0, 1.0]], dtype=float)  # rotate by -psi0
    pos_rot = (R @ pos_rel.T).T

    vel = odo_sel[:, 3:6].copy()       # [u, v, r]

    trajectory = np.column_stack([pos_rot[:, 0], pos_rot[:, 1], pos_rot[:, 2],
                                  vel[:, 0],     vel[:, 1],     vel[:, 2]])

    # Marine sign flips
    trajectory[:, 1] *= -1.0  # y
    trajectory[:, 2] *= -1.0  # psi
    trajectory[:, 4] *= -1.0  # v
    trajectory[:, 5] *= -1.0  # r

    time = (thr_ns - thr_ns[0]) / 1e9
    return time, trajectory, thrusters, date_name


# ----------------- CONFIG: EDIT THESE -----------------
INPUT_DIR  = "./raw-data"       # folder with your CSVs
OUTPUT_DIR = "./processed-data" # folder to write outputs
# ------------------------------------------------------


if __name__ == "__main__":
    in_dir  = Path(INPUT_DIR)
    out_dir = Path(OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    # find all odom files like MATLAB's uigetfile('*-odometry-navsat.csv')
    odom_files = sorted(p.name for p in in_dir.glob("*-odometry-navsat.csv"))

    if not odom_files:
        print(f"[INFO] No files matching '*-odometry-navsat.csv' in {in_dir}")
    else:
        for odom_file in odom_files:
            try:
                t, traj, thr, base = get_roboat_runtime_data(in_dir, odom_file)

                # one CSV per run, named after the date/base like MATLAB pairing
                out_path = out_dir / f"{base}-roboat_runtime.csv"
                df_out = pd.DataFrame({
                    "time_s": t,
                    "x":  traj[:, 0],
                    "y":  traj[:, 1],
                    "psi": traj[:, 2],
                    "u":  traj[:, 3],
                    "v":  traj[:, 4],
                    "r":  traj[:, 5],
                    "f1": thr[:, 0],
                    "f2": thr[:, 1],
                    "f3": thr[:, 2],
                    "f4": thr[:, 3],
                })
                df_out.to_csv(out_path, index=False)
                print(f"[OK] {odom_file} → {out_path.name}  "
                      f"(rows={len(df_out)})")

            except Exception as e:
                print(f"[FAIL] {odom_file}: {e}")
