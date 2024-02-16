import time

import numpy as np
import pandas as pd
import ray
import seaborn as sns
from matplotlib import pyplot as plt

try:
    from mine.mine import Mine
    from mine.mine_tools import Progress, loadmat
except ImportError:
    from .mine.mine import Mine
    from .mine.mine_tools import Progress, loadmat

sns.set_theme(style="ticks", palette="tab10", context="paper")
palette = sns.color_palette("Paired")

CYCLE = "Swing"
NREA = 10  # number of mine realizations

# ----------------------------------------------------------------------------
# paralelization ray
run_torch_on = "cuda"  # run on gpu
# run_torch_on = "cpu"  # run on cpu
ray.init(num_cpus=8)
time.sleep(10)


@ray.remote(num_gpus=0.125, max_calls=1)  # use this comment if you use gpu
# @ray.remote()                           # use this comment if you use gpu
def getMI(toeData, angleData, sim_id):
    model = Mine(toeData, angleData, **model_params, device=run_torch_on)
    model.fit(**training_params)
    progress.update.remote()
    return (sim_id, model.get_mi())


# ----------------------------------------------------------------------------
# load Data
data_fn = "./signals/gaitSignals.mat"
DATA = loadmat(data_fn)
angles = ["rknee", "lknee"]

# ----------------------------------------------------------------------------
# set model and training parameters
model_params = {
    "loss": "mine",
    "hidden_dim": 50,
    "num_hidden_layers": 3,
    "afn": "elu",
}
training_params = {
    "batch_size": "full",  # don't use mini-batches
    "max_epochs": 5000,
    "val_size": 0.2,
    "lr": 1e-3,
    "lr_factor": 0.5,
    "lr_patience": 100,
    "stop_patience": 300,
    "stop_min_delta": 0.0,
    "verbose": False,
}
# ----------------------------------------------------------------------------
# Set up simulations (NREA Mine instances per toe-angle)
sims = []
toeData_id = ray.put(DATA["rtoe"])
for angle in angles:
    angleData_id = ray.put(DATA[angle])
    angle_side = "L" if "l" in angle else "R"
    for _ in range(NREA):
        sim_params_ = {
            "toeData": toeData_id,
            "angleData": angleData_id,
            "sim_id": f"R-{angle_side}",
        }
        sims.append(sim_params_)
# ----------------------------------------------------------------------------
# run simulations
print(f"Running {NREA} MINE realizations per foot-angle pair")
progress = Progress.remote(len(sims), pbar=False)
res = []
for s in sims:
    time.sleep(0.5)
    res.append(getMI.remote(s["toeData"], s["angleData"], s["sim_id"]))
res = ray.get(res)
# ----------------------------------------------------------------------------
# compile results
results = []
results += [(sim_id, mi) for sim_id, mi in res]
results = pd.DataFrame(results, columns=["toeSide-angleSide", "mi"])
results["angle"] = "knee"
results["angle"] = ["knee"] * results.index.size

# ----------------------------------------------------------------------------
# final mi estimation
rr_mi = results[results["toeSide-angleSide"].isin(["R-R"])]["mi"].to_numpy()
rl_mi = results[results["toeSide-angleSide"].isin(["R-L"])]["mi"].to_numpy()
print(f"RTOE-RKNEE MI={np.median(rr_mi):.3f}")
print(f"RTOE-LKNEE MI={np.median(rl_mi):.3f}")

# ----------------------------------------------------------------------------
# results boxplot
fig1, ax1 = plt.subplots(1, 1, sharey=True)
sns.boxplot(
    data=results,
    x="angle",
    y="mi",
    hue="toeSide-angleSide",
    hue_order=["R-R", "R-L"],
    ax=ax1,
    palette=[palette[1], palette[3]],
)
ax1.legend(
    title="Foot side - Joint angle side",
    loc="upper right",
    borderpad=0.5,
    labelspacing=0.5,
    handlelength=1,
    handletextpad=1.5,
)
fig1.suptitle("MI between right toe heigth and right/left joint knee angles")
ax1.set(ylabel="mi")

# ----------------------------------------------------------------------------
# scatter plot
fig2, ax2 = plt.subplots(1, 2, sharey=True)
sns.scatterplot(
    data=DATA,
    x="rknee",
    y="rtoe",
    size=5,
    ax=ax2[0],
    legend=False,
    facecolors=palette[1],
)
sns.scatterplot(
    data=DATA,
    x="lknee",
    y="rtoe",
    size=5,
    ax=ax2[1],
    legend=False,
    facecolors=palette[3],
)
ax2[0].set_ylabel("Right toe height [mm]")
ax2[0].set_xlabel("Right knee angle [deg]")
ax2[1].set_ylabel("")
ax2[1].set_xlabel("Left knee angle [deg]")
fig2.suptitle(f"Scater Plots - {CYCLE} cycle")
plt.show()
