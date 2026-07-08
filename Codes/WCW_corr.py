import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(SCRIPT_DIR, "Training_codes", "IDR_library")

AROMATICS = ["F", "W", "Y"]
MIDDLE_POS = 5          # 1-indexed center of the 9-mer (fixed aromatic, skipped in Q)

# %% [markdown]
# ## WCW octanol hydrophobicity scale (W. C. Wimley, T. P. Creamer, and S. H. White, Biochemistry 35, 5109 (1996))

# %%
WCW_SCALE = {
    'A': +0.50, 'R': +1.81, 'N': +0.85, 'D': +3.64, 'C': -0.02,
    'E': +3.63, 'Q': +0.77, 'G': +1.15, 'H': +2.33, 'I': -1.12,
    'L': -1.25, 'K': +2.80, 'M': -0.67, 'F': -1.71, 'P': +0.14,
    'S': +0.46, 'T': +0.25, 'W': -2.09, 'Y': -0.71, 'V': -0.46,
}
RT               = 0.592   # kcal/mol at 298K
MOLAR_CORRECTION = 2.4     # kcal/mol, mole-fraction -> molar standard state

# Trained AroMIP parameters (same values as Codes/aromip.js).
PARAM_SETS = {
    "F": {
        "A": 1.0758068943930996, "C": 0.7515081115208946, "D": 0.4098861523565832,
        "E": 0.3983119844300048, "F": 8.1603925477323100, "G": 0.6354157796571205,
        "H": 0.7846691451344266, "I": 3.1491216060357460, "K": 1.1009251977458419,
        "L": 3.6046544612488423, "M": 2.8653319073560772, "N": 0.4803481572211203,
        "P": 1.7694984669210487, "Q": 0.4933503638424146, "R": 1.6585225219886708,
        "S": 0.9301680536402084, "T": 1.2417197033480494, "V": 2.0355593428760814,
        "W": 5.9063068700289690, "Y": 1.8801996532692813,
    },
    "W": {
        "A": 1.0206442786302903, "C": 0.6708211019772945, "D": 0.3925586374549908,
        "E": 0.4000948935406380, "F": 12.4315596752144760, "G": 0.5956514706239270,
        "H": 0.7191760324420514, "I": 2.9251386801066905, "K": 1.0112142249802383,
        "L": 4.0861358235697650, "M": 2.8506481939604393, "N": 0.4732539101826204,
        "P": 1.6093272013276745, "Q": 0.4767325148760477, "R": 1.4493178760361400,
        "S": 0.9053164338059225, "T": 1.1254876583549318, "V": 2.0417309116628624,
        "W": 11.5571567222502710, "Y": 1.9236800785696822,
    },
    "Y": {
        "A": 0.6230184781505200, "C": 0.0133352143216332, "D": 0.2214359322487689,
        "E": 0.3402031615640565, "F": 2.1219401227664230, "G": 0.3466014548722871,
        "H": 0.2072649540790862, "I": 1.7025466417820883, "K": 0.5832851280298242,
        "L": 1.5741986228306597, "M": 0.9020719922136957, "N": 0.2975325873058723,
        "P": 0.9559854615896372, "Q": 0.4220647091504225, "R": 1.1369649667516084,
        "S": 0.4189210011424207, "T": 0.5604827271090070, "V": 1.0606855747783748,
        "W": 1.7911226699514830, "Y": 1.2730210580996608,
    },
}

def load_params(aromatic):
    return PARAM_SETS[aromatic]

def wcw_dg(seq):
    """Additive WCW free energy (molar) for a 9-mer."""
    return sum(WCW_SCALE.get(aa, 0.0) for aa in seq) + MOLAR_CORRECTION

def aromip_Q(seq, param_map):
    """Q = product of trained q[aa] over the 8 context positions (skip center)."""
    product = 1.0
    for pos, aa in enumerate(seq, start=1):
        if pos == MIDDLE_POS:
            continue
        product *= max(1e-12, param_map.get(aa, 1.0))
    return product


def aromip_P(Q):
    """P = Q / (1 + Q)."""
    return Q / (1.0 + Q)

data = {}

for aro in AROMATICS:
    param_map = load_params(aro)
    df = pd.concat([
        pd.read_csv(f"{DATA_DIR}/{aro}_rows_train.csv", header=None, sep=r"\s+"),
        pd.read_csv(f"{DATA_DIR}/{aro}_rows_test.csv",  header=None, sep=r"\s+"),
    ], ignore_index=True)

    df = df[~df[0].str.contains("X")].reset_index(drop=True)

    seqs = df[0].tolist()
    Q  = np.array([aromip_Q(s, param_map) for s in seqs])
    P  = aromip_P(Q)
    lnQ = np.log(Q)
    dG  = np.array([wcw_dg(s) for s in seqs])

    data[aro] = pd.DataFrame({"sequence": seqs, "Q": Q, "P": P, "lnQ": lnQ, "dG": dG})
    #print(f"{aro}: n={len(data[aro])}")

colors = {"F": "#2166ac", "W": "#2166ac", "Y": "#2166ac"}

for aro in AROMATICS:
    d = data[aro]
    x = d["dG"].values
    y = -d["lnQ"].values

    fig, ax = plt.subplots(figsize=(6, 4))

    ax.scatter(x, y, s=5, alpha=0.3, color="blue", rasterized=True)

    m, b = np.polyfit(x, y, 1)
    xs = np.linspace(x.min(), x.max(), 200)
    ax.plot(xs, m * xs + b, color="red", lw=2, ls="--")

    ax.set_xlabel(
        r"$\boldsymbol{\Delta G}_{\mathbf{oct}}$ (kcal/mol)",
        fontsize=18,
        fontweight="bold"
    )

    ax.set_ylabel(
    r"$-\mathbf{ln}\,\boldsymbol{Q}$",
    fontsize=18
)

    ax.tick_params(axis="both", labelsize=18)
    ax.set_xlim([-10, 35])

    # Fix y-label distance from the axis
    ax.yaxis.set_label_coords(-0.1, 0.5)

    # Use fixed margins for every figure
    fig.subplots_adjust(
        left=0.22,
        right=0.96,
        bottom=0.22,
        top=0.94
    )

    fig.savefig(f"f1_{aro}.png", dpi=600)
    plt.show()
