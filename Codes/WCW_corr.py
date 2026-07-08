import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

BASE    = os.path.dirname(os.path.abspath(os.getcwd()))  # adjust if run outside WW_correlation/
IDR_DIR = "/home/fidha/Aromatic_Prediction/AroMIP_github/HumanIDR"

AROMATICS = ["F", "W", "Y"]
MIDDLE_POS = 5          # 1-indexed center of the 9-mer (fixed aromatic, skipped in Q)

# %% [markdown]
# ## WW interfacial hydrophobicity scale (Wimley & White, 1996)

# %%
WW_SCALE = {
    'A': +0.50, 'R': +1.81, 'N': +0.85, 'D': +3.64, 'C': -0.02,
    'E': +3.63, 'Q': +0.77, 'G': +1.15, 'H': +2.33, 'I': -1.12,
    'L': -1.25, 'K': +2.80, 'M': -0.67, 'F': -1.71, 'P': +0.14,
    'S': +0.46, 'T': +0.25, 'W': -2.09, 'Y': -0.71, 'V': -0.46,
}
RT               = 0.592   # kcal/mol at 298K
MOLAR_CORRECTION = 2.4     # kcal/mol, mole-fraction -> molar standard state

def load_params(aromatic):
    df = pd.read_csv(f"{IDR_DIR}/results/{aromatic}_params.csv")
    aa_rows = df[df["AA"].str.len() == 1]
    return dict(zip(aa_rows["AA"], aa_rows["Value"]))

def ww_dg(seq):
    """Additive WW free energy (molar) for a 9-mer."""
    return sum(WW_SCALE.get(aa, 0.0) for aa in seq) + MOLAR_CORRECTION

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
        pd.read_csv(f"{IDR_DIR}/data/{aro}_rows_train.csv", header=None, sep=r"\s+"),
        pd.read_csv(f"{IDR_DIR}/data/{aro}_rows_test.csv",  header=None, sep=r"\s+"),
    ], ignore_index=True)

    df = df[~df[0].str.contains("X")].reset_index(drop=True)

    seqs = df[0].tolist()
    Q  = np.array([aromip_Q(s, param_map) for s in seqs])
    P  = aromip_P(Q)
    lnQ = np.log(Q)
    dG  = np.array([ww_dg(s) for s in seqs])

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
