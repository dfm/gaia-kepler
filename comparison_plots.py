#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function

import os
import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from gaia_kepler.data import KOICatalog

kic = pd.read_csv("gaia_kepler/data/kic_photo_match.csv")
kois = KOICatalog().df

fit_columns = ["mass", "feh"]
computed_columns = ["radius", "logg", "teff"]

for k in fit_columns + computed_columns:
    kic["new_"+k] = np.nan + np.zeros(len(kic))
    kic["new_"+k+"_err_minus"] = np.nan + np.zeros(len(kic))
    kic["new_"+k+"_err_plus"] = np.nan + np.zeros(len(kic))
kic["post_pred"] = np.nan + np.zeros(len(kic))

for ind, row in kic.iterrows():
    fn = "results/{0}.h5".format(row.kepid)
    if not os.path.exists(fn):
        continue
    with h5py.File(fn, "r") as f:
        fit_samples = f["fit_parameters"][...]
        computed_samples = f["computed_parameters"][...]
        mags = f["magnitudes"][...]
        bands = dict((b, (f.attrs[b+"_mag"], f.attrs[b+"_mag_err"]))
                     for b in mags.dtype.names)

    # Compute the posterior predictive check
    chi2 = np.zeros(len(mags))
    for b in mags.dtype.names:
        v, e = bands[b]
        chi2 += ((mags[b] - v) / e)**2
    row["post_pred"] = np.mean(chi2) / len(bands)

    for k in computed_columns:
        q = np.percentile(computed_samples[k], [16, 50, 84])
        row["new_"+k] = q[1]
        row["new_"+k+"_err_minus"] = q[1] - q[0]
        row["new_"+k+"_err_plus"] = q[2] - q[1]

    for k in fit_columns:
        q = np.percentile(fit_samples[k], [16, 50, 84])
        row["new_"+k] = q[1]
        row["new_"+k+"_err_minus"] = q[1] - q[0]
        row["new_"+k+"_err_plus"] = q[2] - q[1]

    kic.loc[ind] = row

kic = kic[~kic.new_mass.isnull()]
koi_match = pd.merge(kois, kic, on="kepid")

print("Found {0} stars".format(len(kic)))
print("and {0} exoplanet candidates".format(len(koi_match)))

koi_match["new_prad"] = np.nan + np.zeros(len(koi_match))
koi_match["new_prad_err_minus"] = np.nan + np.zeros(len(koi_match))
koi_match["new_prad_err_plus"] = np.nan + np.zeros(len(koi_match))
K = 1000
for ind, koi in koi_match.iterrows():
    fn = "results/{0}.h5".format(koi.kepid)
    with h5py.File(fn, "r") as f:
        computed_samples = f["computed_parameters"][...]
    sig = 0.5*(koi.koi_ror_err1-koi.koi_ror_err2)
    ror = koi.koi_ror+sig*np.random.randn(K)
    prad = (ror[:, None] * computed_samples["radius"][None, :]).flatten()
    prad = 109.1 * prad
    q = np.percentile(prad, [16, 50, 84])
    koi["new_prad"] = q[1]
    koi["new_prad_err_minus"] = q[1] - q[0]
    koi["new_prad_err_plus"] = q[2] - q[1]
    koi_match.loc[ind] = koi

fig, axes = plt.subplots(2, 3, figsize=(15, 10))

for k, ax in zip(computed_columns+fit_columns, axes.flat):
    rng = ((kic["new_"+k]-kic["new_"+k+"_err_minus"]).min(),
           (kic["new_"+k]+kic["new_"+k+"_err_plus"]).max())
    ax.plot(rng, rng, "k", alpha=0.3, lw=2)
    ax.errorbar(kic[k], kic["new_"+k], xerr=(-kic[k+"_err2"], kic[k+"_err1"]),
                yerr=(kic["new_"+k+"_err_minus"], kic["new_"+k+"_err_plus"]),
                fmt=".k", capsize=0, ms=0, zorder=-10000, alpha=0.5)
    ax.scatter(kic[k], kic["new_"+k], c="k", edgecolor="none", s=9)
    # ax.scatter(kic[k], kic["new_"+k], c=kic["post_pred"], cmap="Greens_r",
    #            edgecolor="none")
    if k == "radius":
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xticks([1, 2, 4, 8])
        ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
        ax.set_yticks([1, 2, 4, 8])
        ax.get_yaxis().set_major_formatter(plt.ScalarFormatter())
    else:
        ax.xaxis.set_major_locator(plt.MaxNLocator(4))
        ax.yaxis.set_major_locator(plt.MaxNLocator(5))
    ax.set_xlabel("KIC "+k)
    ax.set_ylabel("Gaia "+k)
    ax.set_xlim(rng)
    ax.set_ylim(rng)

ax = axes[1, 2]
k = "prad"
rng = (0.2, 12.0)
ax.plot(rng, rng, "k", alpha=0.3, lw=2)
ax.errorbar(koi_match["koi_prad"], koi_match["new_"+k],
            xerr=(-koi_match["koi_prad_err2"], koi_match["koi_prad_err1"]),
            yerr=(koi_match["new_"+k+"_err_minus"],
                  koi_match["new_"+k+"_err_plus"]),
            fmt=".k", capsize=0, ms=0, zorder=-10000, alpha=0.8)
ax.scatter(koi_match["koi_prad"], koi_match["new_"+k], c="k", edgecolor="none")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xticks([0.25, 0.5, 1, 2, 4, 8])
ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
ax.set_yticks([0.25, 0.5, 1, 2, 4, 8])
ax.get_yaxis().set_major_formatter(plt.ScalarFormatter())
ax.set_xlabel("KOI exoplanet radius")
ax.set_ylabel("updated exoplanet radius")
ax.set_xlim(rng)
ax.set_ylim(rng)

plt.tight_layout()
fig.savefig("results/comparison.png", dpi=300, bbox_inches="tight")
