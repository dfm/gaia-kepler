#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function

import os
import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

kic = pd.read_csv("gaia_kepler/data/kic_photo_match.csv")

fit_columns = ["mass", "feh"]
computed_columns = ["radius", "logg", "teff"]

for k in fit_columns + computed_columns:
    kic["new_"+k] = np.nan + np.zeros(len(kic))
    kic["new_"+k+"_err_minus"] = np.nan + np.zeros(len(kic))
    kic["new_"+k+"_err_plus"] = np.nan + np.zeros(len(kic))

for ind, row in kic.iterrows():
    fn = "results/{0}.h5".format(row.kepid)
    if not os.path.exists(fn):
        continue
    print(fn)
    with h5py.File(fn, "r") as f:
        fit_samples = f["fit_parameters"][...]
        computed_samples = f["computed_parameters"][...]

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

fig, axes = plt.subplots(2, 3, figsize=(15, 10))

for k, ax in zip(computed_columns+fit_columns, axes.flat):
    rng = ((kic["new_"+k]-kic["new_"+k+"_err_minus"]).min(),
           (kic["new_"+k]+kic["new_"+k+"_err_plus"]).max())
    ax.plot(rng, rng, "g")
    ax.errorbar(kic[k], kic["new_"+k], xerr=(-kic[k+"_err2"], kic[k+"_err1"]),
                yerr=(kic["new_"+k+"_err_minus"], kic["new_"+k+"_err_plus"]),
                fmt=".k", capsize=0)
    ax.set_xlabel("KIC "+k)
    ax.set_ylabel("Gaia "+k)
    ax.set_xlim(rng)
    ax.set_ylim(rng)
    ax.xaxis.set_major_locator(plt.MaxNLocator(4))
    ax.yaxis.set_major_locator(plt.MaxNLocator(5))

plt.tight_layout()
fig.savefig("results/comparison.png", dpi=300, bbox_inches="tight")
