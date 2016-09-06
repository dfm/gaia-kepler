# -*- coding: utf-8 -*-

from __future__ import division, print_function

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as pl
from scipy.spatial import cKDTree as KDTree

__all__ = []


def load_tycho2(TYCHO2_FILENAME="data/tycho2.h5"):
    if not os.path.exists(TYCHO2_FILENAME):
        df = pd.read_csv("data/tycho2.tsv", sep="|", low_memory=False,
                         usecols=[0, 1, 21], names=["ra", "dec", "btmag"],
                         skiprows=67, dtype=np.float64,
                         na_values=["      "])
        df.to_hdf(TYCHO2_FILENAME, "tycho2")
    else:
        df = pd.read_hdf(TYCHO2_FILENAME, "tycho2")
    return df


def load_kepler():
    return pd.read_hdf("data/q1_q17_dr24_stellar.h5", "q1_q17_dr24_stellar")


def load_kois():
    return pd.read_hdf("data/q1_q17_dr24_koi.h5", "q1_q17_dr24_koi")


def convert_to_cartesian(radec):
    ra, dec = np.radians(radec[:, 0]), np.radians(radec[:, 1])
    xyz = np.empty((len(ra), 3))
    xyz[:, 0] = np.cos(dec) * np.cos(ra)
    xyz[:, 1] = np.cos(dec) * np.sin(ra)
    xyz[:, 2] = np.sin(dec)
    return xyz


if __name__ == "__main__":
    tol = np.radians(1.0 / 3600)  # arcsec -> deg -> rad

    print("Loading the Tycho-2 catalog...")
    tycho2 = load_tycho2()
    tycho2_xyz = convert_to_cartesian(np.array(tycho2[["ra", "dec"]],
                                               dtype=np.float64))

    print("Loading the Kepler catalog...")
    kepler = load_kepler()
    kepler_xyz = convert_to_cartesian(np.array(kepler[["ra", "dec"]],
                                               dtype=np.float64))

    print("Loading the KOI catalog...")
    kois = load_kois()
    kois = kois[kois.koi_pdisposition == "CANDIDATE"]

    # Building KD-tree.
    print("Building KD trees...")
    tycho2_tree = KDTree(tycho2_xyz)
    kepler_tree = KDTree(kepler_xyz)

    # Cross match.
    print("Cross matching tress...")
    match = kepler_tree.query_ball_tree(tycho2_tree, np.sqrt(2-2*np.cos(tol)))
    match_flag = np.zeros(len(kepler), dtype=bool)
    match_id = np.zeros(len(kepler), dtype=np.uint64)
    distances = np.nan + np.zeros(len(kepler))
    for i, m in enumerate(match):
        if len(m):
            # Compute the angular distance.
            d = np.arccos(np.dot(kepler_xyz[i], tycho2_xyz[m].T))
            distances[i] = np.min(d)
            if distances[i] <= tol:
                match_id[i] = m[np.argmin(d)]
                match_flag[i] = True

    kepler["tycho2_match_id"] = match_id
    kepler["tycho2_match_dist_deg"] = np.degrees(distances)
    kepler["tycho2_match_btmag"] = np.array(tycho2.btmag.iloc[match_id])
    kepler["tycho2_match_ra"] = np.array(tycho2.ra.iloc[match_id])
    kepler["tycho2_match_dec"] = np.array(tycho2.dec.iloc[match_id])
    kepler_matched = kepler.iloc[match_flag]
    kepler_matched.to_csv("matched.csv", index=False)

    fig, ax = pl.subplots(1, 1, figsize=(4, 4))
    # x = kepler_matched.kepmag
    x = kepler_matched.tycho2_match_dist_deg * 3600
    y = kepler_matched.jmag - kepler_matched.tycho2_match_btmag
    ax.plot(x, y, ".k", ms=2, rasterized=True)

    # ax.set_ylabel("kepmag - Tycho2 BTmag")
    # ax.set_xlabel("ang.\ sep.\ [arcsec]")

    # ax.plot(kepler.teff, kepler.logg, ".", color="k", ms=2, rasterized=True)
    # ax.plot(kepler_matched.teff, kepler_matched.logg, ".", color="g", ms=2,
    #         rasterized=True)
    # ax.set_xlim(9000, 3000)
    # ax.set_ylim(5.1, 0.0)
    # ax.set_ylabel("$\log g$")
    # ax.set_xlabel("$T_\mathrm{eff}$")

    ax.xaxis.set_major_locator(pl.MaxNLocator(4))
    ax.yaxis.set_major_locator(pl.MaxNLocator(5))
    fig.set_tight_layout(True)
    fig.savefig("cross_match.png", dpi=400, bbox_inches="tight")
    # ax.set_xlim(0, 0.25)
    # fig.savefig("cross_match_zoom.png", dpi=400, bbox_inches="tight")

    kois_matched = pd.merge(kepler_matched[["kepid"]], kois, on="kepid",
                            how="inner")
    print("{0} Kepler targets".format(len(kepler_matched)))
    print("{0} planet candidates".format(len(kois_matched)))
