#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function

import os
import sys
import h5py
import time
import tqdm
import logging
import argparse
import numpy as np
import matplotlib.pyplot as plt

import emcee3
import corner

from isochrones import StarModel
from isochrones.mist import MIST_Isochrone

from gaia_kepler import data


def fit_star(star, verbose=False):
    output_filename = "{0}.h5".format(star.kepid)
    if os.path.exists(output_filename):
        return

    strt = time.time()

    # The KIC parameters
    mean_log_mass = np.log(star.mass)
    sigma_log_mass = (np.log(star.mass+star.mass_err1) -
                      np.log(star.mass+star.mass_err2))  # double the kic value
    mean_feh = star.feh
    sigma_feh = star.feh_err1 - star.feh_err2  # double the kic value

    min_distance, max_distance = 0.0, 3000.0

    # Other bands
    bands = dict(
        J=(star.jmag, star.jmag_err),
        H=(star.hmag, star.hmag_err),
        K=(star.kmag, star.kmag_err),
    )
    if np.isfinite(star.tgas_w1gmag):
        bands = dict(
            W1=(star.tgas_w1gmag, max(star.tgas_w1gmag_error, 0.02)),
            W2=(star.tgas_w2gmag, max(star.tgas_w2gmag_error, 0.02)),
            W3=(star.tgas_w3gmag, max(star.tgas_w3gmag_error, 0.02)),
        )
    if np.isfinite(star.tgas_Vmag):
        bands["V"] = (star.tgas_Vmag, max(star.tgas_e_Vmag, 0.02))
    if np.isfinite(star.tgas_Bmag):
        bands["B"] = (star.tgas_Bmag, max(star.tgas_e_Bmag, 0.02))
    if np.isfinite(star.tgas_gpmag):
        bands["g"] = (star.tgas_gpmag, max(star.tgas_e_gpmag, 0.02))
    if np.isfinite(star.tgas_rpmag):
        bands["r"] = (star.tgas_rpmag, max(star.tgas_e_rpmag, 0.02))
    if np.isfinite(star.tgas_ipmag):
        bands["i"] = (star.tgas_ipmag, max(star.tgas_e_ipmag, 0.02))

    # Build the model
    mist = MIST_Isochrone()
    mod = StarModel(
        mist,
        parallax=(star.tgas_parallax, star.tgas_parallax_error),
        **bands
    )

    # Initialize
    nwalkers = 500
    ndim = 5
    lnpost_init = -np.inf + np.zeros(nwalkers)
    coords_init = np.empty((nwalkers, ndim))
    m = ~np.isfinite(lnpost_init)
    if verbose:
        print("Initializing...")
    while np.any(m):
        K = m.sum()

        # Mass
        coords_init[m, 0] = np.exp(mean_log_mass+sigma_log_mass*np.random.randn(K))

        # Age
        u = np.random.rand(K)
        coords_init[m, 1] = np.log((np.exp(mist.maxage)-np.exp(mist.minage))*u +
                                np.exp(mist.minage))

        # Fe/H
        coords_init[m, 2] = mean_feh + sigma_feh * np.random.randn(K)

        # Distance
        u = np.random.rand(K)
        coords_init[m, 3] = (u * (max_distance**3 - min_distance**3) +
                            min_distance**3)**(1./3)

        # Av
        coords_init[m, 4] = np.random.rand(K)

        lnpost_init[m] = np.array(list(map(mod.lnpost, coords_init[m])))
        m = ~np.isfinite(lnpost_init)
        if verbose:
            print("Resampling {0} points".format(m.sum()))

    class ICModel(emcee3.Model):

        def compute_log_prior(self, state):
            state.log_prior = mod.lnprior(state.coords)
            return state

        def compute_log_likelihood(self, state):
            state.log_likelihood = mod.lnlike(state.coords)
            return state

    sampler = emcee3.Sampler(emcee3.moves.KDEMove())
    ensemble = emcee3.Ensemble(ICModel(), coords_init)

    chunksize = 200
    targetn = 6
    for iteration in range(100):
        if verbose:
            print("Iteration {0}...".format(iteration + 1))
        sampler.run(ensemble, chunksize, progress=verbose)
        mu = np.mean(sampler.get_coords(), axis=1)
        try:
            tau = emcee3.autocorr.integrated_time(mu, c=1)
        except emcee3.autocorr.AutocorrError:
            continue
        tau_max = tau.max()
        neff = ((iteration+1) * chunksize / tau_max - 2.0)
        if verbose:
            print("Maximum autocorrelation time: {0}".format(tau_max))
            print("N_eff: {0}\n".format(neff * nwalkers))
        if neff > targetn:
            break

    burnin = int(2*tau_max)
    ntot = 5000
    if verbose:
        print("Discarding {0} samples for burn-in".format(burnin))
        print("Randomly choosing {0} samples".format(ntot))
    samples = sampler.get_coords(flat=True, discard=burnin)
    total_samples = len(samples)
    inds = np.random.choice(np.arange(len(samples)), size=ntot, replace=False)
    samples = samples[inds]

    fit_parameters = np.empty(len(samples), dtype=[
        ("mass", float), ("log10_age", float), ("feh", float), ("distance", float),
        ("av", float),
    ])
    computed_parameters = np.empty(len(samples), dtype=[
        ("radius", float), ("teff", float), ("logg", float),
    ])
    mags = np.empty(len(samples), dtype=[(b, float) for b in bands.keys()])

    if verbose:
        prog = tqdm.tqdm
    else:
        prog = lambda f, *args, **kwargs: f
    for i, p in prog(enumerate(samples), total=len(samples)):
        ic = mod.ic(*p)
        fit_parameters[i] = p
        computed_parameters[i] = (ic["radius"], ic["Teff"], ic["logg"])
        for b in bands.keys():
            mags[b][i] = ic[b+"_mag"]

    total_time = time.time() - strt
    print("emcee3 took {0} sec".format(total_time))

    with h5py.File(output_filename, "w") as f:
        f.attrs["kepid"] = int(star.kepid)
        f.attrs["neff"] = neff * nwalkers
        f.attrs["runtime"] = total_time
        f.attrs["total_samples"] = total_samples
        for k, (v, e) in bands.items():
            f.attrs[k+"_mag"] = v
            f.attrs[k+"_mag_err"] = e
        f.create_dataset("fit_parameters", data=fit_parameters)
        f.create_dataset("computed_parameters", data=computed_parameters)
        f.create_dataset("magnitudes", data=mags)

    # Plot
    fig = corner.corner(samples)
    fig.savefig("corner-{0}.png".format(star.kepid))
    plt.close(fig)

parser = argparse.ArgumentParser()
parser.add_argument("number", type=int)
parser.add_argument("--verbose", action="store_true")
args = parser.parse_args()

# Load the data
kic_tgas = data.KICPhotoXMatchCatalog().df
kic_tgas["parallax_snr"] = kic_tgas.tgas_parallax/kic_tgas.tgas_parallax_error
kic_tgas = kic_tgas.sort_values("parallax_snr", ascending=False)
kic_tgas = kic_tgas[kic_tgas.parallax_snr > 10.0]
star = kic_tgas.iloc[args.number]
fit_star(star, verbose=args.verbose)
