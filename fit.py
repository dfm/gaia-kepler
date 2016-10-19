#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function

import os
import sys
import h5py
import time
import tqdm
import logging
import numpy as np
import matplotlib.pyplot as plt

import emcee3
import corner
from schwimmbad import MPIPool

from isochrones import StarModel
from isochrones.mist import MIST_Isochrone

from gaia_kepler import data

def fit_star(star, verbose=False):
    output_filename = "{0}.h5".format(star.kepid)
    logging.info("Output filename: {0}".format(output_filename))
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
    other_bands = dict()
    if np.isfinite(star.tgas_w1gmag):
        other_bands = dict(
            W1=(star.tgas_w1gmag, star.tgas_w1gmag_error),
            W2=(star.tgas_w2gmag, star.tgas_w2gmag_error),
            W3=(star.tgas_w3gmag, star.tgas_w3gmag_error),
        )
    if np.isfinite(star.tgas_Vmag):
        other_bands["V"] = (star.tgas_Vmag, star.tgas_e_Vmag)
    if np.isfinite(star.tgas_Bmag):
        other_bands["B"] = (star.tgas_Bmag, star.tgas_e_Bmag)
    if np.isfinite(star.tgas_gpmag):
        other_bands["g"] = (star.tgas_gpmag, star.tgas_e_gpmag)
    if np.isfinite(star.tgas_rpmag):
        other_bands["r"] = (star.tgas_rpmag, star.tgas_e_rpmag)
    if np.isfinite(star.tgas_ipmag):
        other_bands["i"] = (star.tgas_ipmag, star.tgas_e_ipmag)

    # Build the model
    mist = MIST_Isochrone()
    mod = StarModel(
        mist,
        J=(star.jmag, star.jmag_err),
        H=(star.hmag, star.hmag_err),
        K=(star.kmag, star.kmag_err),
        parallax=(star.tgas_parallax, star.tgas_parallax_error),
        **other_bands
    )

    # Initialize
    nwalkers = 500
    ndim = 5
    lnpost_init = -np.inf + np.zeros(nwalkers)
    coords_init = np.empty((nwalkers, ndim))
    m = ~np.isfinite(lnpost_init)
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
    targetn = 3
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
    total_samples = len(total_samples)
    inds = np.random.choice(np.arange(len(samples)), size=ntot, replace=False)
    samples = samples[inds]

    fit_parameters = np.empty(len(samples), dtype=[
        ("mass", float), ("log10_age", float), ("feh", float), ("distance", float),
        ("av", float),
    ])
    computed_parameters = np.empty(len(samples), dtype=[
        ("radius", float), ("teff", float), ("logg", float),
    ])

    if verbose:
        prog = tqdm.tqdm
    else:
        prog = lambda f, *args, **kwargs: f
    for i, p in prog(enumerate(samples), total=len(samples)):
        ic = mod.ic(*p)
        fit_parameters[i] = p
        computed_parameters[i] = (ic["radius"], ic["Teff"], ic["logg"])

    total_time = time.time() - strt
    logging.info("emcee3 took {0} sec".format(total_time))

    with h5py.File(output_filename, "w") as f:
        f.attrs["kepid"] = int(star.kepid)
        f.attrs["neff"] = neff * nwalkers
        f.attrs["runtime"] = total_time
        f.create_dataset("fit_parameters", data=fit_parameters)
        f.create_dataset("computed_parameters", data=computed_parameters)

    # Plot
    fig = corner.corner(samples)
    fig.savefig("corner-{0}.png".format(star.kepid))
    plt.close(fig)

with MPIPool() as pool:
    if not pool.is_master():
        pool.wait()
        sys.exit(0)

    # Load the data
    kic_tgas = data.KICPhotoXMatchCatalog().df
    kic_tgas["parallax_snr"] = kic_tgas.tgas_parallax/kic_tgas.tgas_parallax_error
    kic_tgas = kic_tgas.sort_values("parallax_snr", ascending=False)
    kic_tgas = kic_tgas[kic_tgas.parallax_snr > 10.0]

    # Fit in batches
    rows = [star for _, star in kic_tgas.iterrows()]
    print(rows)
    list(pool.map(fit_star, rows))
