#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function

import time
import tqdm
import numpy as np
import pandas as pd

import emcee3
import corner

from isochrones import StarModel
from isochrones.mist import MIST_Isochrone

# Load the data
kic_tgas = pd.read_csv("kic_photo_match.csv")
kic_tgas["parallax_snr"] = kic_tgas.tgas_parallax/kic_tgas.tgas_parallax_error
kic_tgas = kic_tgas.sort_values("parallax_snr")
kic_tgas = kic_tgas[kic_tgas.parallax_snr > 10.0]
star = kic_tgas.iloc[0]
print(star.parallax_snr)

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
print(other_bands)

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

# strt = time.time()
# mod.fit(basename="demo_")
# print("Multinest took {0} sec".format(time.time() - strt))

# samples = np.array(mod.samples[["mass_0_0", "age_0", "feh_0", "distance_0",
#                                 "AV_0"]])
# print(samples.shape)
# fig = corner.corner(samples)
# fig.savefig("corner-multi.png")

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
    print(m.sum())

class ICModel(emcee3.Model):

    def compute_log_prior(self, state):
        state.log_prior = mod.lnprior(state.coords)
        return state

    def compute_log_likelihood(self, state):
        state.log_likelihood = mod.lnlike(state.coords)
        return state

sampler = emcee3.Sampler(emcee3.moves.KDEMove())
ensemble = emcee3.Ensemble(ICModel(), coords_init)

strt = time.time()

chunksize = 200
targetn = 3
for iteration in range(100):
    print("Iteration {0}...".format(iteration + 1))
    sampler.run(ensemble, chunksize, progress=True)
    mu = np.mean(sampler.get_coords(), axis=1)
    try:
        tau = emcee3.autocorr.integrated_time(mu, c=1)
    except emcee3.autocorr.AutocorrError:
        continue
    tau_max = tau.max()
    neff = ((iteration+1) * chunksize / tau_max - 2.0)
    print("Maximum autocorrelation time: {0}".format(tau_max))
    print("N_eff: {0}\n".format(neff * nwalkers))
    if neff > targetn:
        break

burnin = int(2*tau_max)
ntot = 5000
print("Discarding {0} samples for burn-in".format(burnin))
print("Randomly choosing {0} samples".format(ntot))
samples = sampler.get_coords(flat=True, discard=burnin)
inds = np.random.choice(np.arange(len(samples)), size=ntot, replace=False)
samples = samples[inds]

fit_parameters = np.empty(len(samples), dtype=[
    ("mass", float), ("log10_age", float), ("feh", float), ("distance", float),
    ("av", float),
])
computed_parameters = np.empty(len(samples), dtype=[
    ("radius", float), ("teff", float), ("logg", float),
])

for i, p in tqdm.tqdm(enumerate(samples), total=len(samples)):
    ic = mod.ic(*p)
    fit_parameters[i] = p
    computed_parameters[i] = (ic["radius"], ic["Teff"], ic["logg"])

print("emcee3 took {0} sec".format(time.time() - strt))

# Plot
fig = corner.corner(samples)
fig.savefig("corner-emcee3.png")
