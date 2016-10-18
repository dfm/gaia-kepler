# -*- coding: utf-8 -*-
"""
Code for interfacing with the Exoplanet Archive catalogs.

"""

from __future__ import division, print_function

import os
import logging
from pkg_resources import resource_filename

import pandas as pd

from six.moves import urllib

__all__ = [
    "KICatalog",             # Kepler input catalog

    "EBCatalog",             # Villanova EB catalog
    "KOICatalog",            # DR24 KOI catalog
    "CumulativeCatalog",     # Cumulative KOI catalog
    "ExoplanetCatalog",      # Confirmed planet catalog

    "EPICatalog",            # K2 input catalog
    "K2CandidatesCatalog",   # K2 input catalog

    "AsteroGiantCatalog",    # Pinsonneault+ (2014) asterosesmic catalog

    "TGASDistancesCatalog",  # Distance estimate catalog
]


GAIA_KEPLER_DATA = os.environ.get(
    "GAIA_KEPLER_DATA",
    os.path.expanduser(os.path.join("~", ".gaia_kepler_data"))
)


def download(clobber=False):
    for c in (KICatalog, KOICatalog, CumulativeCatalog, ExoplanetCatalog,
              EPICatalog, K2CandidatesCatalog, TGASDistancesCatalog):
        print("Downloading {0}...".format(c.cls.__name__))
        c().fetch(clobber=clobber)


class Catalog(object):

    url = None
    name = None
    ext = ".h5"

    def __init__(self, data_root=None):
        self.data_root = GAIA_KEPLER_DATA if data_root is None else data_root
        self._df = None
        self._spatial = None

    @property
    def filename(self):
        if self.name is None:
            raise NotImplementedError("subclasses must provide a name")
        return os.path.join(self.data_root, "catalogs", self.name + self.ext)

    def fetch(self, clobber=False):
        # Check for a local file first.
        fn = self.filename
        if os.path.exists(fn) and not clobber:
            logging.info("Found local file: '{0}'".format(fn))
            return

        # Fetch the remote file.
        if self.url is None:
            raise NotImplementedError("subclasses must provide a URL")
        url = self.url
        logging.info("Downloading file from: '{0}'".format(url))
        r = urllib.request.Request(url)
        handler = urllib.request.urlopen(r)
        code = handler.getcode()
        if int(code) != 200:
            raise CatalogDownloadError(code, url, "")

        # Make sure that the root directory exists.
        try:
            os.makedirs(os.path.split(fn)[0])
        except os.error:
            pass

        self._save_fetched_file(handler)

    def _save_fetched_file(self, file_handle):
        raise NotImplementedError("subclasses must implement this method")

    @property
    def df(self):
        if self._df is None:
            if not os.path.exists(self.filename):
                self.fetch()
            self._df = pd.read_hdf(self.filename, self.name)
        return self._df


class TGASDistancesCatalog(Catalog):
    name = "tgas_dist"
    url = ("http://www2.mpia-hd.mpg.de/homes/calj/tgas_distances"
           "/tgas_dist_all_withsys_v01.csv.gz")

    def _save_fetched_file(self, file_handle):
        names = [
            "HIPId", "Tycho2", "SourceId", "LDeg", "BGed", "varpi",
            "errVarpi", "GMag", "rMoExp1", "r5Exp1", "r50Exp1", "r95Exp1",
            "sigmaRExp1", "rMoExp2", "r5Exp2", "r50Exp2", "r95Exp2",
            "sigmaRExp2", "rMoMW", "r5MW", "r50MW", "r95MW", "sigmaRMW",
        ]
        df = pd.read_csv(file_handle, compression="gzip", header=0,
                         names=names)
        df.to_hdf(self.filename, self.name, format="t")


class ExoplanetArchiveCatalog(Catalog):

    @property
    def url(self):
        if self.name is None:
            raise NotImplementedError("subclasses must provide a name")
        return ("http://exoplanetarchive.ipac.caltech.edu/cgi-bin/nstedAPI/"
                "nph-nstedAPI?table={0}&select=*").format(self.name)

    def _save_fetched_file(self, file_handle):
        df = pd.read_csv(file_handle)
        df.to_hdf(self.filename, self.name, format="t")

class KICatalog(ExoplanetArchiveCatalog):
    """
    Kepler Stellar Table (Q1 through Q17)
    """
    name = "q1_q17_dr24_stellar"

class KOICatalog(ExoplanetArchiveCatalog):
    """
    Kepler Kepler Objects of Interest (Q1 through Q17)
    """
    name = "q1_q17_dr24_koi"

    def join_stars(self, df=None):
        if df is None:
            df = self.df
        kic = KICatalog(data_root=self.data_root)
        return pd.merge(df, kic.df, on="kepid")

class CumulativeCatalog(ExoplanetArchiveCatalog):
    """
    Kepler Objects of Interest (Cumulative)
    """
    name = "cumulative"

class ExoplanetCatalog(ExoplanetArchiveCatalog):
    """
    All confirmed planets
    """
    name = "exoplanets"

class EPICatalog(ExoplanetArchiveCatalog):
    """
    K2 Targets
    """
    ext = ".csv"
    name = "k2targets"

    def _save_fetched_file(self, file_handle):
        with open(self.filename, "wb") as f:
            f.write(file_handle.read())

class K2CandidatesCatalog(ExoplanetArchiveCatalog):
    """
    K2 Candidates
    """
    name = "k2candidates"

class CatalogDownloadError(Exception):
    """
    Exception raised when an catalog download request fails.
    :param code:
        The HTTP status code that caused the failure.
    :param url:
        The endpoint (with parameters) of the request.
    :param txt:
        A human readable description of the error.
    """
    def __init__(self, code, url, txt):
        super(CatalogDownloadError, self).__init__(
            "The download returned code {0} for URL: '{1}' with message:\n{2}"
            .format(code, url, txt))
        self.code = code
        self.txt = txt
        self.url = url

class LocalCatalog(object):

    filename = None
    args = dict()

    def __init__(self):
        self._df = None

    @property
    def df(self):
        if self._df is None:
            fn = os.path.join("data", self.filename)
            self._df = pd.read_csv(resource_filename(__name__, fn),
                                   **(self.args))
        return self._df

class EBCatalog(LocalCatalog):
    filename = "Kirk2016.csv"
    args = dict(skiprows=7, header=0, comment="#", names=[
        "kepid", "period", "period_err", "bjd0", "bjd0_err", "morph",
        "GLon", "GLat", "kmag", "Teff", "SC", "",
    ])

class AsteroGiantCatalog(LocalCatalog):
    filename = "Pinsonneault2014.tsv"
    args = dict(sep="|", comment="#", skipinitialspace=True)

class singleton(object):

    def __init__(self, cls):
        self.cls = cls
        self.inst = None

    def __call__(self, *args, **kwargs):
        if self.inst is None:
            self.inst = self.cls(*args, **kwargs)
        return self.inst

# Set all the catalogs to be singletons so that the data are shared across
# instances.
KICatalog = singleton(KICatalog)

EBCatalog = singleton(EBCatalog)
KOICatalog = singleton(KOICatalog)
CumulativeCatalog = singleton(CumulativeCatalog)
ExoplanetCatalog = singleton(ExoplanetCatalog)

EPICatalog = singleton(EPICatalog)
K2CandidatesCatalog = singleton(K2CandidatesCatalog)

AsteroGiantCatalog = singleton(AsteroGiantCatalog)

TGASDistancesCatalog = singleton(TGASDistancesCatalog)
