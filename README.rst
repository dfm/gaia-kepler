Kepler-TGAS cross matching tools
================================

These are some Python scripts that can be used for combining Kepler/K2
catalogs and the Gaia TGAS catalog.


Dependencies
------------

This code requires the standard scientific Python stack (numpy, scipy, and
pandas), `six <https://pythonhosted.org/six/>`_ for Python 2/3 compatibility,
and the ``gaia_tools`` package (see below) requires fitsio, astropy.

This tool uses `Jo Bovy's gaia_tools <https://github.com/jobovy/gaia_tools>`_
package to interface with the TGAS data. To get started with ``gaia_tools``:

1. Install the `gaia_tools <https://github.com/jobovy/gaia_tools>` package,
2. Set the ``GAIA_TOOLS_DATA`` environment variable, and
3. Run ``python -c 'import gaia_tools.load as gload;gload.tgas()'`` to
   download the TGAS catalog.


Getting Kepler data
-------------------

To download the Kepler and K2 data for the first time,

1. Set the ``GAIA_KEPLER_DATA`` environment variable (otherwise, the default
   ``~/.gaia_kepler_data`` will be used),
2. Run ``python -c 'from gaia_kepler.data import download;download()'``


Cross-matching
--------------

To match the confirmed exoplanets catalog (from the exoplanet archive) to
TGAS, try the following:

.. code-block:: python
    from gaia_kepler import data, tgas_match

    exoplanets = data.ExoplanetCatalog().df
    matched = tgas_match(exoplanets)

    print(len(matched))

This should print ``652`` as of 2016-10-14.

To make the match more conservative, change the ``maxdist`` parameter to a
smaller value (the default is ``2``) in arcseconds.


Eclipsing binary catalog
------------------------

The EB catalog doesn't include RA and Dec columns so to cross-match it takes
one extra step:

.. code-block:: python
    import pandas as pd
    from gaia_kepler import data, tgas_match

    kic = data.KICatalog().df
    ebs = pd.merge(data.EBCatalog().df, kic[["kepid", "ra", "dec"]], on="kepid")
    matched = tgas_match(ebs)

    print(len(matched))

This should print ``325``.
