# -*- coding: utf-8 -*-

from __future__ import division, print_function

import pandas as pd

try:
    from gaia_tools import xmatch
    import gaia_tools.load as gload
except ImportError:
    xmatch = None

__all__ = ["tgas_match"]


def tgas_match(cat, tgas=None, **kwargs):
    """
    This cross-matches a pandas DataFrame to the TGAS catalog. The first
    argument should be a data frame and the other keyword arguments are passed
    to the ``gaia_tools.xmatch`` function.

    """
    if xmatch is None:
        raise ImportError("gaia_tools")

    # Load the catalogs
    if tgas is None:
        tgas = gload.tgas()

    # Set the default columns
    kwargs["colRA1"] = kwargs.get("colRA1", "ra")
    kwargs["colDec1"] = kwargs.get("colDec1", "dec")
    kwargs["epoch1"] = kwargs.get("epoch1", 2000.0)
    kwargs["colRA2"] = kwargs.get("colRA2", "ra")
    kwargs["colDec2"] = kwargs.get("colDec2", "dec")
    kwargs["epoch2"] = kwargs.get("epoch2", 2015.0)

    # Do the cross-match using gaia_tools
    m1, m2, dist = xmatch.xmatch(cat.to_records(), tgas, **kwargs)

    # Build a pandas DataFrame out of the match and save the TGAS columns
    matched = pd.DataFrame(cat.iloc[m1])
    tgas_matched = tgas[m2]
    for k in tgas_matched.dtype.names:
        # Ugliness to deal with byte ordering
        if tgas_matched[k].dtype.byteorder in ["=", "|"]:
            matched["tgas_" + k] = tgas_matched[k]
        else:
            matched["tgas_" + k] = tgas_matched[k].byteswap().newbyteorder()
    matched["tgas_match_distance"] = dist

    return matched
