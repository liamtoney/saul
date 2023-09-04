"""
Contains helper functions and constants for tasks related to spectral estimation.
"""

from functools import cache
from pathlib import Path

import numpy as np
from multitaper import MTSpec

# Reference values for PSD dB units
REFERENCE_VELOCITY = 1  # [m/s] For seismic data
REFERENCE_PRESSURE = 20e-6  # [Pa] For infrasound data

# Minimum number of cycles a wave must make within a given time window in order to be
# considered "resolvable"
CYCLES_PER_WINDOW = 4

# This file downloaded from the supplementary material of Macpherson et al. (2022)
AK_INFRA_NOISE = Path(__file__).with_name('alaska_ambient_infrasound_noise_models.txt')


def get_ak_infra_noise():
    """Returns the Alaska ambient infrasound noise models from Macpherson et al. (2022).

    Macpherson, K. A., Coffey, J. R., Witsil, A. J., Fee, D., Holtkamp, S., Dalton, S.,
        McFarlin, H., & West, M. (2022). Ambient infrasound noise, station performance,
        and their relation to land cover across Alaska. *Seismological Research
        Letters*, *93*(4), 2239–2258. https://doi.org/10.1785/0220210365

    Returns:
        tuple: Period [s], high noise model [dB rel. 1 Pa^2 Hz^-1], median noise model
        [dB rel. 1 Pa^2 Hz^-1], low noise model [dB rel. 1 Pa^2 Hz^-1]
    """
    f, hnm, mnm, lnm = np.loadtxt(AK_INFRA_NOISE, delimiter=',', skiprows=12).T
    # We convert frequency to period to match the ObsPy functions
    return 1 / f, hnm, mnm, lnm


@cache
def _mtspec(tr_data_tuple, **kwargs):
    """Wrapper around MTSpec to facilitate tuple input (needed for memoization)."""
    return MTSpec(np.array(tr_data_tuple), **kwargs)