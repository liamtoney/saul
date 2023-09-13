"""
Contains helper functions and constants for tasks related to spectral estimation.
"""

from pathlib import Path

import numpy as np

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

        Macpherson, K. A., Coffey, J. R., Witsil, A. J., Fee, D., Holtkamp, S., Dalton,
        S., McFarlin, H., & West, M. (2022). Ambient infrasound noise, station
        performance, and their relation to land cover across Alaska. *Seismological
        Research Letters*, *93*\ (4), 2239–2258. https://doi.org/10.1785/0220210365

    Usage example:

    .. code-block:: python

        from saul import get_ak_infra_noise
        p, hnm, mnm, lnm = get_ak_infra_noise()

    Returns:
        tuple: Period [s], high noise model [dB rel. 1 Pa\ :sup:`2` Hz\ :sup:`–1`],
        median noise model [dB rel. 1 Pa\ :sup:`2` Hz\ :sup:`–1`], low noise model [dB
        rel. 1 Pa\ :sup:`2` Hz\ :sup:`–1`]
    """
    f, hnm, mnm, lnm = np.loadtxt(AK_INFRA_NOISE, delimiter=',', skiprows=12).T
    # We convert frequency to period to match the ObsPy functions
    return 1 / f, hnm, mnm, lnm


def _data_kind(st):
    """Determine whether an input :class:`~saul.waveform.stream.Stream` contains infrasound or seismic data."""
    if np.all([tr.stats.channel[1:3] == 'DF' for tr in st]):
        return 'infrasound'
    elif np.all([tr.stats.channel[1] == 'H' for tr in st]):
        return 'seismic'
    else:
        raise ValueError(
            'Could not determine whether data are infrasound or seismic — or both data kinds are present.'
        )


def _format_power_label(data_kind, db_ref_val):
    """Format the axis / colorbar label for spectral power quantities."""
    if data_kind == 'infrasound':
        # Convert Pa to µPa
        return f'Power (dB rel. [{db_ref_val * 1e6:g} μPa]$^2$ Hz$^{{-1}}$)'
    else:  # data_kind == 'seismic'
        if db_ref_val == 1:
            # Special formatting case since 1^2 = 1
            return f'Power (dB rel. {db_ref_val:g} [m s$^{{-1}}$]$^2$ Hz$^{{-1}}$)'
        else:
            return f'Power (dB rel. [{db_ref_val:g} m s$^{{-1}}$]$^2$ Hz$^{{-1}}$)'
