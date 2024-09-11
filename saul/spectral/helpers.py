"""
Contains helper functions and constants for tasks related to spectral estimation and
filtering.
"""

import inspect
from pathlib import Path

import numpy as np
from obspy.core.util.base import _get_function_from_entry_point
from obspy.signal.filter import lowpass_cheby_2
from scipy.signal import freqz_zpk, iirfilter, tf2zpk

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


def obspy_filter_response(filter_type, sampling_rate, freqs=1024, **options):
    """Calculate the frequency response of an ObsPy filter.

    Based on ObsPy 1.4.1.

    Args:
        filter_type (str): Type of filter to use. Note that not all of ObsPy's filter
            types are supported; see the match statement in the code
        sampling_rate (int or float): Sampling rate of target data
        freqs (int or array_like): Passed on as ``worN`` argument to
            :func:`scipy.signal.freqz_zpk` — if an array, the response will be computed
            at these frequencies
        **options: Necessary keyword arguments that will be passed on to the respective
            filter function (e.g., ``freqmin=1``, ``freqmax=5`` for
            ``filter_type='bandpass'``)

    Returns:
        tuple: Array of frequencies at which the response was computed [Hz], frequency
        response [dB]
    """
    df = sampling_rate  # Rename so that code below resembles ObsPy more closely
    match filter_type:
        case 'bandpass':
            defaults = _get_defaults_for_filter_func(filter_type)
            corners = options.get('corners', defaults['corners'])
            zerophase = options.get('zerophase', defaults['zerophase'])
            fe = 0.5 * df
            low = options['freqmin'] / fe
            high = options['freqmax'] / fe
            effective_corners = corners * 2 if zerophase else corners
            z, p, k = iirfilter(
                effective_corners,
                [low, high],
                btype='band',
                ftype='butter',
                output='zpk',
            )
        case 'highpass' | 'lowpass':
            defaults = _get_defaults_for_filter_func(filter_type)
            corners = options.get('corners', defaults['corners'])
            zerophase = options.get('zerophase', defaults['zerophase'])
            fe = 0.5 * df
            f = options['freq'] / fe
            effective_corners = corners * 2 if zerophase else corners
            z, p, k = iirfilter(
                effective_corners, f, btype=filter_type, ftype='butter', output='zpk'
            )
        case 'lowpass_cheby_2':
            defaults = _get_defaults_for_filter_func(filter_type)
            maxorder = options.get('maxorder', defaults['maxorder'])
            b, a = lowpass_cheby_2(
                None, options['freq'], df, maxorder=maxorder, ba=True
            )
            z, p, k = tf2zpk(b, a)
        case _:
            raise NotImplementedError(f'Filter type "{filter_type}" is not supported.')
    f, h = freqz_zpk(z, p, k, worN=freqs, fs=df)
    # If the user didn't supply specific frequencies, we remove the DC component
    if isinstance(freqs, (int, float)):
        f = f[1:]
        h = h[1:]
    h_db = 20 * np.log10(abs(h))
    return f, h_db


def _get_defaults_for_filter_func(filter_type):
    """Generate dictionary of default parameters for an ObsPy filter function."""
    func = _get_function_from_entry_point('filter', filter_type)
    # https://stackoverflow.com/a/12627202
    defaults = {
        k: v.default
        for k, v in inspect.signature(func).parameters.items()
        if v.default is not inspect.Parameter.empty
    }
    return defaults


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
