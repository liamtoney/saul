"""
Contains helper functions and constants for tasks related to spectral estimation and
filtering.
"""

import inspect
from pathlib import Path

import matplotlib.pyplot as plt
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


def obspy_filter_response(
    filter_type, sampling_rate, freqs=4096, plot=False, **options
):
    """Calculate the frequency response of an ObsPy filter.

    Based on ObsPy 1.4.1.

    Args:
        filter_type (str): Type of filter to use. Note that not all of ObsPy's filter
            types are supported; see the match statement in the code
        sampling_rate (int or float): Sampling rate of target data
        freqs (int or array_like): Passed on as ``worN`` argument to
            :func:`scipy.signal.freqz_zpk` — if an array, the response will be computed
            at these frequencies
        plot (bool): Whether to plot the frequency response
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
    if plot:
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.semilogx(f, h_db)
        match filter_type:
            case 'bandpass':
                x1, x2 = options['freqmin'], options['freqmax']
                ax.scatter([x1, x2], [-3, -3])
            case 'highpass':
                x1, x2 = options['freq'], f.max()
                ax.scatter(x1, -3)
            case 'lowpass' | 'lowpass_cheby_2':
                x1, x2 = f.min(), options['freq']
                if filter_type != 'lowpass_cheby_2':
                    ax.scatter(x2, -3)
        ax.axvspan(x1, x2, color='tab:gray', alpha=0.5, lw=0, zorder=-5)
        ax.grid(which='both', ls=':', color='tab:gray')
        ax.set_axisbelow(True)
        left = f.min() if x1 == f.min() else x1 / 2
        right = f.max() if x2 == f.max() else x2 * 2
        ax.set_xlim(left, right)
        if filter_type == 'lowpass_cheby_2':
            ax.set_ylim(-100, 5)
            ax.set_yticks([0, -20, -40, -60, -80, -96])
        else:
            ax.set_ylim(-20, 1)
            ax.set_yticks([0, -1, -3, -6, -10, -20])
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('Gain (dB)')
        fig.tight_layout()
        fig.show()
    return f, h_db


def extract_trace_filter_params(tr):
    """Extract filter parameters from an ObsPy :class:`~obspy.core.trace.Trace` object.

    Expects to find a filter operation in the string stored in
    ``tr.stats.processing[-1]``. Can be combined with
    :func:`~saul.spectral.helpers.obspy_filter_response` to conveniently plot the filter
    response of the previous filtering operation like so:

    .. code-block:: python

        tr.filter(...)
        obspy_filter_response(plot=True, **extract_trace_filter_params(tr))

    Args:
        tr (:class:`~obspy.core.trace.Trace`): Input trace

    Returns:
        dict: Extracted filter parameters

    Warning:
        The uses sketchy string processing and :func:`eval` to extract the filter
        parameters!
    """
    string = tr.stats.processing[-1]
    assert 'filter' in string, 'Was filtering the last processing step?'
    part_options, part_filter_type = string.rstrip(')').split('::')
    options = eval(part_options.split('=')[1])
    filter_type = eval(part_filter_type.split('=')[1])
    return dict(
        filter_type=filter_type, sampling_rate=tr.stats.sampling_rate, **options
    )


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
