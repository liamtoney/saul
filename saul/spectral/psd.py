"""
Contains the definition of SAUL's :class:`PSD` class.
"""

import copy
from functools import cache

import matplotlib.pyplot as plt
import numpy as np
from esi_core.gmprocess.waveform_processing.smoothing.konno_ohmachi import (
    konno_ohmachi_smooth,
)
from multitaper import mtspec
from obspy.signal.spectral_estimation import (
    get_idc_infra_hi_noise,
    get_idc_infra_low_noise,
    get_nhnm,
    get_nlnm,
)
from scipy.fft import next_fast_len
from scipy.signal import welch

from saul.spectral.helpers import (
    CYCLES_PER_WINDOW,
    REFERENCE_PRESSURE,
    REFERENCE_VELOCITY,
    _format_power_label,
    get_ak_infra_noise,
)
from saul.waveform.stream import Stream
from saul.waveform.units import get_waveform_units


class PSD:
    """A class for calculating and plotting PSDs of one or more waveforms.

    Attributes:
        method (str): See :meth:`__init__`
        win_dur (int or float): See :meth:`__init__`; only defined if ``method='welch'``
        time_bandwidth_product (float): See :meth:`__init__`; only defined if
            ``method='multitaper'``
        number_of_tapers (int): See :meth:`__init__`; only defined if
            ``method='multitaper'``
        st (SAUL :class:`~saul.waveform.stream.Stream`): Input waveforms (single
            :class:`~obspy.core.trace.Trace` input is converted to SAUL
            :class:`~saul.waveform.stream.Stream`)
        data_kind (str): Input waveform data kind; ``'infrasound'`` or ``'seismic'``
            (inferred from channel code)
        db_ref_val (int or float): dB reference value for PSD (data kind dependent)
        psd (list): List of PSDs (in dB) calculated from input waveforms; of the form
            ``[(f1, pxx_db1), (f2, pxx_db2), ...]`` given a
            :class:`~saul.waveform.stream.Stream` consisting of
            :class:`~obspy.core.trace.Trace` entries ``[tr1, tr2, ...]``
    """

    def __init__(
        self,
        tr_or_st,
        method='welch',
        win_dur=60,
        time_bandwidth_product=4,
        number_of_tapers=7,
    ):
        """Create a :class:`PSD` object.

        The PSDs of the input waveforms are estimated in this method. Two spectral
        estimation approaches are supported: Welch's method (:func:`scipy.signal.welch`)
        and the multitaper method (:class:`mtspec.MTSpec`). The input arguments (below)
        relevant for each method are marked with a **[W]** for Welch's method and an
        **[M]** for the multitaper method. Arguments corresponding to the non-selected
        method are ignored.

        Args:
            tr_or_st (:class:`~obspy.core.trace.Trace` or :class:`~saul.waveform.stream.Stream`):
                Input waveforms (response is expected to be removed; SAUL expects units
                of pressure [Pa] for infrasound data and velocity [m/s] for seismic
                data!)
            method (str): Either ``'welch'`` **[W]** or ``'multitaper'`` **[M]**
            win_dur (int or float): **[W]** Segment length in seconds. This usually must
                be tweaked to obtain the cleanest-looking plot and to ensure that the
                longest-period signals of interest are included
            time_bandwidth_product (float): **[M]** Time-bandwidth product
            number_of_tapers (int): **[M]** Number of tapers to use
        """
        # Pre-processing and checks
        assert method in [
            'welch',
            'multitaper',
        ], 'Method must be either \'welch\' or \'multitaper\''
        self.method = method
        if method == 'welch':
            self.win_dur = win_dur
        else:  # self.method == 'multitaper'
            self.time_bandwidth_product = time_bandwidth_product
            self.number_of_tapers = number_of_tapers
        self.st = Stream(tr_or_st).copy()  # Always use *copied* saul.Stream objects
        assert self.st.count() > 0, 'No waveforms provided!'
        data_kinds = set(get_waveform_units(tr)[0] for tr in self.st)  # Unique kinds
        assert len(data_kinds) == 1, 'Input waveforms have mixed units — not supported!'
        self.data_kind = list(data_kinds)[0]

        # Set reference value for PSD from data kind
        self.db_ref_val = (
            REFERENCE_PRESSURE if self.data_kind == 'infrasound' else REFERENCE_VELOCITY
        )

        # KEY: Calculate PSD (in dB relative to self.db_ref_val)
        self.psd = []
        for tr in self.st:
            if method == 'welch':
                fs = tr.stats.sampling_rate
                nperseg = int(win_dur * fs)  # Samples
                nfft = np.power(2, int(np.ceil(np.log2(nperseg))) + 1)  # Pad FFT
                f, pxx = welch(tr.data, fs, nperseg=nperseg, nfft=nfft)
            else:  # method == 'multitaper'
                mtspec = self._mtspec(
                    tuple(tr.data),
                    nw=time_bandwidth_product,
                    kspec=number_of_tapers,  # After a certain point this saturates
                    dt=tr.stats.delta,
                    nfft=next_fast_len(tr.stats.npts),
                )
                f, pxx = mtspec.rspec()
                f, pxx = f.squeeze(), pxx.squeeze()
            f, pxx = f[1:], pxx[1:]  # Remove DC component
            # Convert to dB [dB rel. (db_ref_val <db_ref_val_unit>)^2 Hz^-1]
            pxx_db = 10 * np.log10(pxx / (self.db_ref_val**2))
            self.psd.append((f, pxx_db))

    def plot(
        self,
        db_lim='smart',
        use_period=False,
        log_x=True,
        show_noise_models=False,
        infra_noise_model='ak',
    ):
        """Plot the calculated PSDs.

        Args:
            db_lim (tuple, str, or None): Tuple defining min and max dB cutoffs,
                ``'smart'`` for a sensible automatic choice, or ``None`` for no clipping
            use_period (bool): If ``True``, *x*-axis will be period [s] instead of
                frequency [Hz]
            log_x (bool): If ``True``, use log scaling for *x*-axis
            show_noise_models (bool): Whether to plot reference noise models
            infra_noise_model (str): Which infrasound noise model to use (only used if
                ``show_noise_models`` is ``True`` and ``self.data_kind`` is
                ``'infrasound'``), one of ``'ak'`` (Alaska noise model) or ``'idc'``
                (IMS array noise model)
        """
        assert not (use_period and not log_x), 'Cannot use period with linear x-scale!'
        assert infra_noise_model in [
            'ak',
            'idc',
        ], 'Infrasound noise model must be either \'ak\' or \'idc\''
        fig, ax = plt.subplots()
        for tr, (f, pxx_db) in zip(self.st, self.psd):
            ax.plot(1 / f if use_period else f, pxx_db, label=tr.id)
        if log_x:
            ax.set_xscale('log')
        if show_noise_models:
            if self.data_kind == 'infrasound':
                if infra_noise_model == 'ak':
                    period, *nms = get_ak_infra_noise()
                    noise_models = [(period, nm) for nm in nms]
                else:  # infra_noise_model == 'idc':
                    noise_models = [get_idc_infra_low_noise(), get_idc_infra_hi_noise()]
                # These are all given relative to 1 Pa, so need to convert to ref_val
                for i, noise_model in enumerate(noise_models):
                    period, pxx_db_rel_1_pa = noise_model
                    pxx_db_rel_ref_val = pxx_db_rel_1_pa - 10 * np.log10(
                        self.db_ref_val**2
                    )
                    noise_models[i] = period, pxx_db_rel_ref_val
            else:  # self.data_kind == 'seismic'
                noise_models = [get_nlnm(), get_nhnm()]
                # These are in units of acceleration, so need to convert to velocity
                for i, noise_model in enumerate(noise_models):
                    period, pxx_db_acc = noise_model
                    # Below equation is taken from Table 3 in Peterson (1993)
                    # https://pubs.usgs.gov/of/1993/0322/ofr93-322.pdf
                    pxx_db_vel = pxx_db_acc + 20.0 * np.log10(period / (2 * np.pi))
                    noise_models[i] = period, pxx_db_vel
            xlim, ylim = ax.get_xlim(), ax.get_ylim()  # Store these to restore limits
            for i, noise_model in enumerate(noise_models):
                period, pxx_db = noise_model
                ax.plot(
                    period if use_period else 1 / period,
                    pxx_db,
                    color='tab:gray',
                    linestyle=':',
                    zorder=-5,
                    label='Noise model' if not i else None,  # Only label one line
                )
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
        legend = ax.legend()
        # For every ID in the legend, use monospace font (ignore noise model label!)
        for label in legend.get_texts()[: len(self.psd)]:
            label.set_family('monospace')
        if self.method == 'welch':
            fmin = 1 / (self.win_dur / CYCLES_PER_WINDOW)  # [Hz] Min. resolvable freq.
        else:  # self.method == 'multitaper'
            fmin = np.min([f for f, _ in self.psd])  # [Hz] Show the full PSD... bad?
        fmax = max([tr.stats.sampling_rate for tr in self.st]) / 2  # [Hz] Max. Nyquist
        if use_period:
            xlabel = 'Period (s)'
            ax.set_xlim(1 / fmax, 1 / fmin)  # Follow convention (increasing period)
        else:
            xlabel = 'Frequency (Hz)'
            ax.set_xlim(fmin, fmax)
        # Pick smart limits "ceiled" to nearest 10 dB
        if db_lim == 'smart':
            pxx_db_all = []
            for _, pxx_db in self.psd:
                pxx_db_all += pxx_db.tolist()
            db_min = np.percentile(pxx_db_all, 5)  # Percentile across all PSDs
            db_max = np.max(pxx_db_all)  # Max value across all PSDs
            db_lim = np.ceil(db_min / 10) * 10, np.ceil(db_max / 10) * 10
        ax.set_ylim(db_lim)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(_format_power_label(self.data_kind, self.db_ref_val))
        fig.tight_layout()
        fig.show()

    def smooth(self, bandwidth):
        """Smooth the calculated PSDs via the Konno–Ohmachi method.

        The Konno–Ohmachi method smooths PSDs using fixed-bandwith windows. The C code
        used by this method is
        `here <https://code.usgs.gov/ghsc/esi/esi-core/-/blob/main/src/esi_core/gmprocess/waveform_processing/smoothing/smoothing.c>`_.
        The
        `ObsPy documentation <https://docs.obspy.org/packages/autogen/obspy.signal.konnoohmachismoothing.konno_ohmachi_smoothing.html>`_
        for a similar function may also be helpful.

        For more information, see equation 4 in Konno and Ohmachi (1998) — the :math:`b`
        in that equation is the ``bandwidth`` parameter here.

            Konno, K., & Ohmachi, T. (1998). Ground-motion characteristics estimated
            from spectral ratio between horizontal and vertical components of
            microtremor. *Bulletin of the Seismological Society of America*, *88*\ (1),
            228–241. https://doi.org/10.1785/BSSA0880010228

        Note:
            The smoothing is performed in-place on the existing spectra in this object!

        Args:
            bandwidth (int or float): Bandwidth for smoothing — lower values produce a
                broader smoothing effect
        """
        for f, pxx_db in self.psd:
            konno_ohmachi_smooth(
                spec=pxx_db,
                freqs=f,
                ko_freqs=f,
                spec_smooth=pxx_db,
                bandwidth=bandwidth,
            )
        return self

    def copy(self):
        """Return a deep copy of the :class:`PSD` object."""
        return copy.deepcopy(self)

    @staticmethod
    @cache
    def _mtspec(tr_data_tuple, **kwargs):
        """Wrapper around :class:`mtspec.MTSpec` to facilitate tuple input (needed for memoization).

        Warning:
            For large input arrays (many samples), conversion to tuple and then back to
            :class:`numpy.ndarray` can be **slow**. In this case, memoization may not be
            worth it.
        """
        return mtspec.MTSpec(np.array(tr_data_tuple), **kwargs)
