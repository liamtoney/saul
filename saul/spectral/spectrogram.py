"""
Contains the definition of SAUL's :class:`Spectrogram` class.
"""

import copy
from functools import cache

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from multitaper import mtspec
from scipy.signal import spectrogram

from saul.spectral.helpers import (
    CYCLES_PER_WINDOW,
    _format_power_label,
    _get_db_reference_value,
)
from saul.waveform.stream import Stream
from saul.waveform.units import _validate_provided_vs_inferred_units, get_waveform_units


class Spectrogram:
    """A class for calculating and plotting spectrograms of waveforms.

    Attributes
    ==========

    Attributes:
        method (str): See :meth:`__init__`
        win_dur (int or float): See :meth:`__init__`
        time_bandwidth_product (float): See :meth:`__init__`; only defined if
            ``method='multitaper'``
        number_of_tapers (int): See :meth:`__init__`; only defined if
            ``method='multitaper'``
        tr (:class:`~obspy.core.trace.Trace`): Input waveform
        data_kind (str): Input waveform data kind; ``'infrasound'`` or ``'seismic'``
            (inferred from channel code)
        db_ref_val (int or float): dB reference value for PSD (data kind dependent)
        waveform_units (str): Units of the input waveform
        spectrogram (tuple): Spectrogram (in dB) calculated from the input waveform; of
            the form ``(f, t, sxx_db)`` where ``f`` and ``t`` are 1D arrays and
            ``sxx_db`` is a 2D array with shape ``(f.size, t.size)``

    Methods
    =======
    """

    def __init__(
        self,
        tr_or_st,
        method='scipy',
        win_dur=8,
        time_bandwidth_product=4,
        number_of_tapers=7,
        units='infer',
    ):
        """Create a :class:`Spectrogram` object.

        The spectrogram of the input waveform is estimated in this method (only a single
        waveform may be provided). Two spectral estimation approaches are supported:
        The method implemented by SciPy (:func:`scipy.signal.spectrogram`) and the
        multitaper method (:func:`mtspec.spectrogram`). Additional input arguments
        (below) relevant only for the multitaper method are marked with an **[M]**.
        These arguments are ignored when the SciPy method is selected.

        Args:
            tr_or_st (:class:`~obspy.core.trace.Trace` or :class:`~saul.waveform.stream.Stream`):
                Input waveform (response is expected to be removed; see ``units``
                argument)
            method (str): Either ``'scipy'`` or ``'multitaper'`` **[M]**
            win_dur (int or float): Segment length in seconds. This usually must be
                adjusted, within the constraints of the total signal duration, to ensure
                that the longest-period signals of interest are included
            time_bandwidth_product (float): **[M]** Time-bandwidth product
            number_of_tapers (int): **[M]** Number of tapers to use
            units (str): Units of the input waveform; either ``'infer'`` to guess from
                input response information or a string explicitly defining the units
                (see ``_VALID_UNIT_OPTIONS`` in :mod:`saul.waveform.units` for supported
                options)
        """
        # Pre-processing and checks
        assert method in [
            'scipy',
            'multitaper',
        ], 'Method must be either \'scipy\' or \'multitaper\''
        self.method = method
        self.win_dur = win_dur
        if method == 'multitaper':
            self.time_bandwidth_product = time_bandwidth_product
            self.number_of_tapers = number_of_tapers
        st = Stream(tr_or_st)  # Cast input to saul.Stream
        assert st.count() > 0, 'No waveform provided!'
        assert st.count() == 1, 'Must provide only a single waveform!'
        self.tr = st[0].copy()  # Always use *copied* saul.Stream objects

        # Handle data kind, units, and reference dB value
        data_kind, inferred_units = get_waveform_units(self.tr)
        self.data_kind = data_kind
        self.db_ref_val = _get_db_reference_value(self.data_kind)
        self.waveform_units = _validate_provided_vs_inferred_units(
            units, inferred_units, self.data_kind
        )

        # KEY: Calculate spectrogram (in dB relative to self.db_ref_val)
        if method == 'scipy':
            fs = self.tr.stats.sampling_rate
            nperseg = int(win_dur * fs)  # Samples
            nfft = np.power(2, int(np.ceil(np.log2(nperseg))) + 1)  # Pad FFT
            f, t, sxx = spectrogram(
                self.tr.data,
                fs,
                window='hann',
                nperseg=nperseg,
                noverlap=nperseg // 2,  # 50 % overlap
                nfft=nfft,
            )
        else:  # method == 'multitaper'
            t, f, _, sxx = self._spectrogram(
                tuple(self.tr.data),
                dt=self.tr.stats.delta,
                twin=win_dur,
                nw=time_bandwidth_product,
                kspec=number_of_tapers,
                olap=0.5,  # 50 % overlap
                iadapt=0,  # "Adaptive multitaper" <- change?
            )
            f = f.squeeze()
        f, sxx = f[1:], sxx[1:, :]  # Remove DC component
        # Convert to dB [dB rel. (db_ref_val <db_ref_val_unit>)^2 Hz^-1]
        sxx_db = 10 * np.log10(sxx / (self.db_ref_val**2))
        self.spectrogram = (f, t, sxx_db)

    def plot(
        self,
        db_lim='smart',
        use_period=False,
        log_y=False,
    ):
        """Plot the calculated spectrogram.

        Args:
            db_lim (tuple, str, or None): Tuple defining min and max dB cutoffs,
                ``'smart'`` for a sensible automatic choice, or ``None`` for no clipping
            use_period (bool): If ``True``, spectrogram *y*-axis will be period [s]
                instead of frequency [Hz]
            log_y (bool): If ``True``, use log scaling for spectrogram *y*-axis
        """
        assert not (use_period and not log_y), 'Cannot use period with linear y-scale!'
        fig = plt.figure(figsize=(7, 5))
        # width_ratios effectively controls the colorbar width
        gs = GridSpec(2, 2, figure=fig, height_ratios=[2, 1], width_ratios=[40, 1])
        # Set up the three required axes
        spec_ax = fig.add_subplot(gs[0, 0])
        wf_ax = fig.add_subplot(gs[1, 0], sharex=spec_ax)  # Common time axis
        cax = fig.add_subplot(gs[0, 1])
        rescale = 1e6 if self.data_kind == 'seismic' else 1  # Use μ prefix for seismic
        wf_ax.plot(
            self.tr.times('matplotlib'), self.tr.data * rescale, 'black', linewidth=0.5
        )
        match self.waveform_units:
            case 'pa':
                ylabel = 'Pressure (Pa)'
            case 'm':
                ylabel = 'Displacement (μm)'
            case 'm/s':
                ylabel = 'Velocity (μm s$^{-1}$)'
            case 'm/s**2':
                ylabel = 'Acceleration (μm s$^{-2}$)'
            case _:
                raise ValueError(f'Invalid units: {self.waveform_units}')
        wf_ax.set_ylabel(ylabel)
        wf_ax.grid(linestyle=':', zorder=-5)
        f, t, sxx_db = self.spectrogram
        t_mpl = self.tr.stats.starttime.matplotlib_date + (t / mdates.SEC_PER_DAY)
        x = t_mpl
        dx = np.diff(x)[0]
        y = f
        dy = np.diff(y)[0]
        im = spec_ax.imshow(
            sxx_db,
            cmap='magma',
            interpolation='none',
            rasterized=True,
            aspect='auto',
            origin='lower',
            extent=(  # Carefully handling the registration here
                x.min() - dx / 2,
                x.max() + dx / 2,
                y.min() - dy / 2,
                y.max() + dy / 2,
            ),
        )
        if use_period:
            grid_axis = 'x'  # Just place horizontal gridlines; we'll add vertical later
        else:
            grid_axis = 'both'
            spec_ax.set_ylabel('Frequency (Hz)')  # Go ahead and set this now
        spec_ax.grid(linestyle=':', zorder=5, axis=grid_axis)
        fmin = 1 / (self.win_dur / CYCLES_PER_WINDOW)  # [Hz] Min. resolvable freq.
        fmax = self.tr.stats.sampling_rate / 2  # [Hz] Nyquist
        spec_ax.set_ylim(fmin, fmax)
        if log_y:
            spec_ax.set_yscale('log')
        if use_period:  # Overcome imshow() limitations by defining an axis overlay
            # Set up overlay and scale it properly
            spec_ax_overlay = spec_ax.twinx()
            spec_ax_overlay.set_ylim(1 / fmin, 1 / fmax)
            spec_ax_overlay.set_yscale('log')  # log_y is guaranteed to be True
            # Remove the ticks and labels from the underlying plot
            spec_ax.tick_params(axis='y', which='both', left=False, labelleft=False)
            # Configure ticks and axis labels for the overlay
            spec_ax_overlay.yaxis.tick_left()
            spec_ax_overlay.set_ylabel('Period (s)')
            spec_ax_overlay.yaxis.set_label_position('left')
            # Finally, we add the y-axis grid (to the overlay to ensure correct ticking)
            spec_ax_overlay.grid(linestyle=':', axis='y')
        wf_ax.set_xlim(t_mpl[0], t_mpl[-1])
        # Tick locating and formatting
        locator = mdates.AutoDateLocator()
        wf_ax.xaxis.set_major_locator(locator)
        formatter = mdates.AutoDateFormatter(locator)
        formatter.scaled[30.0] = '%b. %Y'
        formatter.scaled[1] = '%-d %b. %Y'
        formatter.scaled[1 / mdates.HOURS_PER_DAY] = '%H:%M'
        formatter.scaled[1 / mdates.MINUTES_PER_DAY] = '%H:%M'
        formatter.scaled[1 / mdates.MUSECONDS_PER_DAY] = '%H:%M:%S.%f'
        wf_ax.xaxis.set_major_formatter(formatter)
        fig.autofmt_xdate()
        start_date = self.tr.stats.starttime.strftime('%-d %B %Y')
        wf_ax.set_xlabel(f'UTC time starting on {start_date}')
        # Pick smart limits rounded to nearest 10 dB
        if db_lim == 'smart':
            db_min = np.percentile(sxx_db, 20)
            db_max = sxx_db.max()
            db_lim = np.ceil(db_min / 10) * 10, np.floor(db_max / 10) * 10
        im.set_clim(db_lim)
        # Automatically determine whether to show triangle extensions on colorbar (kind
        # of adopted from xarray)
        if db_lim:
            min_extend = sxx_db.min() < db_lim[0]
            max_extend = sxx_db.max() > db_lim[1]
        else:
            min_extend = False
            max_extend = False
        if min_extend and max_extend:
            extend = 'both'
        elif min_extend:
            extend = 'min'
        elif max_extend:
            extend = 'max'
        else:
            extend = 'neither'
        extendfrac = 0.04
        fig.colorbar(
            im,
            cax,
            extend=extend,
            extendfrac=extendfrac,
            label=_format_power_label(self.db_ref_val, self.waveform_units),
        )
        spec_ax.set_title(self.tr.id, family='monospace')
        # Layout adjustment
        gs.tight_layout(fig)
        gs.update(hspace=0.1, wspace=0.07)
        # Finnicky formatting to get extension triangles (if they exist) to extend above
        # and below the vertical extent of the spectrogram axes
        pos = cax.get_position()
        triangle_height = extendfrac * pos.height
        ymin = pos.ymin
        height = pos.height
        if min_extend and max_extend:
            ymin -= triangle_height
            height += 2 * triangle_height
        elif min_extend and not max_extend:
            ymin -= triangle_height
            height += triangle_height
        elif max_extend and not min_extend:
            height += triangle_height
        cax.set_position([pos.xmin, ymin, pos.width, height])
        fig.show()

    def copy(self):
        """Return a deep copy of the :class:`Spectrogram` object."""
        return copy.deepcopy(self)

    @staticmethod
    @cache
    def _spectrogram(tr_data_tuple, **kwargs):
        """Wrapper around :func:`mtspec.spectrogram` to facilitate tuple input (needed for memoization).

        Warning:
            For large input arrays (many samples), conversion to tuple and then back to
            :class:`numpy.ndarray` can be **slow**. In this case, memoization may not be
            worth it.
        """
        return mtspec.spectrogram(np.array(tr_data_tuple), **kwargs)
