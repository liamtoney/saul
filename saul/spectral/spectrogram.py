"""
Contains the definition of the Spectrogram class.
"""

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from scipy.signal import spectrogram

from saul.spectral.helpers import (
    CYCLES_PER_WINDOW,
    REFERENCE_PRESSURE,
    REFERENCE_VELOCITY,
    _data_kind,
)
from saul.waveform.stream import Stream


class Spectrogram:
    """A class for calculating and plotting spectrograms of waveforms.

    Attributes:
        win_dur (int or float): See __init__()
        tr (Trace): Input waveform
        data_kind (str): Input waveform data kind; 'infrasound' or 'seismic' (inferred
            from channel code)
        db_ref_val (int or float): dB reference value for PSD (data kind dependent)
        spectrogram (tuple): Spectrogram (in dB) calculated from the input waveform; of
            the form (f, t, sxx_db) where f and t are 1D arrays and sxx_db is a 2D array
            with shape (f.size, t.size)
    """

    def __init__(self, tr_or_st, win_dur=8):
        """Create a Spectrogram object.

        The spectrogram of the input waveform is estimated in this method (only a single
        waveform may be provided).

        Documentation for scipy.signal.spectrogram:
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.spectrogram.html

        Args:
            tr_or_st (Trace or Stream): Input waveforms (response is expected to be
                removed; SAUL expects units of pressure [Pa] for infrasound data and
                velocity [m/s] for seismic data!)
            win_dur (int or float): Segment length in seconds. This usually must be
                adjusted, within the constraints of the total signal duration, to ensure
                that the longest-period signals of interest are included
        """
        # Pre-processing and checks
        st = Stream(tr_or_st)  # Cast input to saul.Stream
        assert st.count() == 1, 'Must provide only a single Trace!'
        self.data_kind = _data_kind(st)
        self.tr = st[0].copy()  # Always use *copied* saul.Stream objects
        self.win_dur = win_dur

        # Set reference value for spectrogram from data kind
        if self.data_kind == 'infrasound':
            self.db_ref_val = REFERENCE_PRESSURE
        else:  # self.data_kind == 'seismic'
            self.db_ref_val = REFERENCE_VELOCITY

        # KEY: Calculate spectrogram (in dB relative to self.db_ref_val)
        fs = self.tr.stats.sampling_rate
        nperseg = int(win_dur * fs)  # Samples
        nfft = np.power(2, int(np.ceil(np.log2(nperseg))) + 1)  # Pad fft with zeroes
        f, t, sxx = spectrogram(
            self.tr.data,
            fs,
            window='hann',
            nperseg=nperseg,
            noverlap=nperseg // 2,
            nfft=nfft,
        )
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
            db_lim (tuple, str, or None): Tuple defining min and max dB cutoffs, 'smart'
                for a sensible automatic choice, or None for no clipping
            use_period (bool): If True, spectrogram y-axis will be period [s] instead of
                frequency [Hz]
            log_y (bool): If True, use log scaling for spectrogram y-axis
        """
        assert not (use_period and not log_y), 'Cannot use period with linear y-scale!'
        fig = plt.figure(figsize=(7, 5))
        # width_ratios effectively controls the colorbar width
        gs = GridSpec(2, 2, figure=fig, height_ratios=[2, 1], width_ratios=[40, 1])
        # Set up the three required axes
        spec_ax = fig.add_subplot(gs[0, 0])
        wf_ax = fig.add_subplot(gs[1, 0], sharex=spec_ax)  # Common time axis
        cax = fig.add_subplot(gs[0, 1])
        data = self.tr.data
        if self.data_kind == 'seismic':
            data *= 1e6  # Convert to μm/s
        wf_ax.plot(self.tr.times('matplotlib'), data, 'black', linewidth=0.5)
        wf_ax.set_ylabel(
            'Velocity (μm s$^{-1}$)' if self.data_kind == 'seismic' else 'Pressure (Pa)'
        )
        wf_ax.grid(linestyle=':', zorder=-5)
        f, t, sxx_db = self.spectrogram
        t_mpl = self.tr.stats.starttime.matplotlib_date + (t / mdates.SEC_PER_DAY)
        im = spec_ax.pcolormesh(
            t_mpl,
            1 / f if use_period else f,
            sxx_db,
            cmap='inferno',
            rasterized=True,
            shading='nearest',
        )
        spec_ax.set_ylabel('Period (s)' if use_period else 'Frequency (Hz)')
        spec_ax.grid(linestyle=':', zorder=5)

        fmin = 1 / (self.win_dur / CYCLES_PER_WINDOW)  # [Hz] Min. resolvable freq.
        fmax = self.tr.stats.sampling_rate / 2  # [Hz] Nyquist
        if use_period:
            ylim = (1 / fmin, 1 / fmax)
        else:
            ylim = (fmin, fmax)
        spec_ax.set_ylim(ylim)
        if log_y:
            spec_ax.set_yscale('log')
        wf_ax.set_xlim(t_mpl[0], t_mpl[-1])
        # Tick locating and formatting
        locator = mdates.AutoDateLocator()
        wf_ax.xaxis.set_major_locator(locator)
        wf_ax.xaxis.set_major_formatter(mdates.AutoDateFormatter(locator))
        fig.autofmt_xdate()
        start_date = self.tr.stats.starttime.strftime('%Y-%m-%d')
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
        # Get proper label for colorbar
        if self.data_kind == 'infrasound':
            # Convert Pa to µPa
            clab = f'Power (dB rel. [{self.db_ref_val * 1e6:g} μPa]$^2$ Hz$^{{-1}}$)'
        else:  # self.data_kind == 'seismic'
            if self.db_ref_val == 1:
                # Special formatting case since 1^2 = 1
                clab = f'Power (dB rel. {self.db_ref_val:g} [m s$^{{-1}}$]$^2$ Hz$^{{-1}}$)'
            else:
                clab = f'Power (dB rel. [{self.db_ref_val:g} m s$^{{-1}}$]$^2$ Hz$^{{-1}}$)'
        extendfrac = 0.04
        fig.colorbar(im, cax, extend=extend, extendfrac=extendfrac, label=clab)
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
