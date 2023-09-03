"""
Contains the definition of the PSD class.
"""

import matplotlib.pyplot as plt
import numpy as np
from obspy import Stream
from obspy.signal.spectral_estimation import (
    get_idc_infra_hi_noise,
    get_idc_infra_low_noise,
    get_nhnm,
    get_nlnm,
)
from scipy.fft import next_fast_len

from saul.spectral.helpers import (
    REFERENCE_PRESSURE,
    REFERENCE_VELOCITY,
    _mtspec,
    get_ak_infra_noise,
)


class PSD:
    """A class for calculating and plotting PSDs of one or more waveforms."""

    def __init__(self, tr_or_st, time_bandwidth_product=4, number_of_tapers=7):
        """Create a PSD object.

        The PSDs of the input waveforms are estimated in this method. See the
        documentation for `multitaper.mtspec.MTSpec` for details on some of the input
        arguments here.

        Args:
            tr_or_st (Trace or Stream): Input waveforms (response is expected to be
                removed; SAUL expects units of pressure [Pa] for infrasound data and
                velocity [m/s] for seismic data!)
            time_bandwidth_product (float): Time-bandwidth product
            number_of_tapers (int): Number of tapers to use
        """
        # Pre-processing and checks
        self.st = Stream(tr_or_st).copy()  # Always use *copied* Stream objects
        if np.all([tr.stats.channel[1:3] == 'DF' for tr in self.st]):
            self.data_kind = 'infrasound'
        elif np.all([tr.stats.channel[1] == 'H' for tr in self.st]):
            self.data_kind = 'seismic'
        else:
            raise ValueError(
                'Could not determine whether data are infrasound or seismic — or both data kinds are present.'
            )

        # Set reference value for PSD from data kind
        if self.data_kind == 'infrasound':
            self.db_ref_val = REFERENCE_PRESSURE
        else:  # self.data_kind == 'seismic'
            self.db_ref_val = REFERENCE_VELOCITY

        # KEY: Calculate PSD (in dB relative to `self.db_ref_val`)
        self.psd = []
        for tr in self.st:
            mtspec = _mtspec(
                tuple(tr.data),
                nw=time_bandwidth_product,
                kspec=number_of_tapers,  # After a certain point this saturates
                dt=tr.stats.delta,
                nfft=next_fast_len(tr.stats.npts),
            )
            f, pxx = mtspec.rspec()
            f, pxx = f.squeeze(), pxx.squeeze()
            f, pxx = f[1:], pxx[1:]  # Remove DC component
            pxx_db = 10 * np.log10(pxx / (self.db_ref_val**2))  # Convert to dB
            self.psd.append((f, pxx_db))

        # Calculate peak frequency of PSD
        self.peak_frequency = []
        for psd in self.psd:
            f, pxx_db = psd
            self.peak_frequency.append(f[np.argmax(pxx_db)])

    def __str__(self):
        """A custom string representation of the PSD object."""
        text = f'{len(self.psd)} PSD(s):'
        for tr, peak_f in zip(self.st, self.peak_frequency):
            text += f'\n{tr.id} | {peak_f:.3f} Hz peak frequency'
        return text

    def _repr_pretty_(self, p, cycle):
        """Pretty-printing for IPython usage."""
        p.text(self.__str__())

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
            db_lim (tuple, str, or None): Tuple defining min and max dB cutoffs, 'smart'
                for a sensible automatic choice, or None for no clipping
            use_period (bool): If True, x-axis will be period [s] instead of frequency
                [Hz]
            log_x (bool): If True, use log scaling for x-axis
            show_noise_models (bool): Whether to plot reference noise models
            infra_noise_model (str): Which infrasound noise model to use (only used if
                `show_noise_models` is True and `self.data_kind` is 'infrasound'), one
                of 'ak' (Alaska noise model) or 'idc' (IMS array noise model)
        """
        assert not (use_period and not log_x), 'Cannot use period with linear x-scale!'
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
                elif infra_noise_model == 'idc':
                    noise_models = [get_idc_infra_low_noise(), get_idc_infra_hi_noise()]
                else:
                    raise ValueError(
                        'Infrasound noise model must be either \'ak\' or \'idc\''
                    )
                # These are all given relative to 1 Pa, so need to convert to `ref_val`
                for i, noise_model in enumerate(noise_models):
                    period, pxx_db_rel_1_pa = noise_model
                    pxx_db_rel_ref_val = pxx_db_rel_1_pa - 10 * np.log10(
                        self.db_ref_val**2
                    )
                    noise_models[i] = period, pxx_db_rel_ref_val
            else:  # self.data_kind == 'seismic'
                noise_models = [get_nlnm(), get_nhnm()]
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
        fmax = max([tr.stats.sampling_rate for tr in self.st]) / 2  # [Hz] Max. Nyquist
        if use_period:
            xlabel = 'Period (s)'
            ax.set_xlim(left=1 / fmax)  # Follow convention (increasing period)
        else:
            xlabel = 'Frequency (Hz)'
            ax.set_xlim(right=fmax)
        # Pick smart limits "ceiled" to nearest 10 dB
        if db_lim == 'smart':
            pxx_db_all = [pxx_db for _, pxx_db in self.psd]
            db_min = np.percentile(pxx_db_all, 5)  # Percentile across all PSDs
            db_max = np.max(pxx_db_all)  # Max value across all PSDs
            db_lim = np.ceil(db_min / 10) * 10, np.ceil(db_max / 10) * 10
        ax.set_ylim(db_lim)
        ax.set_xlabel(xlabel)
        if self.data_kind == 'infrasound':
            # Convert Pa to µPa
            ylabel = f'Power (dB rel. [{self.db_ref_val * 1e6:g} μPa]$^2$ Hz$^{{-1}}$)'
        else:  # self.data_kind == 'seismic'
            if self.db_ref_val == 1:
                # Special formatting case since 1^2 = 1
                ylabel = f'Power (dB rel. {self.db_ref_val:g} [m s$^{{-1}}$]$^2$ Hz$^{{-1}}$)'
            else:
                ylabel = f'Power (dB rel. [{self.db_ref_val:g} m s$^{{-1}}$]$^2$ Hz$^{{-1}}$)'
        ax.set_ylabel(ylabel)
        fig.tight_layout()
        fig.show()
