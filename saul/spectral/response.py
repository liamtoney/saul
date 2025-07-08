"""
This file contains code for calculating sensor response and corner frequencies, with
optional plotting.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from saul.waveform.units import _VALID_UNIT_OPTIONS

# [Hz] Minimum frequency for response computation (playing it safe here by going lower
# than the lowest expected corner of 240 s)
_MIN_FREQ = 1 / 300

# [dB] The "CORNER_DB_REF dB point", e.g. "–3 dB point" — determines where to measure
# the corner frequency
_CORNER_DB_REF = -3

# [dB] Tolerance for corner frequency search; if the derived dB value at the corner
# frequency is not within this tolerance of `_CORNER_DB_REF` an error is raised
_DB_TOL = 0.01


def _convert_timestamp(utcdatetime):
    """Convert a UTCDateTime object to a pandas Timestamp."""
    return (
        pd.NaT if utcdatetime is None else pd.Timestamp(utcdatetime.datetime, tz='UTC')
    )


def _compute_sensor_response(response, sampling_rate, min_freq):
    """Compute instrument (sensor only!) response using a nicely padded FFT."""
    nfft = 2 ** (int(np.ceil(np.log2(sampling_rate / min_freq))) + 7)  # TODO: Padding
    cpx_response, freqs = response.get_evalresp_response(
        t_samp=1 / sampling_rate,
        nfft=nfft,
        output='DEF',
        end_stage=1,  # Includes only stage sequence number 1
        hide_sensitivity_mismatch_warning=True,  # Since we're skipping some stages
    )
    return cpx_response, freqs


def _compute_db_relative_to_ref(cpx_response, freqs, ref_freq):
    """Compute response in dB relative to sensor sensitivity reference frequency."""
    abs_response = np.abs(cpx_response)
    abs_response[abs_response == 0] = np.nan  # Avoid log10(0)
    ref_value = abs_response[np.argmin(np.abs(freqs - ref_freq))]
    db_response = 20 * np.log10(abs_response / ref_value)
    return db_response


def calculate_responses(inventory, sampling_rate=10, plot=False):
    """Calculate sensor responses and corner frequencies from an ObsPy inventory.

    Args:
        inventory (:class:`~obspy.core.inventory.Inventory`): ObsPy inventory object
            containing station metadata.
        sampling_rate (int or float): Sampling rate for response computation in Hz.
        plot (bool): If True, plot the responses and corner frequencies.

    Returns:
        :class:`~pandas.DataFrame`: DataFrame with columns for network, station,
        location code, start date, end date, sensor type, and corner frequency.
    """

    # Set up lists to store key info for the DataFrame
    networks, stations, location_codes = [], [], []
    start_dates, end_dates = [], []
    sensor_types = []
    corner_frequencies = []

    # Plot, if requested
    if plot:
        fig, (ax1, ax2) = plt.subplots(nrows=2, sharex=True)

    # Iterate over the inventory
    print('Calculating responses...')
    for network in inventory:
        for station in network:

            # Handle multiple location codes for a single station, which implies multiple
            # sensors
            if len(set(channel.location_code for channel in station)) != 1:
                raise NotImplementedError(
                    'Multiple location codes for a single station!'
                )

            # First channel representative of sensor
            channel_sensor = station.channels[0]

            # Store some metadata
            networks.append(network.code)
            stations.append(station.code)
            location_codes.append(channel_sensor.location_code)
            start_dates.append(_convert_timestamp(station.start_date))
            end_dates.append(_convert_timestamp(station.end_date))

            # KEY: The sensor type, which can provide clues on the response & corners
            sensor_types.append(channel_sensor.sensor.type)

            # Check the sensor response stage
            sensor_stage = channel_sensor.response.response_stages[0]
            assert sensor_stage.input_units.lower() in _VALID_UNIT_OPTIONS
            assert sensor_stage.output_units.upper() == 'V'

            # Calculate the response
            cpx_response, freqs = _compute_sensor_response(
                channel_sensor.response, sampling_rate, _MIN_FREQ
            )
            ref_freq = sensor_stage.stage_gain_frequency  # [Hz]  # TODO: Correct?
            db_response = _compute_db_relative_to_ref(cpx_response, freqs, ref_freq)

            # Find frequency of corner
            mask = freqs <= ref_freq  # We only look below the reference frequency
            db_response_lower = db_response[mask]
            freqs_lower = freqs[mask]
            corner_db_ref_idx = np.nanargmin(np.abs(db_response_lower - _CORNER_DB_REF))
            corner_db_ref_freq = freqs_lower[corner_db_ref_idx]
            corner_db_ref_value = db_response_lower[corner_db_ref_idx]
            msg = 'Corner frequency not found within tolerance!'
            assert abs(_CORNER_DB_REF - corner_db_ref_value) < _DB_TOL, msg
            corner_frequencies.append(corner_db_ref_freq)

            # Optional plotting
            if plot:
                label = f'{network.code}.{station.code}.{channel_sensor.location_code}'
                ax1.semilogx(freqs, db_response)
                ax2.semilogx(freqs, np.angle(cpx_response, deg=True), label=label)
                ax1.scatter(corner_db_ref_freq, corner_db_ref_value)

    print('Done')

    # Make DataFrame with results
    df = pd.DataFrame(
        dict(
            network=networks,
            station=stations,
            location_code=location_codes,
            start_date=start_dates,
            end_date=end_dates,
            sensor_type=sensor_types,
            corner_frequency=corner_frequencies,
        )
    )

    # Optionally finish the plot
    if plot:
        yticks1 = [-20, -10, -6, -3, 0]  # [dB]
        ax1.set_ylim(yticks1[0], yticks1[-1])
        ax1.set_yticks(yticks1)
        ax2.set_ylim(-180, 180)
        ax2.yaxis.set_major_locator(plt.MultipleLocator(90))
        ax2.yaxis.set_minor_locator(plt.MultipleLocator(30))
        ax2.set_xlim(_MIN_FREQ, sampling_rate / 2)
        ax1.set_ylabel('Amplitude\n(dB re. val. @ ref. freq.)')
        ax2.set_ylabel('Phase (°)')
        ax2.set_xlabel('Frequency (Hz)')
        for ax in ax1, ax2:
            ax.grid(ls=':')
            ax.set_axisbelow(True)
        legend = fig.legend()
        for text in legend.get_texts():
            text.set_family('monospace')
        fig.tight_layout()
        fig.show()

    # Return
    return df
