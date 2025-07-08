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
    nfft = 2 ** (int(np.ceil(np.log2(sampling_rate / min_freq))) + 8)  # TODO: Padding
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
            containing station metadata, including response information. This means the
            inventory should have been fetched with the ``level='response'`` option!
        sampling_rate (int or float): Sampling rate for response computation in Hz. This
            must be high enough such that the Nyquist frequency is above the reference
            frequency ("stage_gain_frequency") of the sensors in the input inventory.
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

        if len(network.stations) == 0:
            continue  # No stations in this network

        for station in network:

            if len(station.channels) == 0:
                continue  # No channels for this station

            # Handle multiple location codes for a single station, which implies
            # multiple sensors
            unique_location_codes = set(channel.location_code for channel in station)
            for location_code in unique_location_codes:

                location = station.select(location=location_code)

                # TODO: First channel encountered at this location is considered
                # representative of sensor
                channel_sensor = location.channels[0]

                # Is a response present?
                if len(channel_sensor.response.response_stages) == 0:
                    continue  # No response for the representative channel

                # Use double dash for empty location codes
                location_code_str = '--' if location_code == '' else location_code

                # Store some metadata
                networks.append(network.code)
                stations.append(station.code)
                location_codes.append(location_code_str)
                start_dates.append(_convert_timestamp(channel_sensor.start_date))
                end_dates.append(_convert_timestamp(channel_sensor.end_date))

                # KEY: The sensor type, which can provide clues on response & corners
                sensor_types.append(channel_sensor.sensor.type)

                # Check the sensor response stage
                sensor_stage = channel_sensor.response.response_stages[0]
                assert sensor_stage.input_units.lower() in _VALID_UNIT_OPTIONS
                assert sensor_stage.output_units.upper() == 'V'
                ref_freq = sensor_stage.stage_gain_frequency  # [Hz]  # TODO: Correct?
                msg = 'Sampling rate too low for reference frequency!'
                assert ref_freq < sampling_rate / 2, msg

                # Calculate the response
                cpx_response, freqs = _compute_sensor_response(
                    channel_sensor.response, sampling_rate, _MIN_FREQ
                )
                db_response = _compute_db_relative_to_ref(cpx_response, freqs, ref_freq)

                # Find frequency of corner
                mask = freqs <= ref_freq  # We only look below the reference frequency
                db_response_lower = db_response[mask]
                freqs_lower = freqs[mask]
                corner_db_ref_idx = np.nanargmin(
                    np.abs(db_response_lower - _CORNER_DB_REF)
                )
                corner_db_ref_freq = freqs_lower[corner_db_ref_idx]
                corner_db_ref_value = db_response_lower[corner_db_ref_idx]
                msg = 'Corner frequency not found within tolerance!'
                assert abs(_CORNER_DB_REF - corner_db_ref_value) < _DB_TOL, msg
                corner_frequencies.append(corner_db_ref_freq)

                # Optional plotting
                if plot:
                    label = f'{network.code}.{station.code}.{location_code_str}'
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
        if df.empty:
            plt.close(fig)
        else:
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
            _ax1 = ax1.twiny()
            _ax2 = ax2.twiny()
            for _ax in _ax1, _ax2:
                _ax.set_xscale('log')
                _ax.set_xlim(1 / _MIN_FREQ, 1 / (sampling_rate / 2))
            _ax1.set_xlabel('Period (s)', labelpad=10)
            _ax2.tick_params(labeltop=False)
            fig.tight_layout()
            fig.show()

    # Warn if DataFrame is empty
    if df.empty:
        print('No responses found in the inventory!')

    # Return
    return df
