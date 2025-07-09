"""
Calculation of sensor response and corner frequencies, with optional plotting.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from obspy.core.inventory import PolesZerosResponseStage

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
        inventory (:class:`~obspy.core.inventory.inventory.Inventory`): ObsPy inventory
            object containing station metadata, including response information. This
            means the inventory should have been fetched with the ``level='response'``
            option!
        sampling_rate (int or float): Sampling rate for response computation in Hz. This
            must be high enough such that the Nyquist frequency is above the reference
            frequency ("stage_gain_frequency") of the sensors in the input inventory.
        plot (bool): If True, plot the responses and corner frequencies.

    Returns:
        :class:`~pandas.DataFrame`: DataFrame with columns for network code, station
        code, location code, channel code, sensor type, and corner frequency.
    """

    # Set up lists to store key info for the DataFrame
    networks, stations, locations, channels = [], [], [], []
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
            unique_location_codes = sorted(
                set(channel.location_code for channel in station)
            )
            for location_code in unique_location_codes:

                location = station.select(location=location_code)

                # Use double dash for empty location codes
                location_code_str = '--' if location_code == '' else location_code

                # Get nice regex for channel codes
                channel_letters = np.array(
                    [list(channel.code) for channel in location.channels]
                )
                channel_code = ''
                for column in channel_letters.T:
                    if len(set(column)) == 1:  # All the same letter
                        channel_code += column[0]
                    else:
                        channel_code += '?'  # Different letters, use '?' wildcard

                # Form trace ID
                tr_id = (
                    f'{network.code}.{station.code}.{location_code_str}.{channel_code}'
                )

                # Gather responses for this location
                _responses = [channel.response for channel in location.channels]

                # Are all responses present for this location?
                empty_responses = [
                    len(_response.response_stages) == 0 for _response in _responses
                ]
                if any(empty_responses):
                    continue  # Incomplete or fully absent responses for this location

                # Check that all sensor responses are identical for this location
                _sensor_stages = [
                    _response.response_stages[0] for _response in _responses
                ]
                if not all(
                    isinstance(_sensor_stage, PolesZerosResponseStage)
                    for _sensor_stage in _sensor_stages
                ):
                    plt.close(fig) if plot else None
                    raise NotImplementedError(
                        'Only PolesZerosResponseStage objects are supported!'
                    )
                for _sensor_stage in _sensor_stages[1:]:
                    same_poles = _sensor_stage.poles == _sensor_stages[0].poles
                    same_zeros = _sensor_stage.zeros == _sensor_stages[0].zeros
                    same_gain = _sensor_stage.stage_gain == _sensor_stages[0].stage_gain
                    same_frequency = (
                        _sensor_stage.stage_gain_frequency
                        == _sensor_stages[0].stage_gain_frequency
                    )
                    if not same_poles and same_zeros and same_gain and same_frequency:
                        plt.close(fig) if plot else None
                        raise ValueError(
                            f'Multiple sensor responses found:\n'
                            f'{tr_id} | {location.start_date} – {location.end_date}'
                        )

                # Given the above check passed the first channel encountered at this
                # location is considered representative of the sensor
                channel_sensor = location.channels[0]

                # Store some metadata
                networks.append(network.code)
                stations.append(station.code)
                locations.append(location_code_str)
                channels.append(channel_code)

                # KEY: The sensor type, which can provide clues on response & corners
                sensor_type = channel_sensor.sensor.type
                sensor_types.append('' if sensor_type is None else sensor_type)

                # Check the sensor response stage
                sensor_stage = channel_sensor.response.response_stages[0]
                input_valid = sensor_stage.input_units.lower() in _VALID_UNIT_OPTIONS
                output_valid = sensor_stage.output_units.upper() == 'V'
                if not (input_valid and output_valid):
                    plt.close(fig) if plot else None
                    raise ValueError(f'Invalid sensor response stage: {tr_id}')
                ref_freq = sensor_stage.stage_gain_frequency  # [Hz]  # TODO: Correct?
                if ref_freq > sampling_rate / 2:
                    plt.close(fig) if plot else None
                    raise ValueError(
                        f'Sampling rate too low for reference frequency: {tr_id}'
                    )

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
                if abs(_CORNER_DB_REF - corner_db_ref_value) > _DB_TOL:
                    plt.close(fig) if plot else None
                    raise ValueError(f'Corner frequency not found within tolerance!')
                corner_frequencies.append(corner_db_ref_freq)

                # Optional plotting
                if plot:
                    ax1.semilogx(freqs, db_response)
                    ax2.semilogx(freqs, np.angle(cpx_response, deg=True), label=tr_id)
                    ax1.scatter(corner_db_ref_freq, corner_db_ref_value)

    print('Done')

    # Make DataFrame with results
    df = pd.DataFrame(
        dict(
            network=networks,
            station=stations,
            location=locations,
            channel=channels,
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
            legend = fig.legend(draggable=True)
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
