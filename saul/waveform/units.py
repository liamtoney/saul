"""
After response removal, waveform data have physical units. These functions perform
various checks and inferences to infer the data kind (e.g., infrasound or seismic) and
data physical units (e.g., Pa or m/s) of input waveforms.
"""

from typing import Tuple

from obspy import Trace

# These are ObsPy's available response outputs (as of ObsPy version 1.4.1)
_VALID_OUTPUT_OPTIONS = 'ACC', 'VEL', 'DISP', 'DEF'

# We only know how to handle these response units (lowercase)
# TODO: ObsPy has a more complete mapping of possible variations of these we could use
_VALID_UNIT_OPTIONS = 'm/s**2', 'm/s', 'm', 'pa'


def _get_data_kind(tr: Trace) -> str:
    """Determine what kind of data a :class:`~obspy.core.trace.Trace` contains."""
    assert len(tr.stats.channel) == 3, 'Only 3-character channel codes are supported!'
    if tr.stats.channel[1:] == 'DF':
        data_kind = 'infrasound'
    elif tr.stats.channel[1] == 'H':
        data_kind = 'seismic'
    else:
        msg = f'Could not determine data kind for channel code: {tr.stats.channel}'
        raise ValueError(msg)
    return data_kind


def _get_response_output(tr: Trace) -> Tuple[bool, str | None]:
    """Evaluate a :class:`~obspy.core.trace.Trace`'s response removal history — ``None`` means unknown."""
    # Determine which entries in `tr.stats.processing` are "response" calls
    try:
        is_response_call = [
            ('remove_response(' in entry) | ('remove_sensitivity(' in entry)
            for entry in tr.stats.processing
        ]
    except AttributeError:  # No `tr.stats.processing` attribute
        is_response_call = []
    # Determine the output units of the response removal call, if present
    match sum(is_response_call):
        case 0:
            # No response removal calls present
            response_is_removed = False
            response_output = None
        case 1:
            # One response removal call present
            response_is_removed = True
            response_call = tr.stats.processing[is_response_call.index(True)]
            parts = response_call.split('::')
            is_output_kw = ['output=' in part for part in parts]
            if not sum(is_output_kw):
                # This was a `remove_sensitivity()` call, since no `output` kwarg is present
                response_output = None
            else:
                # This was a `remove_reponse()` call
                response_output = eval(parts[is_output_kw.index(True)].split('=')[1])
                msg = f'Invalid response output: {response_output}'
                assert response_output in _VALID_OUTPUT_OPTIONS, msg
        case _:
            # More than one response removal call present — almost certainly an error
            msg = 'Processing record indicates response removed more than once!'
            raise ValueError(msg)
    return response_is_removed, response_output


def _get_response_units(tr: Trace) -> str | None:
    """Get physical units from a :class:`~obspy.core.trace.Trace`'s attached response — ``None`` means unknown."""
    response = tr.stats.get('response')
    if not response:
        # No response info attached, so we don't know the units
        response_units = None
    else:
        # We can get the units
        response_units = response.instrument_sensitivity.input_units.lower()
        msg = f'Invalid response units: {response_units}'
        assert response_units in _VALID_UNIT_OPTIONS, msg
    return response_units


def get_waveform_units(tr: Trace) -> Tuple[str, str | None]:
    """Infer the data kind and physical units of an input waveform.

    Args:
        tr (:class:`~obspy.core.trace.Trace`): Input ObsPy :class:`~obspy.core.trace.Trace` object

    Returns:
        :class:`tuple`: Tuple of ``(data_kind, waveform_units)`` where ``data_kind``
        (:class:`str`) is the waveform data kind and ``waveform_units`` (:class:`str` or
        :class:`None`) is the inferred physical units of the waveform — ``None`` means
        unknown

    Warning:
        This function is designed to carefully check in multiple places to infer the
        units of the waveform, but it is not guaranteed to be correct since there are
        too many weird edge cases! Don't rely on this function for mission-critical
        applications!
    """
    # Get helper function outputs
    data_kind = _get_data_kind(tr)
    response_is_removed, response_output = _get_response_output(tr)
    response_units = _get_response_units(tr)

    # Do some basic data kind compatability checks
    match data_kind:
        case 'infrasound':
            msg = f'Invalid response units for infrasound: {response_units}'
            assert response_units in ('pa', None), msg
            # Only 'VEL' and 'DEF' make sense for `remove_response()` call here
            msg = f'Invalid response output for infrasound: {response_output}'
            assert response_output in ('VEL', 'DEF', None), msg
        case 'seismic':
            msg = f'Invalid response units for seismic: {response_units}'
            assert response_units != 'pa', msg
        case _:
            raise ValueError(f'Unknown data kind: {data_kind}')

    # Determine the units of the waveform data
    if not response_is_removed:
        # Unknown units, probably counts but we can't be certain
        waveform_units = None
    else:
        # The waveform data have physical units... but what are they?
        if not response_output:
            # This was a `remove_sensitivity()` call, get units from `response_units`
            # (this can be `None`!)
            waveform_units = response_units
        else:
            # This was a `remove_reponse()` call, check compatibility between
            # `response_output` and `response_units`
            match data_kind:
                case 'infrasound':
                    waveform_units = 'pa'
                case 'seismic':
                    match response_output:
                        case 'ACC':
                            waveform_units = 'm/s**2'
                        case 'VEL':
                            waveform_units = 'm/s'
                        case 'DISP':
                            waveform_units = 'm'
                        case 'DEF':
                            waveform_units = response_units  # This can be `None`!
                        case _:
                            msg = f'Invalid response output: {response_output}'
                            raise ValueError(msg)
                case _:
                    raise ValueError(f'Unknown data kind: {data_kind}')

    return data_kind, waveform_units


def _validate_provided_vs_inferred_units(
    provided_units: str, inferred_units: str | None, data_kind: str
) -> str:
    """Validate user-provided units against inferred units."""
    if provided_units == 'infer':
        if not inferred_units:
            raise ValueError('Could not infer units; please provide them explicitly!')
        else:
            validated_units = inferred_units
    else:  # Use explicitly provided units
        provided_units = provided_units.lower()
        assert provided_units in _VALID_UNIT_OPTIONS, f'Invalid units: {provided_units}'
        if inferred_units is not None:  # We can additionally for consistency
            msg = f'Provided units ({provided_units}) do not match inferred units ({inferred_units})'
            assert provided_units == inferred_units, msg
        validated_units = provided_units
    # Now, check that units are compatible with the data kind
    match data_kind:
        case 'infrasound':
            msg = f'Invalid waveform units for infrasound: {validated_units}'
            assert validated_units == 'pa', msg
        case 'seismic':
            msg = f'Invalid waveform units for seismic: {validated_units}'
            assert validated_units in ('m', 'm/s', 'm/s**2'), msg
        case _:
            raise ValueError(f'Unknown data kind: {data_kind}')
    return validated_units
