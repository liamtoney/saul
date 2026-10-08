"""
Contains tools for estimating and plotting spectra.
"""

import warnings

with warnings.catch_warnings():
    # Ignore "SyntaxWarning: invalid escape sequence '\ '" arising from the docstring
    # formatting in `get_ak_infra_noise()` and `PSD.smooth()`
    warnings.simplefilter('ignore', category=SyntaxWarning)
    from saul.spectral.helpers import (
        extract_trace_filter_params,
        get_ak_infra_noise,
        obspy_filter_response,
    )
    from saul.spectral.psd import PSD

from saul.spectral.response import calculate_responses
from saul.spectral.spectrogram import Spectrogram

__all__ = [
    'PSD',
    'Spectrogram',
    'calculate_responses',
    'extract_trace_filter_params',
    'get_ak_infra_noise',
    'obspy_filter_response',
]
