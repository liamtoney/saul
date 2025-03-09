"""
Contains tools for estimating and plotting spectra.
"""

import warnings

with warnings.catch_warnings():
    # Ignore "SyntaxWarning: invalid escape sequence '\ '" arising from the docstring
    # formatting in `get_ak_infra_noise()` and `PSD.smooth()`
    warnings.simplefilter('ignore', category=SyntaxWarning)
    from saul.spectral.helpers import (
        get_ak_infra_noise,
        obspy_filter_response,
        extract_trace_filter_params,
    )
    from saul.spectral.psd import PSD
from saul.spectral.spectrogram import Spectrogram
