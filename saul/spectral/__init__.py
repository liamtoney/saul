"""
Contains tools for estimating and plotting spectra.
"""

import warnings

with warnings.catch_warnings():
    # Ignore "SyntaxWarning: invalid escape sequence '\ '" arising from the docstring
    # formatting in `get_ak_infra_noise()`
    warnings.simplefilter('ignore', category=SyntaxWarning)
    from saul.spectral.helpers import get_ak_infra_noise
from saul.spectral.psd import PSD
from saul.spectral.spectrogram import Spectrogram
