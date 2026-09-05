import logging
import sys
from importlib.metadata import version

from saul.spectral import (
    PSD,
    Spectrogram,
    calculate_responses,
    extract_trace_filter_params,
    get_ak_infra_noise,
    obspy_filter_response,
)
from saul.waveform import Stream, get_availability, get_waveform_units

__version__ = version('saul')


# Define colored formatter for logging
class _ColorFormatter(logging.Formatter):
    _COLORS = {
        logging.ERROR: '\033[31m',  # Red
        logging.WARNING: '\033[33m',  # Yellow
    }
    _RESET = '\033[0m'

    def format(self, record):
        message = super().format(record)
        color = self._COLORS.get(record.levelno)
        if color and sys.stderr.isatty():
            return f'{color}{message}{self._RESET}'
        return message


logger = logging.getLogger('saul')
logger.setLevel(logging.INFO)
_handler = logging.StreamHandler()
_handler.setFormatter(_ColorFormatter('%(message)s'))
logger.addHandler(_handler)
