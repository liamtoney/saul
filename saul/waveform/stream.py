"""
Contains the definition of the Stream class.
"""

import obspy
from obspy import UTCDateTime
from waveform_collection import gather_waveforms


class Stream(obspy.Stream):
    def __init__(self, traces=None, **kwargs):
        if kwargs:
            assert traces is None, 'Cannot provide `traces` if requesting waveforms!'
            if 'loc' not in kwargs:
                kwargs['loc'] = '*'  # Basically, we're making `loc` an optional arg.
            st = gather_waveforms(
                source='IRIS',
                network=kwargs['net'],
                station=kwargs['sta'],
                location=kwargs['loc'],
                channel=kwargs['cha'],
                starttime=UTCDateTime(*kwargs['start']),  # Convert from tuple
                endtime=UTCDateTime(*kwargs['end']),  # Convert from tuple
                merge_fill_value=False,
                trim_fill_value=False,
                verbose=True,
            )
            traces = st.traces
        super().__init__(traces=traces)
