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
                remove_response=False,
                verbose=True,
            )
            traces = st.traces

        # Some checks and metadata assignment
        for tr in obspy.Stream(traces):
            assert all([hasattr(tr.stats, a) for a in ('latitude', 'longitude', 'elevation')]), f'{tr.id} is missing coordinates!'
            assert hasattr(tr.stats, 'response'), f'{tr.id} is missing response information!'
            tr.stats.response_removed = False
            if hasattr(tr.stats, 'processing'):
                for process_string in tr.stats.processing:
                    if 'remove_response' in process_string:
                        tr.stats.response_removed = True

        # Actually create the Stream
        super().__init__(traces=traces)

    def remove_response(self, *args, **kwargs):
        for tr in self:
            if tr.stats.response_removed:
                print('Response already removed for some or all traces! Nothing done.')
                return
        st = super().remove_response(*args, **kwargs)
        for tr in self:
            tr.stats.response_removed = True
        return st
