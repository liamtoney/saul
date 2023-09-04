"""
Contains the definition of the Stream class.
"""

import obspy
from obspy import UTCDateTime
from waveform_collection import gather_waveforms


class Stream(obspy.Stream):
    @classmethod
    def from_server(cls, network, station, channel, starttime, endtime, location='*'):
        st = gather_waveforms(
            source='IRIS',
            network=network,
            station=station,
            location=location,
            channel=channel,
            starttime=UTCDateTime(*starttime),  # Convert from tuple
            endtime=UTCDateTime(*endtime),  # Convert from tuple
            merge_fill_value=False,
            trim_fill_value=False,
            remove_response=False,  # KEY
            verbose=True,
        )
        return cls(traces=st.traces)
