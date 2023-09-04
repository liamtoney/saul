"""
Contains the definition of the Stream class.
"""

from pathlib import Path

import obspy
from obspy import UTCDateTime
from waveform_collection import gather_waveforms, read_local


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
        )
        return cls(traces=st.traces)

    @classmethod
    def from_local(
        cls,
        data_dir,
        coord_file,
        network,
        station,
        channel,
        starttime,
        endtime,
        location='*',
    ):
        assert Path(data_dir).is_dir(), f'Directory `{data_dir}` doesn\'t exist!'
        assert Path(coord_file).is_file(), f'File `{coord_file}` doesn\'t exist!'
        st = read_local(
            data_dir=data_dir,
            coord_file=coord_file,
            network=network,
            station=station,
            location=location,
            channel=channel,
            starttime=UTCDateTime(*starttime),  # Convert from tuple
            endtime=UTCDateTime(*endtime),  # Convert from tuple
            merge=False,
        )
        return cls(traces=st.traces)
