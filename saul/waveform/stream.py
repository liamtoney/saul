"""
Contains the definition of the Stream class.
"""

from pathlib import Path

import obspy
from obspy import UTCDateTime
from waveform_collection import gather_waveforms, read_local


class Stream(obspy.Stream):
    """A subclass of the `obspy.Stream` object with extra functionality.

    See the docstring for that class for documentation on the attributes and methods
    inherited by this class.
    """

    @classmethod
    def from_server(cls, network, station, channel, starttime, endtime, location='*'):
        """Create a `saul.Stream` object containing waveforms obtained from a server.

        This class method wraps `waveform_collection.server.gather_waveforms()` with the
        `source` argument set to 'IRIS'; for documentation of that function see:
        https://uaf-waveform-collection.readthedocs.io/en/master/api/waveform_collection.server.html

        Wildcards (*, ?) are accepted for the `network`, `station`, `channel`, and
        `location` parameters.

        Args:
            network (str): SEED network code
            station (str): SEED station code
            channel (str): SEED channel code
            starttime (tuple): Start time for data request; format is a tuple of
                integers: (year, month, day[, hour[, minute[, second[, microsecond]]])
            endtime (tuple): End time for data request (same format as `starttime`)
            location (str): SEED location code

        Returns:
            saul.Stream: Newly-created object with the server-obtained waveforms
        """
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
        """Create a `saul.Stream` object containing waveforms obtained from local files.

        This class method wraps `waveform_collection.local.local.read_local()`; for
        documentation of that function see:
        https://uaf-waveform-collection.readthedocs.io/en/master/api/waveform_collection.local.local.html

        Wildcards (*, ?) are accepted for the `network`, `station`, `channel`, and
        `location` parameters.

        Args:
            data_dir (str): Directory containing miniSEED files
            coord_file (str): JSON file containing coordinates for local stations (full path
                required)
            network (str): SEED network code
            station (str): SEED station code
            channel (str): SEED channel code
            starttime (tuple): Start time for data request; format is a tuple of
                integers: (year, month, day[, hour[, minute[, second[, microsecond]]])
            endtime (tuple): End time for data request (same format as `starttime`)
            location (str): SEED location code

        Returns:
            saul.Stream: Newly-created object with the locally obtained waveforms
        """
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
