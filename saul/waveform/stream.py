"""
Contains the definition of the Stream class.
"""

from datetime import timedelta
from pathlib import Path

import matplotlib.dates as mdates
import numpy as np
import obspy
from obspy import UTCDateTime
from waveform_collection import gather_waveforms, read_local


class Stream(obspy.Stream):
    """A subclass of the obspy.Stream object with extra functionality.

    See the docstring for that class for documentation on the attributes and methods
    inherited by this class.
    """

    @staticmethod
    def _duration_string(tr):
        """Calculate and return a nicely-formatted string duration of a Trace."""
        duration = tr.stats.endtime - tr.stats.starttime  # [s]
        assert (not np.isnan(duration)) and (duration >= 0), 'Invalid duration!'
        # Take the ceil if duration is < 1 s; otherwise round to nearest second
        td = timedelta(seconds=max(round(duration), 1))
        days = td.days
        hours, remainder = divmod(td.seconds, int(mdates.SEC_PER_HOUR))
        minutes, seconds = divmod(remainder, int(mdates.SEC_PER_MIN))
        out = f'{days} days' if days > 1 else f'{days} day' if days == 1 else ''
        for increment, unit in zip((hours, minutes, seconds), ('hr', 'min', 's')):
            if increment > 0:
                out += f', {increment} {unit}'
        if out == '':
            out = '0 s'
        return out.lstrip(', ')

    def __mul__(self, *args, **kwargs):
        """Modify this method to return a saul.Stream instead of an obspy.Stream.

        TODO:
            Is this something in ObsPy that should be changed? Why don't they call
            st = self.__class__() in their __mul__() method?
        """
        return self.__class__(super().__mul__(*args, **kwargs))

    def __str__(self, *args, **kwargs):
        """Overwrite this method to always show all Traces, and to show durations.

        The `extended` argument is ignored. Code below copied (and then modified) from
        obspy.Stream.__str__(), from ObsPy v1.4.0.
        """
        if self.traces:
            id_length = self and max(len(tr.id) for tr in self) or 0
        else:
            id_length = 0
        out = str(len(self.traces)) + ' Trace(s) in saul.Stream:\n'
        out = out + '\n'.join(
            [
                _i.__str__(id_length) + ' | ' + self._duration_string(_i)
                for _i in self.traces
            ]
        )
        return out

    def plot(self, *args, **kwargs):
        """Slightly modify this method to ALWAYS plot into a new figure.

        Note:
            Can obtain standard obspy.Stream.plot() behavior by setting fig=None.
        """
        if 'fig' not in kwargs:
            from matplotlib.pyplot import figure

            kwargs['fig'] = figure()
        return super().plot(*args, **kwargs)

    @classmethod
    def from_iris(cls, network, station, channel, starttime, endtime, location='*'):
        """Create a saul.Stream object containing waveforms obtained from IRIS servers.

        This class method wraps waveform_collection.server.gather_waveforms() with the
        source argument set to 'IRIS'; for documentation of that function see:
        https://uaf-waveform-collection.readthedocs.io/en/master/api/waveform_collection.server.html

        Wildcards (*, ?) are accepted for the network, station, channel, and location
        parameters.

        Args:
            network (str): SEED network code
            station (str): SEED station code
            channel (str): SEED channel code
            starttime (tuple): Start time for data request; format is a tuple of
                integers: (year, month, day[, hour[, minute[, second[, microsecond]]])
            endtime (tuple): End time for data request (same format as starttime)
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
        """Create a saul.Stream object containing waveforms obtained from local files.

        This class method wraps waveform_collection.local.local.read_local(); for
        documentation of that function see:
        https://uaf-waveform-collection.readthedocs.io/en/master/api/waveform_collection.local.local.html

        Wildcards (*, ?) are accepted for the network, station, channel, and location
        parameters.

        Args:
            data_dir (str): Directory containing miniSEED files
            coord_file (str): JSON file containing coordinates for local stations (full path
                required)
            network (str): SEED network code
            station (str): SEED station code
            channel (str): SEED channel code
            starttime (tuple): Start time for data request; format is a tuple of
                integers: (year, month, day[, hour[, minute[, second[, microsecond]]])
            endtime (tuple): End time for data request (same format as starttime)
            location (str): SEED location code

        Returns:
            saul.Stream: Newly-created object with the locally obtained waveforms
        """
        assert Path(data_dir).is_dir(), f'Directory {data_dir} doesn\'t exist!'
        assert Path(coord_file).is_file(), f'File {coord_file} doesn\'t exist!'
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
