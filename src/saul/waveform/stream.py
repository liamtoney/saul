"""
Contains the definition of SAUL's :class:`Stream` class.
"""

import logging
import subprocess
import sys
from datetime import timedelta
from functools import cache
from pathlib import Path

import matplotlib.dates as mdates
import numpy as np
import obspy
from lxml.etree import Element, SubElement, tostring
from matplotlib import colormaps
from matplotlib.ticker import FuncFormatter
from matplotlib.transforms import blended_transform_factory
from obspy.geodetics.base import gps2dist_azimuth
from obspy.io.kml.core import _rgba_tuple_to_kml_color_code

from saul.waveform.earthscope import _gather_waveforms
from saul.waveform.helpers import _num2date, _preprocess_time

logger = logging.getLogger(__name__)


class Stream(obspy.Stream):
    """A subclass of ObsPy's :class:`~obspy.core.stream.Stream` class with extra functionality.

    See the docstring for that class for documentation on the attributes and methods
    inherited by this class.
    """

    @classmethod
    def from_earthscope(
        cls, network, station, channel, starttime, endtime, location='*', cache=False
    ):
        """Create a SAUL :class:`Stream` object containing waveforms obtained via EarthScope.

        Wildcards (``*``, ``?``) are accepted for the ``network``, ``station``,
        ``channel``, and ``location`` parameters. The user can optionally choose to
        cache waveform data to avoid redundant data download for repeated identical
        requests (see ``cache`` argument).

        Warning:
            Caching (see ``cache`` argument), while convenient, is sketchy since data
            are not guaranteed to be the same between calls!

        Args:
            network (str): SEED network code
            station (str): SEED station code
            channel (str): SEED channel code
            starttime (tuple or :class:`~obspy.core.utcdatetime.UTCDateTime`): Start
                time for data request; for tuple input the format is integers: ``(year,
                month, day[, hour[, minute[, second[, microsecond]]])``
            endtime (tuple or :class:`~obspy.core.utcdatetime.UTCDateTime`): End time
                for data request (same format as ``starttime``)
            location (str): SEED location code
            cache (bool): Toggle whether to cache the
                :func:`saul.waveform.earthscope._gather_waveforms` function call to
                avoid downloading data again in subsequent calls

        Returns:
            SAUL :class:`Stream`: Newly-created object with the server-obtained
            waveforms in units of integer counts
        """
        # Ensure we have UTCDateTime objects to start
        starttime = _preprocess_time(starttime)
        endtime = _preprocess_time(endtime)
        if cache:  # We must convert times to tuples
            starttime = cls._time_tuple(starttime)
            endtime = cls._time_tuple(endtime)
            gather_func = cls._gather_waveforms_cache
            logger.warning('[CACHING ENABLED]')
        else:
            gather_func = _gather_waveforms
        st = gather_func(
            network=network,
            station=station,
            location=location,
            channel=channel,
            starttime=starttime,
            endtime=endtime,
        )
        st = st.copy() if cache else st  # Ensure we don't modify the cached object
        return cls(traces=st.traces)

    def to_kml(self, filename='saul.Stream.kml', ge=False):
        """Write the SAUL :class:`Stream` station locations to a KML file and optionally open it.

        Adopted from the ObsPy code `here
        <https://github.com/obspy/obspy/blob/master/obspy/io/kml/core.py#L21-L137>`_.

        Args:
            filename (str): Output KML file name (including path)
            ge (bool): If ``True``, immediately open the output KML file in Google Earth
                Pro (only supported on macOS systems with Google Earth Pro installed)
        """
        st_sort = self.copy()  # Work on a copy of the Stream, since we modify it!
        st_sort.merge().sort(keys=['network', 'station', 'location', 'channel'])
        networks = list(set([tr.stats.network for tr in st_sort]))[::-1]  # Reverse?

        # Construct KML file
        kml = Element('kml')
        kml.set('xmlns', 'http://www.opengis.net/kml/2.2')

        document = SubElement(kml, 'Document')
        title = st_sort.__str__().split('\n')[0][:-1] + ' (merged)'
        SubElement(document, 'name').text = title  # 1st line of __str__(), note merge
        SubElement(document, 'open').text = '1'

        # Style definition
        cmap = colormaps['Pastel1']
        for i in range(len(networks)):
            color = _rgba_tuple_to_kml_color_code(cmap(i))
            style = SubElement(document, 'Style')
            style.set('id', f'station_{i}')

            iconstyle = SubElement(style, 'IconStyle')
            SubElement(iconstyle, 'color').text = color
            SubElement(iconstyle, 'scale').text = str(1.3)
            icon = SubElement(iconstyle, 'Icon')
            SubElement(icon, 'href').text = (
                'https://maps.google.com/mapfiles/kml/shapes/triangle.png'
            )

            labelstyle = SubElement(style, 'LabelStyle')
            SubElement(labelstyle, 'color').text = color

        for i, network in enumerate(networks):
            folder = SubElement(document, 'Folder')
            SubElement(folder, 'name').text = network
            SubElement(folder, 'open').text = '1'

            # Add marker for each Trace in Stream with this network code
            for tr in st_sort.select(network=network):
                placemark = SubElement(folder, 'Placemark')
                SubElement(placemark, 'name').text = tr.id
                SubElement(placemark, 'styleUrl').text = f'#station_{i}'
                SubElement(placemark, 'color').text = color
                if hasattr(tr.stats, 'longitude') and hasattr(tr.stats, 'latitude'):
                    point = SubElement(placemark, 'Point')
                    SubElement(point, 'coordinates').text = (
                        f'{tr.stats.longitude:.6f},{tr.stats.latitude:.6f},0'
                    )
                else:
                    SubElement(placemark, 'description').text = 'No coordinates!'
                    logger.warning(f'No coordinates for {tr.id}')

        # Generate KML string and write to file
        kml_string = tostring(
            kml, pretty_print=True, xml_declaration=True, encoding='UTF-8'
        )
        filename = Path(filename).expanduser().resolve()
        with filename.open('wb') as f:
            f.write(kml_string)
        assert filename.is_file(), 'Issue saving KML file!'
        logger.info(f'KML file saved to `{filename}`')

        # Optionally open file
        if ge:
            if sys.platform == 'darwin':  # If we're on macOS
                subprocess.run(
                    ['open', '-a', '/Applications/Google Earth Pro.app', filename]
                )
            else:
                raise NotImplementedError('`open_file` currently only works on macOS!')

    def plot(self, *args, **kwargs):
        """Modify this method to **always** plot into a new figure, and make record section plotting easier.

        If the user specifies ``src_lat`` and ``src_lon``, ``type='section'`` is assumed
        and distances are automatically calculated. This saves the user from having to
        calculate these in a prior step. If this plotting mode is used, the user can
        also specify a ``wavespeed`` to plot a moveout line on the record section.

        Args:
            src_lat (int or float): Source latitude [°]
            src_lon (int or float): Source longitude [°]
            wavespeed (int or float): Moveout line [m/s] to plot on the record section

        Note:
            Can obtain standard :meth:`obspy.core.stream.Stream.plot` behavior by
            setting ``fig=None``.
        """
        if 'reftime' in kwargs:
            kwargs['reftime'] = _preprocess_time(kwargs['reftime'])  # Allow tuples
        if 'fig' not in kwargs:
            from matplotlib.pyplot import figure

            kwargs['fig'] = figure()
        if 'src_lat' in kwargs and 'src_lon' in kwargs:
            kwargs['type'] = 'section'
            for kwarg in 'orientation', 'outfile', 'format', 'handle':
                if kwarg in kwargs:
                    logger.warning(f'Ignoring `{kwarg}` kwarg!')  # Since we set below
            kwargs['orientation'] = 'horizontal'
            kwargs['outfile'] = None  # Disable
            kwargs['format'] = None  # Disable
            kwargs['handle'] = True  # Ensures that the Figure instance is returned
            if 'alpha' not in kwargs:
                kwargs['alpha'] = 1
            src_lat, src_lon = kwargs.pop('src_lat'), kwargs.pop('src_lon')
            wavespeed = kwargs.pop('wavespeed', None)
            st_plot = self.copy()  # Work on a copy of the Stream, since we modify it!
            for tr in st_plot:
                tr.stats.distance = gps2dist_azimuth(  # [m]
                    src_lat, src_lon, tr.stats.latitude, tr.stats.longitude
                )[0]
            fig = st_plot.plot(*args, **kwargs)
            ax = fig.axes[0]  # Get the Axes instance
            # Use km y-axis labels
            ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x / 1000:g}'))
            # Cursor hover labels
            ax.format_coord = lambda x, y: FuncFormatter.fix_minus(
                f'({x:.2f} s, {y / 1000:.2f} km)'
            )
            # Label the stations
            transform = blended_transform_factory(ax.transAxes, ax.transData)
            for tr in st_plot:
                ax.text(
                    1.01,  # Axes space
                    tr.stats.distance,  # [m] Data space
                    tr.id,
                    ha='left',
                    va='center',
                    family='monospace',
                    transform=transform,
                )
            vred = kwargs.get('vred', None)
            if wavespeed is not None:
                color = 'tab:red'
                ymax = ax.get_ylim()[1]  # [m]
                xmax = ymax / wavespeed  # [s]
                if vred is not None:
                    xmax -= ymax / vred  # [s]
                ax.plot(
                    [0, xmax],
                    [0, ymax],
                    color=color,
                    scalex=False,
                    scaley=False,
                )
                transform = blended_transform_factory(ax.transData, ax.transAxes)
                ax.text(
                    xmax,
                    1.02,
                    f'{wavespeed:,g} m/s',
                    color=color,
                    transform=transform,
                    ha='center',
                    va='baseline',
                )
            reftime = kwargs.get('reftime', min([tr.stats.starttime for tr in st_plot]))
            time_format = '%Y-%m-%d %H:%M:%S'
            vred_str = '' if vred is None else f', reduced by {vred:,g} m/s'
            ax.set_xlabel(
                f'Time (s) after {reftime.strftime(time_format)} UTC{vred_str}'
            )
            ax.set_ylabel(
                FuncFormatter.fix_minus(f'Distance (km) from ({src_lat}°, {src_lon}°)')
            )
            fig.tight_layout(pad=0.5)
            return fig
        else:
            if 'wavespeed' in kwargs:
                logger.warning('Ignoring `wavespeed` kwarg!')  # Can't use this here
            fig = super().plot(*args, **kwargs)
            for ax in fig.axes:
                ax.format_coord = (
                    lambda x, y: f'({_num2date(x)}, {FuncFormatter.fix_minus(f"{y:.2g}")} unknown units)'
                )
            return fig

    @staticmethod
    def _time_tuple(t):
        """Cast class:`~obspy.core.utcdatetime.UTCDateTime` to tuples of integers."""
        return (t.year, t.month, t.day, t.hour, t.minute, t.second, t.microsecond)

    @staticmethod
    def _duration_string(tr):
        """Calculate and return a nicely-formatted string duration of a :class:`~obspy.core.trace.Trace`."""
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

    @staticmethod
    @cache
    def _gather_waveforms_cache(starttime, endtime, **kwargs):
        """Wrapper around :func:`saul.waveform.earthscope._gather_waveforms`.

        The input ``starttime`` and ``endtime`` will always be tuples.
        """
        return _gather_waveforms(
            starttime=_preprocess_time(starttime),
            endtime=_preprocess_time(endtime),
            **kwargs,
        )

    def __mul__(self, *args, **kwargs):
        """Modify this method to return a SAUL :class:`Stream` instead of an ObsPy :class:`~obspy.core.stream.Stream`.

        TODO:
            Is this something in ObsPy that should be changed? As-is it's inconsistent;
            why don't they call ``st = self.__class__()`` in their
            :meth:`~obspy.core.stream.Stream.__mul__` method?
        """
        return self.__class__(super().__mul__(*args, **kwargs))

    def __str__(self, *args, **kwargs):
        """Overwrite this method to always show all :class:`~obspy.core.trace.Trace` entries, and to show durations.

        The ``extended`` argument is ignored. Code here was copied (and then modified)
        from :meth:`~obspy.core.stream.Stream.__str__`, from ObsPy v1.4.0.
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
