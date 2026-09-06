import fnmatch
import logging
import warnings
from functools import cache

from obspy import Inventory, Stream
from obspy.clients.fdsn import RoutingClient
from obspy.clients.fdsn.header import FDSNException
from obspy.core.util.deprecation_helpers import ObsPyDeprecationWarning

logger = logging.getLogger(__name__)


# Lazy-load the RoutingClient so that it is only created when needed
@cache
def _get_client():
    return RoutingClient('earthscope-federator', timeout=30)


def _gather_waveforms(network, station, location, channel, starttime, endtime):
    """Get waveforms (with metadata) from the EarthScope federated FDSN web service."""
    # Create the client (lazy-loaded)
    client = _get_client()
    # Get waveforms
    logger.info('Gathering waveforms...')
    try:
        st = client.get_waveforms(
            network=network,
            station=station,
            location=location,
            channel=channel,
            starttime=starttime,
            endtime=endtime,
        )
    except FDSNException as e:
        msg = str(e).splitlines()[0]
        if '404' in msg:  # "Error 404: No data matched request."
            st = Stream()
        else:
            logger.error(
                f'Error downloading waveforms — returning empty Stream. Message:\n"{msg}"'
            )
            return Stream()
    if not st:
        logger.warning('No valid data gathered — returning empty Stream.')
        return st
    logger.info('Done')
    # Check that all requested stations are present in Stream
    requested_stations = set(station.split(','))
    downloaded_stations = set(tr.stats.station for tr in st)
    for requested_station in requested_stations:
        # The below check works with wildcards, but obviously cannot detect if ALL
        # stations corresponding to a given wildcard (e.g., O??K) were downloaded. Thus,
        # if careful station selection is desired, specify each station explicitly and
        # the below check will then be effective.
        if not fnmatch.filter(downloaded_stations, requested_station):
            logger.warning(
                f'Station {requested_station} not downloaded for this time period.',
            )
    # Get station information
    try:
        inv = client.get_stations(
            network=network,
            station=station,
            location=location,
            channel=channel,
            starttime=starttime,
            endtime=endtime,
            level='response',
        )
    except FDSNException as e:
        msg = str(e).splitlines()[0]
        if '404' in msg:  # "Error 404: No data matched request."
            inv = Inventory()
        else:
            logger.error(
                f'Error downloading station information — returning empty Stream. Message:\n"{msg}"'
            )
            return Stream()
    st.sort()
    for tr in st:
        try:
            coordinates = inv.get_coordinates(tr.id)
            tr.stats.longitude = coordinates['longitude']
            tr.stats.latitude = coordinates['latitude']
            tr.stats.elevation = coordinates['elevation']
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', category=ObsPyDeprecationWarning)
                tr.attach_response(inv)  # TODO: Will be deprecated soon...
        except Exception as e:
            logger.error(f'Error attaching metadata for {tr.id} — removing: {e}')
            st.remove(tr)
    if not st:
        logger.warning('No valid data gathered — returning empty Stream.')
    return st
