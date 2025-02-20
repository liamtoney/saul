"""
Contains helper functions for tasks related to waveform data gathering, availablity
assessment, and plotting.
"""

import io

import pandas as pd
import requests
from obspy import UTCDateTime

# Base URL for NSF SAGE availability web service
_BASE_URL = 'https://service.iris.edu/fdsnws/availability/1/query'

# Datetime format for pretty-printed availability timespans
_DATETIME_FORMAT = '%Y-%m-%dT%H:%M'


def get_availability(network, station, channel, starttime, endtime, location='*'):
    """Obtain waveform availability timespans.

    Uses the NSF SAGE availability web service:
    https://service.iris.edu/fdsnws/availability/1/

    Args:
        network (str): SEED network code
        station (str): SEED station code
        channel (str): SEED channel code
        starttime (tuple or :class:`~obspy.core.utcdatetime.UTCDateTime`): Start
            time for availability search; for tuple input the format is integers:
            ``(year, month, day[, hour[, minute[, second[, microsecond]]])``
        endtime (tuple or :class:`~obspy.core.utcdatetime.UTCDateTime`): End time
            for availability search (same format as ``starttime``)
        location (str): SEED location code

    Returns:
        :class:`~pandas.DataFrame` Table of data availability information
    """
    # Ensure we have UTCDateTime objects to start
    starttime = _preprocess_time(starttime)
    endtime = _preprocess_time(endtime)
    params = dict(
        net=network,
        sta=station,
        loc=location,
        cha=channel,
        start=starttime,
        end=endtime,
        merge='samplerate,quality,overlap',
        format='geocsv',
        nodata='404',
    )
    print('-------------------------')
    print('GETTING AVAILABILITY INFO')
    print('-------------------------')
    response = requests.get(_BASE_URL, params=params)
    if response.status_code == 404:
        print('No data available for this request!')
        return pd.DataFrame()
    print('Done')
    df = pd.read_table(
        io.StringIO(response.text),
        sep='|',
        comment='#',
        dtype=dict(
            Network=str,
            Station=str,
            Location=str,
            Channel=str,
        ),
        parse_dates=['Earliest', 'Latest'],
        keep_default_na=False,
    )
    print()
    _print_availability_df(df)
    return df


def _print_availability_df(df):
    tr_ids = df.apply(
        lambda row: f'{row.Network}.{row.Station}.{row.Location}.{row.Channel}',
        axis='columns',
    )
    id_length = tr_ids.map(len).max()
    lines = []
    for i, row in df.iterrows():
        tr_id_str = tr_ids[i].ljust(id_length)
        starttime_str = row.Earliest.strftime(_DATETIME_FORMAT)
        endtime_str = row.Latest.strftime(_DATETIME_FORMAT)
        lines.append(f'{tr_id_str} | {starttime_str} - {endtime_str}')
    print('\n'.join(lines))


def _plot_availability_df():
    pass


def _preprocess_time(starttime_or_endtime):
    """Cast tuples of integers to :class:`~obspy.core.utcdatetime.UTCDateTime`."""
    if isinstance(starttime_or_endtime, tuple):
        starttime_or_endtime = UTCDateTime(*starttime_or_endtime)
    elif not isinstance(starttime_or_endtime, UTCDateTime):
        raise TypeError('Time must be either a tuple or a UTCDateTime!')
    return starttime_or_endtime
