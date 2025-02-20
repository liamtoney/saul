"""
Contains helper functions for tasks related to waveform data gathering, availablity
assessment, and plotting.
"""

import io

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from obspy import UTCDateTime

# Base URL for NSF SAGE availability web service
_BASE_URL = 'https://service.iris.edu/fdsnws/availability/1/query'

# Datetime format for pretty-printed availability timespans
_DATETIME_FORMAT = '%Y-%m-%dT%H:%M:%S'

# Colors for availability timespan plotting
_AVAILABLE_COLOR = 'tab:green'
_UNAVAILABLE_COLOR = 'tab:gray'


def get_availability(
    network,
    station,
    channel,
    starttime,
    endtime,
    location='*',
    print_timespans=True,
    plot=False,
):
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
        print_timespans (bool): Toggle pretty-printing availability timespans to the
            console
        plot (bool): Toggle plotting availability timespans

    Returns:
        :class:`~pandas.DataFrame`: Table of data availability information, with columns
        ``Network``, ``Station``, ``Location``, ``Channel``, ``Earliest``, and
        ``Latest``
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
        dtype={'Network': str, 'Station': str, 'Location': str, 'Channel': str},
        parse_dates=['Earliest', 'Latest'],
        keep_default_na=False,
    )
    if print_timespans:
        _print_availability_df(df, leading_newline=True)
    if plot:
        _plot_availability_df(df, starttime, endtime)
    return df


def _print_availability_df(df, leading_newline=False):
    """Pretty-print availability timespans to the console."""
    max_id_length = (
        df[['Network', 'Station', 'Location', 'Channel']]
        .map(len)  # Number of characters in each SEED code
        .sum(axis='columns')
        .max()
        + 3  # Since the 4 SEED codes are separated by dots in the ID
    )
    lines = []
    for _, row in df.iterrows():
        id_str = f'{row.Network}.{row.Station}.{row.Location}.{row.Channel}'.ljust(
            max_id_length  # Ensures all IDs are aligned
        )
        earliest_str = row.Earliest.strftime(_DATETIME_FORMAT)
        latest_str = row.Latest.strftime(_DATETIME_FORMAT)
        lines.append(f'{id_str} | {earliest_str} - {latest_str}')
    out = '\n'.join(lines)
    if leading_newline:
        out = '\n' + out
    print(out)


def _plot_availability_df(df, starttime, endtime):
    """TODO convert from this draft"""
    df_plot = df.copy()
    df_plot['tr_id'] = df.apply(
        lambda row: f'{row.Network}.{row.Station}.{row.Location}.{row.Channel}',
        axis='columns',
    )
    unique_tr_ids = df_plot['tr_id'].unique()
    figsize = (6.4, min(0.3 * unique_tr_ids.size, 8))
    fig, axs = plt.subplots(nrows=unique_tr_ids.size, sharex=True, figsize=figsize)
    axs = np.atleast_1d(axs)
    for i, tr_id in enumerate(unique_tr_ids):
        # Plot background timespan of query as gray
        axs[i].axvspan(
            starttime.matplotlib_date,
            endtime.matplotlib_date,
            lw=0,
            color=_UNAVAILABLE_COLOR,
            zorder=1,
        )
        df_plot_tr_id = df_plot[df_plot['tr_id'] == tr_id]
        for _, row in df_plot_tr_id.iterrows():
            # Plot availability timespans as green
            axs[i].axvspan(
                row['Earliest'],
                row['Latest'],
                lw=0,
                color=_AVAILABLE_COLOR,
                zorder=2,
            )
        axs[i].set_ylabel(
            tr_id,
            rotation=0,
            ha='right',
            va='center',
            family='monospace',
            fontsize=plt.rcParams['font.size'] - 1,
        )
        axs[i].set_yticks([])
        axs[i].tick_params(axis='x', bottom=False, labelbottom=False)
        for spine in axs[i].spines.values():
            spine.set_visible(False)
        if axs[i] is axs[-1]:
            axs[i].spines['bottom'].set_visible(True)
            axs[i].spines['bottom'].set_position(('outward', 5))
            axs[i].tick_params(axis='x', bottom=True, labelbottom=True)
    loc = axs[-1].xaxis.set_major_locator(mdates.AutoDateLocator())
    axs[-1].xaxis.set_major_formatter(mdates.ConciseDateFormatter(loc))
    axs[-1].set_xlim(starttime.matplotlib_date, endtime.matplotlib_date)
    fig.tight_layout()
    fig.subplots_adjust(hspace=0.3)
    axs[-1].legend(
        handles=[
            plt.Rectangle((0, 0), 1, 1, color=_AVAILABLE_COLOR, label='Available'),
            plt.Rectangle((0, 0), 1, 1, color=_UNAVAILABLE_COLOR, label='Unavailable'),
        ],
        loc='upper right',
        bbox_to_anchor=(0, 0),
        frameon=False,
        borderaxespad=0.5,
        borderpad=0.5,
        handlelength=1.2,
        handleheight=0.6,
    )
    fig.show()


def _preprocess_time(starttime_or_endtime):
    """Cast tuples of integers to :class:`~obspy.core.utcdatetime.UTCDateTime`."""
    if isinstance(starttime_or_endtime, tuple):
        starttime_or_endtime = UTCDateTime(*starttime_or_endtime)
    elif not isinstance(starttime_or_endtime, UTCDateTime):
        raise TypeError('Time must be either a tuple or a UTCDateTime!')
    return starttime_or_endtime
