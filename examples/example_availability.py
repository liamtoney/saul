from saul import get_availability

df = get_availability(
    'AK,AV',
    'HOM,RC01,BAEI',
    '?DF,BHZ',
    (2024, 8, 1),
    (2025, 2, 1),
)

# Make one function for pretty-printing
# Make one function for plotting

# Testing ploting below... shoddy...
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np

df_plot = df.copy()
df_plot['tr_id'] = df.apply(
    lambda row: f'{row.Network}.{row.Station}.{row.Location}.{row.Channel}',
    axis='columns',
)
unique_tr_ids = df_plot['tr_id'].unique()
figsize = (6.4, min(1 * unique_tr_ids.size, 8))
fig, axs = plt.subplots(nrows=unique_tr_ids.size, sharex=True, figsize=figsize)
axs = np.atleast_1d(axs)
for i, tr_id in enumerate(unique_tr_ids):
    df_plot_tr_id = df_plot[df_plot['tr_id'] == tr_id]
    for _, row in df_plot_tr_id.iterrows():
        axs[i].axvspan(
            row['Earliest'],
            row['Latest'],
            lw=0,
            color='tab:green',
        )
    axs[i].patch.set_facecolor('tab:gray')
    axs[i].set_ylabel(tr_id, rotation=0, ha='right', va='center', family='monospace')
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
axs[-1].autoscale(enable=True, axis='x', tight=True)
fig.tight_layout()
fig.subplots_adjust(hspace=0.1)
fig.show()
