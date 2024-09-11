import matplotlib.pyplot as plt
from obspy import read

from saul import PSD, obspy_filter_response

tr = read()[0]
tr.detrend('demean').taper(0.05).remove_sensitivity()

filter_type = 'bandpass'
options = dict(freqmin=1, freqmax=5, corners=8)
trf = tr.copy().filter(filter_type, **options)

psd = PSD([tr, trf], win_dur=20)
psd_max = psd.psd[1][1].max()
f, h_db = obspy_filter_response(filter_type, tr.stats.sampling_rate, **options)

psd.plot()
fig = plt.gcf()
ax = fig.axes[0]
ax.axvspan(options['freqmin'], options['freqmax'], color='tab:gray', alpha=0.5, lw=0)
ax.semilogx(f, h_db + psd_max, color='tab:gray', ls='--')
ax.set_ylim(bottom=-200)
ax.legend(
    labels=[
        'original\nsignal',
        'filtered\nsignal',
        'filter\npassband',
        'filter\nresponse',
    ],
    loc='upper right',
)
fig.show()
