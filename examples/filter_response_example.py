import matplotlib.pyplot as plt
from obspy import read

from saul import PSD, extract_trace_filter_params, obspy_filter_response

tr = read()[0]
tr.detrend('demean').taper(0.05).remove_sensitivity()
freqmin, freqmax = 1, 5
trf = tr.copy().filter('bandpass', freqmin=freqmin, freqmax=freqmax, corners=8)

psd = PSD([tr, trf], win_dur=20)
psd_max = psd.psd[1][1].max()
f, h_db = obspy_filter_response(**extract_trace_filter_params(trf))

psd.plot()
fig = plt.gcf()
ax = fig.axes[0]
ax.axvspan(freqmin, freqmax, color='tab:gray', alpha=0.5, lw=0)
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
