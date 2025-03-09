import matplotlib.pyplot as plt

from saul import PSD, Stream

# Get and pre-process array data
st = Stream.from_earthscope(
    'AV', 'BAEI', 'HDF', (2023, 9, 1, 0, 5), (2023, 9, 1, 0, 15)
)
st.detrend().taper(0.05).remove_sensitivity()

# Figure 1 — No smoothing
PSD(st).plot()
fig1 = plt.gcf()
fig1.axes[0].set_title('No smoothing')
fig1.tight_layout()
fig1.show()

# Figure 2 — Konno–Ohmachi smoothing
bandwidth = 40
PSD(st).smooth(bandwidth).plot()
fig2 = plt.gcf()
fig2.axes[0].set_title(f'Smoothing, bandwidth = {bandwidth}')
fig2.axes[0].set_ylim(fig1.axes[0].get_ylim())  # Easier comparison
fig2.tight_layout()
fig2.show()
