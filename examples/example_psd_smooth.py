from saul import PSD, Stream

# Get and pre-process array data
st = Stream.from_earthscope(
    'AV', 'BAEI', 'HDF', (2023, 9, 1, 0, 5), (2023, 9, 1, 0, 15)
)
st.detrend().taper(0.05).remove_sensitivity()

# Plot PSD without and with smoothing
db_lim = 20, 110
PSD(st).plot(db_lim=db_lim)
PSD(st).smooth(40).plot(db_lim=db_lim)
