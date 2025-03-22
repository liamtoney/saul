from saul import Spectrogram, Stream

st = Stream.from_earthscope(
    'TA', 'O20K', 'BHZ', (2019, 6, 21, 0, 0), (2019, 6, 21, 0, 10)
)
st.detrend().taper(0.05).remove_sensitivity()
Spectrogram(st).plot()
