from saul import PSD, Stream

st = Stream.from_earthscope('AK', 'HOM', 'BDF', (2023, 9, 1, 0, 5), (2023, 9, 1, 0, 15))
st.detrend().taper(0.05).remove_response()  # SAUL Stream objects behave like ObsPy's
PSD(st, method='multitaper').plot(show_noise_models=True)

# ---------------------------------------
# Below code not to be shown in README.md
# ---------------------------------------
from pathlib import Path

from matplotlib.pyplot import gcf

output_path = Path(__file__).parents[1] / '_doc' / 'psd_example.png'
gcf().savefig(output_path, dpi=300, bbox_inches='tight')
