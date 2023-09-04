# SAUL

**SAUL** is the **S**eismo**A**coustic **U**tilities **L**ibrary.

## Installing

Both of these options assume that you've already installed the
[`mamba`](https://mamba.readthedocs.io/en/latest/) package manager (don't bother with
[`conda`](https://docs.conda.io/en/latest/)), and that you've cloned this repository and
have navigated to the root directory.

**Option 1:** Create a new environment named `saul`.
```
mamba env create
```

**Option 2:** Install into an existing environment of your choosing.
```
mamba env update --name <existing_environment>
```

## Using

Here's a simple usage example that highlights SAUL's object-oriented interface:
```python
from saul import PSD, Stream

st = Stream.from_server('AK', 'HOM', 'BDF', (2023, 9, 1, 0, 5), (2023, 9, 1, 0, 15))
st.detrend().taper(0.05).remove_response()  # SAUL Stream objects behave like ObsPy's
PSD(st, method='multitaper').plot(show_noise_models=True)
```
<img src="psd_example.png" width=550>

## Developing

To install the development packages for SAUL, run the following command from the root
directory of this repository, with your environment (see [Installing](#installing))
activated.
```
pip install --requirement requirements.txt
```
