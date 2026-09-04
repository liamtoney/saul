# SAUL

[![API documentation status](https://readthedocs.org/projects/saul/badge/?version=latest)](https://saul.rtfd.io/)

**SAUL** is the **S**eismo**A**coustic **U**tilities **L**ibrary. It's my take on the
collection of tools that I imagine exist, in some form, on every seismoacoustican's
computer — utilities for gathering waveform data, plotting waveforms in the time and
frequency domain, visualizing key metadata such as station locations, _et cetera._ The
goal of SAUL is to make these fundamental data exploration tools as easy-to-use as
possible. Thus, priority is placed upon straightforward (e.g., easily memorized)
commands and time-saving helper functions — while attempting to leverage existing
dependencies as much as possible to avoid duplicated effort.

> 🚧 **Disclaimer** 🚧  
> As a workhorse "everyday tools" repository, SAUL is currently (perpetually?) under
> rapid development. Expect to encounter breaking changes after a `git pull` update!

## Installing

SAUL is primarily developed on macOS, but it ought to work on Linux — and Windows via
[Windows Subsystem for Linux (WSL)](https://learn.microsoft.com/en-us/windows/wsl/).

This assumes that you've already
[installed uv](https://docs.astral.sh/uv/getting-started/installation/), and that
you've navigated to a target directory of your choosing:
```shell
uv venv --python 3.11
uv pip install git+https://github.com/liamtoney/saul.git
```
This creates a `.venv/` folder in the target directory with SAUL and its dependencies
installed.

If you'd rather install SAUL into an existing, e.g.,
[`mamba`](https://mamba.readthedocs.io/en/latest/index.html) environment instead of
using uv, activate that environment and use `pip` directly:
```shell
pip install git+https://github.com/liamtoney/saul.git
```

## Using

SAUL works best in an interactive Python console, which is why
[IPython](https://ipython.org/) is included as a dependency. There are multiple options
for launching IPython with SAUL available. You can run IPython directly:
```shell
<target_directory>/.venv/bin/ipython
```
Where `<target_directory>` is the path to the directory in which you installed SAUL. If
you're already in that directory, you can also just run:
```shell
uv run ipython
```
Finally, you can activate the virtual environment and launch IPython directly, via (for
example):
```shell
source <target_directory>/.venv/bin/activate.fish  # Other shells have different scripts
ipython
```

Here's a simple
[usage example](https://github.com/liamtoney/saul/blob/main/examples/example_psd.py)
which highlights SAUL's object-oriented interface:
```python
from saul import PSD, Stream

st = Stream.from_earthscope('AK', 'HOM', 'BDF', (2023, 9, 1, 0, 5), (2023, 9, 1, 0, 15))
st.detrend().taper(0.05).remove_response()  # SAUL Stream objects behave like ObsPy's
PSD(st, method='multitaper').plot(show_noise_models=True)
```
<img src="_doc/example_psd.png" width=550>

For detailed usage information, see the [API documentation](https://saul.rtfd.io/).

## Developing

To develop SAUL, first clone this repository and navigate to the root directory. Then
run:
```shell
uv sync --all-groups
```
This creates a `.venv/` folder in the repository root with an editable SAUL and all of
its dependencies — including those for development and documentation — installed. You
can run IPython with SAUL in the same way as described above, but from anywhere:
```shell
uv run --project <repository_root> ipython
```
Where `<repository_root>` is the path to the root of the SAUL repository.
