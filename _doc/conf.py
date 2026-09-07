# Project information ------------------------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

from importlib.metadata import version

project = 'SAUL'
author = 'Liam Toney'
version = version('saul-earth')  # Must match the name in `pyproject.toml`

# General configuration ----------------------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.napoleon',
    'sphinx.ext.autodoc',
    'sphinx.ext.todo',
    'sphinx.ext.intersphinx',
    'sphinx.ext.viewcode',
]

templates_path = ['_templates']

# Options for HTML output --------------------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_theme_options = {'collapse_navigation': False}
html_show_copyright = False

# Options for various extensions -------------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/extensions/#built-in-extensions

napoleon_numpy_docstring = False  # We are using Google docstring style

autodoc_mock_imports = [
    'esi_core',
    'lxml',
    'matplotlib',
    'multitaper',
    'numpy',
    'obspy',
    'pandas',
    'scipy',
    'stockwell',
]
autodoc_member_order = 'bysource'
autoclass_content = 'class'
autodoc_class_signature = 'separated'

todo_include_todos = True

# These only need to cover the packages we reference from the docstrings
# fmt: off
intersphinx_mapping = dict(
    multitaper=('https://multitaper.readthedocs.io/en/latest/', None),
    numpy=('https://numpy.org/doc/stable/', None),
    obspy=('https://docs.obspy.org/', None),
    pandas=('https://pandas.pydata.org/docs/', None),
    python=('https://docs.python.org/3/', None),
    scipy=('https://docs.scipy.org/doc/scipy/', None),
)
# fmt: on
