# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import sys
from datetime import datetime

sys.path.insert(0, '../saul/')  # Needed so we can import saul

project = 'SAUL'
author = 'Liam Toney'
copyright = f'{datetime.now().year}, {author}'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.intersphinx',
]
napoleon_numpy_docstring = False  # We are using Google docstring style
autodoc_mock_imports = [
    'lxml',
    'matplotlib',
    'multitaper',
    'numpy',
    'obspy',
    'scipy',
    'waveform_collection',
]
autoclass_content = 'init'

# These only need to cover the packages we reference from the docstrings
intersphinx_mapping = dict(obspy=('https://docs.obspy.org/', None))

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
