# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'pymsprog'
copyright = '2025, Noemi Montobbio'
author = 'Noemi Montobbio'


# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

import os
import sys
sys.path.insert(0, os.path.abspath('../..'))


import pymsprog
release = pymsprog.__version__
version = ".".join(release.split(".")[:2])  # major.minor


extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",     # for Google-style or NumPy-style docstrings
    #"myst_parser",             # if using Markdown 
    "myst_nb",         # renders MyST notebooks (if using this, remove myst_parser!)
    #"nbsphinx",          # renders standard Jupyter notebooks
    "sphinxcontrib.bibtex",
]

myst_enable_extensions = [
    "dollarmath",
    "linkify",
    "deflist",
    "colon_fence",
    "attrs_inline",
    "attrs_block",
]

bibtex_bibfiles = ["MSbiblio.bib"]   # relative to docs/source/

# How citations look in-text
bibtex_default_style = "plain"        # numeric references
bibtex_reference_style = "author_year"  # for showing only cited references
bibtex_style = "html"  # force HTML rendering

templates_path = ['_templates']
exclude_patterns = []



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "furo"            # "alabaster", or "sphinx_rtd_theme", or "furo", etc.

html_static_path = ['_static']

# Logo setup

html_logo = "_static/logo_py.png"
