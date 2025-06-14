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
    "myst_parser",             # if using Markdown
]


templates_path = ['_templates']
exclude_patterns = []



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "furo"            # "alabaster", or "sphinx_rtd_theme", or "furo", etc.

html_static_path = ['_static']

# Logo setup

html_logo = "_static/logo_py.png"
