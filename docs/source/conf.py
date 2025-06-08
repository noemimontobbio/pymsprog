# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'pymsprog'
copyright = '2025, Noemi Montobbio'
author = 'Noemi Montobbio'
release = '0.1.1'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

import os
import sys
sys.path.insert(0, os.path.abspath('../..'))

############ test
import logging
try:
    import pymsprog
    logging.warning("pymsprog imported successfully")
except ImportError:
    logging.error("Failed to import pymsprog")
    sys.exit(1)  # stop the build with an error
############


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
