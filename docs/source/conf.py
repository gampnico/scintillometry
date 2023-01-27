# pylint: skip-file
# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- General configuration ---------------------------------------------------

import os
import sys

sys.path.insert(0, os.path.abspath("../src/scintillometry/*"))

# -- Remove Copyright Boilerplate --------------------------------------------


def remove_boilerplate(app, what, name, obj, options, lines):
    if what == "module":
        del lines[:16]


def setup(app):
    app.connect("autodoc-process-docstring", remove_boilerplate)


# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "Scintillometry Tools"
copyright = "2023, Nicolas Gampierakis"
author = "Nicolas Gampierakis"
release = "0.4.a3"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.coverage",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
]

templates_path = ["_templates"]
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
