# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html
#
# pylint: skip-file
# flake8: noqa: F541

# -- General configuration ---------------------------------------------------

import os
import sys
from datetime import date

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
copyright = f"2019-{date.today().year}, Scintillometry Tools Contributors"
author = "Scintillometry Tools Contributors"
release = "0.17.a0"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.coverage",
    "sphinx.ext.napoleon",
    "sphinx.ext.todo",
    "sphinx.ext.viewcode",
]

templates_path = ["_templates"]
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]

rst_prolog = f"""
.. |K^-1| replace:: K :sup:`-1`
.. |kg^-1| replace:: kg :sup:`-1`
.. |kgkg^-1| replace:: :math:`kg\\cdotkg^{{-1}}`
.. |kgm^-3| replace:: kgm :sup:`-3`
.. |m^(7/3)| replace:: m :sup:`7/3`
.. |m^-3| replace:: m :sup:`-3`
.. |ms^-2| replace:: ms :sup:`-2`
.. |s^-1| replace:: s :sup:`-1`
.. |10^-5| replace:: 10 :sup:`-5`
.. |a_11| replace:: :math:`\\alpha_{{11}}`
.. |a_12| replace:: :math:`\\alpha_{{12}}`
.. |AT| replace:: A :sub:`T`
.. |Aq| replace:: A :sub:`q`
.. |c_p| replace:: c :sub:`p`
.. |Cn2| replace:: :math:`C_{{n}}^{{2}}`
.. |CT2| replace:: :math:`C_{{T}}^{{2}}`
.. |e| replace:: :math:`e`
.. |epsilon| replace:: :math:`\\epsilon`
.. |f_CT2| replace:: :math:`f_{{C_{{T}}^{{2}}}}`
.. |lamda| replace:: :math:`\\lambda`
.. |LOb| replace:: :math:`L_{{Ob}}`
.. |P_0| replace:: :math:`P_{{0}}`
.. |P_MSLP| replace:: :math:`P_{{MSLP}}`
.. |Psi_m| replace:: :math:`\\Psi_{{m}}`
.. |R_dry| replace:: R :sub:`dry`
.. |R_v| replace:: R :sub:`v`
.. |rho| replace:: :math:`\\rho`
.. |rho_v| replace:: :math:`\\rho_{{v}}`
.. |T_v| replace:: T :sub:`v`
.. |theta| replace:: :math:`\\theta`
.. |theta*| replace:: :math:`\\theta^{{*}}`
.. |u| replace:: :math:`u`
.. |u*| replace:: :math:`u^{{*}}`
.. |z_eff| replace:: z :sub:`eff`
.. |z_mean| replace:: :math:`\\bar{{z}}`
.. |z_u| replace:: z :sub:`u`
.. |z_0| replace:: z :sub:`0`
"""
