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
release = "0.20.a0"

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
.. |gm^-3| replace:: g :math:`\\cdot` m :sup:`-3`
.. |Jkg^-1| replace:: J :math:`\\cdot` kg :sup:`-1`
.. |JK^-1| replace:: J :math:`\\cdot` K :sup:`-1`
.. |K^-1| replace:: K :sup:`-1`
.. |K^2m^-2/3| replace:: K :math:`^{{2}}\\cdot` m :math:`^{{-2/3}}`
.. |Kms^-1| replace:: K :math:`\\cdot` ms :sup:`-1`
.. |kg^-1| replace:: kg :sup:`-1`
.. |kgkg^-1| replace:: kg :math:`\\cdot` kg :sup:`-1`
.. |kgm^-3| replace:: kg :math:`\\cdot` m :sup:`-3`
.. |m^(7/3)| replace:: m :sup:`7/3`
.. |m^-3| replace:: m :sup:`-3`
.. |ms^-1| replace:: ms :sup:`-1`
.. |ms^-2| replace:: ms :sup:`-2`
.. |s^-1| replace:: s :sup:`-1`
.. |Wm^-2| replace:: W :math:`\\cdot` m :sup:`-2`
.. |10^-5| replace:: 10 :sup:`-5`
.. |a_11| replace:: :math:`\\alpha_{{11}}`
.. |a_12| replace:: :math:`\\alpha_{{12}}`
.. |A_T| replace:: :math:`A_{{T}}`
.. |A_q| replace:: :math:`A_{{q}}`
.. |c_p| replace:: :math:`c_{{p}}`
.. |Cn2| replace:: :math:`C_{{n}}^{{2}}`
.. |CT2| replace:: :math:`C_{{T}}^{{2}}`
.. |dy/dx| replace:: :math:`\\partial y/\\partial x`
.. |e| replace:: :math:`e`
.. |epsilon| replace:: :math:`\\epsilon`
.. |f_CT2| replace:: :math:`f_{{C_{{T}}^{{2}}}}`
.. |H| replace:: :math:`H`
.. |H_free| replace:: :math:`H_{{free}}`
.. |lamda| replace:: :math:`\\lambda`
.. |LOb| replace:: :math:`L_{{Ob}}`
.. |P| replace:: :math:`P`
.. |P_0| replace:: :math:`P_{{0}}`
.. |P_dry| replace:: :math:`P_{{dry}}`
.. |P_MSL| replace:: :math:`P_{{MSLP}}`
.. |P_z| replace:: :math:`P_{{z}}`
.. |Psi_m| replace:: :math:`\\Psi_{{m}}`
.. |Q_0| replace:: :math:`Q_{{0}}`
.. |r| replace:: :math:`r`
.. |R_dry| replace:: :math:`R_{{dry}}`
.. |R_v| replace:: :math:`R_{{v}}`
.. |rho| replace:: :math:`\\rho`
.. |rho_v| replace:: :math:`\\rho_{{v}}`
.. |T| replace:: :math:`T`
.. |T_v| replace:: :math:`T_{{v}}`
.. |T_z| replace:: :math:`T_{{z}}`
.. |theta| replace:: :math:`\\theta`
.. |theta*| replace:: :math:`\\theta^{{*}}`
.. |u| replace:: :math:`u`
.. |u*| replace:: :math:`u^{{*}}`
.. |z| replace:: :math:`z`
.. |z_eff| replace:: :math:`z_{{eff}}`
.. |z_mean| replace:: :math:`\\bar{{z}}`
.. |z_scan| replace:: :math:`z_{{scan}}`
.. |z_u| replace:: :math:`z_{{u}}`
.. |z_0| replace:: :math:`z_{{0}}`
"""
