"""Copyright 2023 Scintillometry-Tools Contributors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

=====

Configuration file for the Sphinx documentation builder.

For the full list of built-in configuration values, see the documentation:
https://www.sphinx-doc.org/en/master/usage/configuration.html
"""

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
release = "0.22.a2"

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
html_theme_options = {"navigation_depth": 5}
html_static_path = ["_static"]
html_show_sphinx = False
trim_footnote_reference_space = True

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
.. |N^2| replace:: :math:`N^{{2}}`
.. |P| replace:: :math:`P`
.. |P_0| replace:: :math:`P_{{0}}`
.. |P_b| replace:: :math:`P_{{b}}`
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
.. |z_stn| replace:: :math:`z_{{stn}}`
.. |z_u| replace:: :math:`z_{{u}}`
.. |z_0| replace:: :math:`z_{{0}}`
"""

rst_epilog = f"""
.. |andreas1988| replace:: E.L. Andreas (1988).
    *Estimating Cn2 Over Snow and Sea Ice from Meteorological Data*.
    J. Opt. Soc. Am. A.,  Vol. 5:4, [481-495].
    DOI: `<https://doi.org/10.1364/JOSAA.5.000481>`__

.. |braam2014| replace:: M. Braam, A.F. Moene, F. Beyrich, *et al.* (2014).
    *Similarity Relations for CT2 in the Unstable Atmospheric Surface Layer:
    Dependence on Regression Approach, Observation Height and Stability Range*.
    Boundary-Layer Meteorology Vol. 153, [63-87].
    DOI: `<https://doi.org/10.1007/s10546-014-9938-y>`__

.. |hartogensis2003| replace:: O.K. Hartogensis, C.J. Watts, J. Rodriguez, and
    H.A.R. De Bruin (2003).
    *Derivation of an Effective Height for Scintillometers: La Poza Experiment
    in Northwest Mexico*.
    J. Hydrometeor., Vol. 4, [915-928].
    DOI: `<https://doi.org/10.1175/1525-7541(2003)004\<0915:DOAEHF\>2.0.CO;2>`__

.. |kleissl2008| replace:: J. Kleissl, J. Gomez, S.H. Hong, *et al.* (2008).
    *Large Aperture Scintillometer Intercomparison Study*.
    Boundary-Layer Meteorology, Vol. 128, [133-150] (2008).
    DOI: `<https://doi.org/10.1007/s10546-008-9274-1>`__

.. |kooijmans2016| replace:: L.M.J. Kooijmans, O.K. Hartogensis (2014).
    *Surface-Layer Similarity Functions for Dissipation Rate and Structure
    Parameters of Temperature and Humidity Based on Eleven Field Experiments*.
    Boundary-Layer Meteorol Vol. 160, [501-527].
    DOI: `<https://doi.org/10.1007/s10546-016-0152-y>`__

.. |li2012| replace:: D. Li, E. Bou-Zeid & H.A.R. De Bruin (2012).
    *Monin-Obukhov Similarity Functions for the Structure Parameters of
    Temperature and Humidity*.
    Boundary-Layer Meteorology Vol. 145, [45-67].
    DOI: `<https://doi.org/10.1007/s10546-011-9660-y>`__

.. |maronga2014| replace:: B. Maronga, O.K. Hartogensis, S. Raasch
    *et al.* (2014).
    *The Effect of Surface Heterogeneity on the Structure Parameters of
    Temperature and Specific Humidity: A Large-Eddy Simulation Case Study for
    the LITFASS-2003 Experiment*.
    Boundary-Layer Meteorology, Vol. 153, [441-470].
    DOI: `<https://doi.org/10.1007/s10546-014-9955-x>`__

.. |moene2003| replace:: A.F. Moene (2003).
    *Effects of Water Vapour on the Structure Parameter of the Refractive Index
    for Near-Infrared Radiation*.
    Boundary-Layer Meteorology, Vol. 107, [635-653].
    DOI: `<https://doi.org/10.1023/A:1022807617073>`__

.. |owens1967| replace:: J.C. Owens (1967).
    *Optical Refractive Index of Air: Dependence on Pressure, Temperature and
    Composition*.
    Appl. Opt., 6:1 [51-9]. PMID: 20057695.
    DOI: `<https://doi.org/10.1364/AO.6.000051>`__

.. |paulson1970| replace:: C.A. Paulson (1970).
    *The Mathematical Representation of Wind Speed and Temperature Profiles in
    the Unstable Atmospheric Surface Layer*.
    J. Appl. Meteorology, Vol. 9:6 [857-861].
    DOI: `<https://doi.org/10.1175/1520-0450(1970)009\<0857\:TMROWS\>2.0.CO;2>`__

.. |scintec2022| replace:: Scintec AG (2022).
    *Scintec Scintillometers Theory Manual (SLS/BLS)*. Version 1.05.
    Scintec AG, Rottenburg, Germany.

.. |scintec2008| replace:: Scintec AG (2008).
    *Scintec Boundary Layer Scintillometer User Manual*. Version 1.49.
    Scintec AG, Rottenburg, Germany.

.. |thiermann1992| replace:: V. Thiermann, H. Grassl (1992).
    *The Measurement of Turbulent Surface-Layer Fluxes by Use of Bichromatic
    Scintillation*.
    Boundary-Layer Meteorology, Vol. 58, [367-389].
    DOI: `<https://doi.org/10.1007/BF00120238>`__

.. |ward2013| replace:: H.C. Ward, J.G. Evans, O.K. Hartogensis, A.F. Moene,
    H.A.R. De Bruin, C.S.B. Grimmond (2013).
    *A Critical Revision of the Estimation of the Latent Heat Flux from
    Two-Wavelength Scintillometry*.
    Q.J.R. Meteorology Soc., Vol. 139 [1912-1922].
    DOI: `<https://doi.org/10.1002/qj.2076>`__

.. |wyngaard1971| replace:: J. C. Wyngaard, Y. Izumi, and S. A. Collins (1971).
    *Behavior of the Refractive-Index-Structure Parameter near the Ground*.
    J. Opt. Soc. Am., Vol. 61:12, [1646-1650].
    DOI: `<https://doi.org/10.1364/JOSA.61.001646>`__

.. |zhangAnthes1982| replace:: D. Zhang & R. A. Anthes (1982).
    *A High-Resolution Model of the Planetary Boundary Layer - Sensitivity Tests
    and Comparisons with SESAME-79 Data*.
    J. Appl. Meteorology, Vol. 21 [1594-1609].
"""
