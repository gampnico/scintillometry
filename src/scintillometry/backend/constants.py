"""Copyright 2019-2023 Helen Ward, Nicolas Gampierakis, Josef Zink.

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

Defines classes and functions for determining constants.
"""


class AtmosConstants(object):
    """Atmospheric constants.

    There seems to be a contradiction between the values for |a_11|
    in the Scintillometer Theory Manual 1.05 (Chapter 1.1.5), and the
    BLS Manual 1.49 (A.2 Eq.6). The Scintillometer Theory Manual is more
    recent.

    Attributes:
        alpha11 (dict): |Cn2| measurement coefficients |a_11| for each
            BLS type, [|m^(7/3)|].
        alpha12 (dict): |Cn2| measurement coefficients |a_12| for each
            BLS type.
        lamda (float): BLS wavelength, λ [nm].
        lamda_error (float): BLS wavelength error, [nm].
        m1_opt (float): Needed for AT and Aq, from Owens (1967).
        m2_opt (float): Needed for AT and Aq, from Owens (1967).

        at_opt (float): AT coefficient for 880 nm & typical atmospheric
            conditions.
        aq_opt (float): Aq coefficient for 880 nm & typical atmospheric
            conditions.

        most_coeffs_ft (dict[list[tuple, tuple]]): Coefficients for MOST
            functions |f_CT2|, in format:
            [(unstable 01, unstable 02), (stable 01, stable 02)]

        kelvin (float): 0°C in kelvins.
        r_dry (float): Specific gas constant for dry air,
            |R_dry| [J |K^-1| |kg^-1|].
        r_vapour (float): Specific gas contstant for water vapour,
            |R_v| [J |K^-1| |kg^-1|].
        ratio_rmm (float): Ratio of molecular masses of water vapour and
            dry air.
        cp (float): Specific heat capacity of air at constant pressure,
            |c_p| [J |K^-1| |kg^-1|].
        latent_vapour (float): Latent heat of vapourisation at
            20°C [J |kg^-1|].
        rho (float): Density of air at STP, ρ [kg |m^-3|].
        k (float): von Kármán's constant.
        g (float): Gravitational acceleration [|ms^-2|].

    .. |K^-1| replace:: K :sup:`-1`
    .. |kg^-1| replace:: kg :sup:`-1`
    .. |m^(7/3)| replace:: m :sup:`7/3`
    .. |m^-3| replace:: m :sup:`-3`
    .. |ms^-2| replace:: ms :sup:`-2`
    .. |10^-5| replace:: 10 :sup:`-5`
    .. |a_11| replace:: α :sub:`11`
    .. |a_12| replace:: α :sub:`12`
    .. |R_dry| replace:: R :sub:`dry`
    .. |R_v| replace:: R :sub:`v`
    .. |c_p| replace:: c :sub:`p`
    .. |z_eff| replace:: z :sub:`eff`
    .. |z_mean| replace:: :math:`\\bar{z}`
    .. |Cn2| replace:: C :sub:`n`:sup:`2`
    .. |CT2| replace:: C :sub:`T`:sup:`2`
    .. |f_CT2| replace:: f :sub:`CT`:sup:`2`
    .. |L_Ob| replace:: L:sub:`Ob`
    """

    def __init__(self):
        super().__init__()

        self.alpha11 = {
            "450": 4.9491e-2,  # BLS450, 4.48 * D^(7/3), Theory manual 1.1.5
            "900": 4.9491e-2,  # BLS900, 4.48 * D^(7/3), Theory manual 1.1.5
            "2000": 1.9775e-1,  # BLS2000 4.48 * D^(7/3), Theory manual 1.1.5
            "450_pra": 1.0964e-2,  # with path reduction aperture
            "900_pra": 1.0964e-2,
            "2000_pra": 4.5618e-2,
            "450_alt": 5.5339e-2,  # BLS Manual A.2 Eq.6
            "900_alt": 5.5339e-2,
            "2000_alt": 1.9372e-1,
        }
        self.alpha12 = {  # N/A for BLS 450
            "900": 0.50175,
            "2000": 1.4768,
            "900_pra": 0.86308,  # with path reduction aperture
            "2000_pra": 2.2726,
        }

        self.lamda = 880e-9  # Scintillometer Theory Manual recommends 880 nm
        self.lamda_error = 20e-9

        # From Ward et al., (2012)
        self.at_opt = -270e-6  # AT coefficient for 880 nm & typical atmos conditions
        self.aq_opt = -0.685e-6  # Aq coefficient for 880 nm & typical atmos conditions

        # MOST coefficients for f_T
        self.most_coeffs_ft = {  # [(unstable, unstable), (stable, stable)]
            "an1988": [(4.9, 6.1), (4.9, 2.2)],  # Andreas (1998)
            "li2012": [(6.7, 14.9), (4.5, 1.3)],  # Li et al. (2012)
            "wy1971": [(4.9, 7), (4.9, 2.75)],  # Wyngaard et al. (1971)
            "wy1973": [(4.9, 7), (4.9, 2.4)],  # Wyngaard et al. (1973)
            "ma2014": [(6.1, 7.6), (0, 0)],  # Maronga et al. (2014)
            "br2014": [(4.4, 10.2), (0, 0)],  # Braam et al. (2014)
        }

        # Physical constants
        self.kelvin = 273.15
        self.r_dry = 287.04
        self.r_vapour = 461.5
        self.ratio_rmm = 0.622
        self.cp = 1004.67
        self.latent_vapour = 2.45e6
        self.rho = 1.225
        self.k = 0.4
        self.g = 9.81

    def get(self, constant_name: str):
        return getattr(self, constant_name)

    def overwrite(self, constant_name: str, value: float):
        return setattr(self, constant_name, value)
