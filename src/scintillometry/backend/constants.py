"""Copyright 2019-2023 Scintillometry Contributors.

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
    in the Scintillometer Theory Manual 1.05
    (Chapter 1.1.5) [#scintec2022]_, and the BLS Manual 1.49
    (A.2 Eq.6) [#scintec2008]_. The Scintillometer Theory Manual is
    more recent.

    Attributes:
        alpha11 (dict): |Cn2| measurement coefficients |a_11| for each
            BLS type, [|m^(7/3)|].
        alpha12 (dict): |Cn2| measurement coefficients |a_12| for each
            BLS type.
        lamda (float): BLS wavelength, |lamda| [nm].
        lamda_error (float): BLS wavelength error, [nm].

        at_opt (float): |A_T| coefficient for 880 nm & typical
            atmospheric conditions, from Ward et al. (2013).
            [#ward2013]_
        aq_opt (float): |A_q| coefficient for 880 nm & typical
            atmospheric conditions, Ward et al. (2013). [#ward2013]_

        most_coeffs_ft (dict[list[tuple, tuple]]): Coefficients for MOST
            functions |f_CT2|, in format:
            [(unstable 01, unstable 02), (stable 01, stable 02)]

        cp (float): Specific heat capacity of air at constant pressure,
            |c_p| [|JK^-1| |kg^-1|].
        dalr (float): Dry adiabatic lapse rate |Gamma_d| [|Km^-1|]. The
            lapse rate is positive.
        g (float): Gravitational acceleration [|ms^-2|].
        k (float): von Kármán's constant.
        kelvin (float): 0°C in kelvins.
        latent_vapour (float): Latent heat of vapourisation at
            20°C [|Jkg^-1|].
        r_dry (float): Specific gas constant for dry air,
            |R_dry| [|JK^-1| |kg^-1|].
        r_vapour (float): Specific gas constant for water vapour,
            |R_v| [|JK^-1| |kg^-1|].
        ratio_rmm (float): Ratio of molecular masses of water vapour and
            dry air i.e. ratio of gas constants |epsilon|.
        ref_pressure (int): Reference pressure (not SLP), |P_b| [Pa].
        rho (float): Density of air at STP, |rho| [|kgm^-3|].

    """

    def __init__(self):
        """Initialise hardcoded constants."""

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
        self.at_opt = -270e-6  # A_T coefficient for 880 nm & typical atmos conditions
        self.aq_opt = -0.685e-6  # A_q coefficient for 880 nm & typical atmos conditions

        # MOST coefficients for similarity function
        self.most_coeffs_ft = {  # [(unstable, unstable), (stable, stable)]
            "an1988": [(4.9, 6.1), (4.9, 2.2)],  # Andreas (1998)
            "li2012": [(6.7, 14.9), (4.5, 1.3)],  # Li et al. (2012)
            "wy1971": [(4.9, 7), (4.9, 2.75)],  # Wyngaard et al. (1971)
            "wy1973": [(4.9, 7), (4.9, 2.4)],  # Wyngaard et al. (1973)
            "ma2014": [(6.1, 7.6), (0, 0)],  # Maronga et al. (2014)
            "br2014": [(4.4, 10.2), (0, 0)],  # Braam et al. (2014)
        }

        # Physical constants
        self.cp = 1004.67
        self.cp_dry = 1003.5
        self.g = 9.81
        self.dalr = self.g / self.cp
        self.kelvin = 273.15
        self.k = 0.4
        self.latent_vapour = 2.45e6
        self.r_dry = 287.04
        self.r_vapour = 461.5
        self.ratio_rmm = self.r_dry / self.r_vapour
        self.ref_pressure = 100000  # Pa
        self.rho = 1.225

    def get(self, constant_name: str):
        return getattr(self, constant_name)

    def overwrite(self, constant_name: str, value: float):
        return setattr(self, constant_name, value)

    def convert_pressure(self, pressure, base=True):
        """Converts pressure data to pascals [Pa] or hectopascals [hPa].

        The input dataframe or series is copied, as this function may be
        called before the processing of the original data has finished.

        This method handles 0 and NaN values, but only converts
        correctly for pressure values in these intervals:

        - |P| [bar] < 2 bar
        - 2 hPa < |P| [hPa] < 2000 hPa
        - |P| [Pa] > 2000 Pa

        This method should therefore only be used on pre-processed data
        as a *convenience*. By default, converts to pascals.

        Args:
            pressure (Union[pd.DataFrame, pd.Series]): Pressure
                measurements |P| in pascals [Pa], hectopascals [hPa], or
                bars [bar].
            base (bool): If True, converts to pascals [Pa]. Otherwise,
                converts to hectopascals [hPa]. Default True.

        Returns:
            Union[pd.DataFrame, pd.Series]: Pressure measurements |P| in
            either pascals [Pa] or hectopascals [hPa].
        """

        convert = pressure.copy(deep=True)
        check = convert[convert.gt(0)].dropna()  # NaN and 0 fail check

        if not base:
            if (check.lt(2)).any().any():  # P in bar
                convert = convert.multiply(1000)
            elif not (check.lt(2000)).any().any():  # P in Pa
                convert = convert.divide(100)
        else:
            if (check.lt(2)).any().any():  # P in bar
                convert = convert.multiply(100000)
            elif (check.lt(2000)).any().any():  # P in hPa/mbar
                convert = convert.multiply(100)

        return convert

    def convert_temperature(self, temperature, base=True):
        """Converts temperature to Celsius [°C] or kelvins [K].

        The input dataframe or series is copied, as this function may be
        called before the processing of the original data has finished.

        This method handles 0 and NaN values, but only converts
        correctly for temperature values in these intervals:

        - T [K] > 130 K
        - T [°C] < 130 °C

        This method should therefore only be used on pre-processed data
        as a *convenience*. By default, converts to kelvins.

        Args:
            temperature (Union[pd.DataFrame, pd.Series]): Temperature
                measurements |T| in kelvins [K] or Celsius [°C].
            base (bool): If True, converts to kelvins [K]. Otherwise,
                converts to Celsius [°C]. Default True.

        Returns:
            Union[pd.DataFrame, pd.Series]: Temperature measurements |T|
            in either kelvins [K] or Celsius [°C].
        """

        convert = temperature.copy(deep=True)
        check = convert[convert.gt(0)].dropna()  # NaN and 0 fail check

        if (not base) and ((check.gt(130)).any().any()):  # convert to Celsius
            convert = convert - AtmosConstants().kelvin
        elif base and (check.lt(130)).any().any():  # convert to Kelvins
            convert = convert + AtmosConstants().kelvin
        else:
            pass

        return convert
