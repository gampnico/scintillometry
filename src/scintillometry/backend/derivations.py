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

Derives heat fluxes.
"""

from scintillometry.backend.constants import AtmosConstants


class DeriveScintillometer:
    """Derives structure parameters and fluxes from scintillometer.

    Attributes:
        constants (AtmosConstants): Inherited atmospheric constants.
    """

    def __init__(self):
        super().__init__()
        self.constants = AtmosConstants()

    def derive_ct2(self, dataframe, wavelength=880):
        """Derives the structure parameter of temperature, |CT2|.

        Derives |CT2| from measured structure parameter of refractive
        index |Cn2| and weather data. The equation to derive |CT2| is
        precise to |10^-5| but doesn't account for humidity
        fluctuations: errors are within 3% of inverse Bowen ratio
        (Moene 2003). [#moene2003]_

        Typical transmission beam wavelengths::

            * Ward et al., (2013) [#ward2013]_, Scintillometer Theory
            Manual [#scintec2022]_: 880 nm
            * BLS Manual [#scintec2008]_: 850 nm (± 20 nm)
            * SLS Manual: 670 nm (± 10 nm)

        See specifications in the corresponding hardware maintenance
        manual.

        Args:
            dataframe (pd.DataFrame): Parsed and localised data,
                containing at least |Cn2|, temperature, and pressure.
            wavelength (int): Wavelength of transmitter beam, in
                nanometres. Default 880 nm.

        Returns:
            pd.DataFrame: Input dataframe with updated values for the
            structure parameter of temperature, |CT2| [|K^2m^-2/3|].
        """

        transmit_lambda = wavelength
        # Wavelength-dependent proportionality factor (for 880nm)
        lambda_2 = 7.53 * (10**-3)  # micron^2
        lambda_2_m = lambda_2 / (10**6) ** 2  # m^2
        alpha_factor_2 = 77.6 * (10**-6)  # K hPa^-1
        alpha_factor_1 = alpha_factor_2 * (1 + (lambda_2_m / (transmit_lambda**2)))

        dataframe["CT2"] = (
            dataframe["Cn2"]
            * ((dataframe["temperature_2m"]) ** 4)
            * ((alpha_factor_1 * dataframe["pressure"]) ** -2)  # pressure in hPa!
        )

        return dataframe

    def kinematic_shf(self, dataframe, z_eff):
        """Calculates kinematic sensible heat fluxes.

        Args:
            dataframe (pd.DataFrame): Must contain at least |CT2| and
                temperature data.
            z_eff (float): Effective path height, |z_eff| [m].

        Returns:
            pd.DataFrame: Dataframe with new column for kinematic
            sensible heat flux |Q_0| [|Kms^-1|].
        """

        dataframe["Q_0"] = (
            1.165
            * self.constants.k  # von Kármán's constant
            * z_eff
            * (dataframe["CT2"] ** (3 / 4))
            * (self.constants.g / dataframe["temperature_2m"]) ** (1 / 2)
        )

        return dataframe

    def free_convection_shf(self, dataframe):
        """Calculates surface sensible heat flux under free convection.

        Args:
            dataframe (pd.DataFrame): Must contain at least kinematic
                sensible heat flux, pressure, and temperature data.

        Returns:
            pd.DataFrame: Dataframe with new columns for air density and
            surface sensible heat flux under free convection,
            |H_free| [|Wm^-2|].
        """

        # Air density
        dataframe["rho_air"] = (dataframe["pressure"].multiply(100)).divide(
            dataframe["temperature_2m"].multiply(self.constants.r_dry)
        )
        # Free convection
        dataframe["H_free"] = dataframe["Q_0"].multiply(
            dataframe["rho_air"].multiply(self.constants.cp)
        )

        return dataframe

    def compute_fluxes(self, input_data, effective_height, beam_params=None):
        """Compute kinematic and surface sensible heat fluxes.

        The surface sensible heat flux |H_free| is calculated under free
        convection (little wind shear, high instability).

        Args:
            input_data (pd.DataFrame): Parsed and localised
                scintillometry and weather data. This must contain at
                least |Cn2|, temperature, and pressure.
            effective_height (np.floating): Effective path height,
                |z_eff| [m].
            beam_params (tuple[int, int]): Wavelength and wavelength
                error interval in nanometres. For BLS this is
                typically (880, 20). Default None.

        Returns:
            pd.DataFrame: Updated dataframe containing derived values
            for |CT2|, [|K^2m^-2/3|].
        """

        if beam_params:  # modify the class
            setattr(self.constants, "lamda", beam_params[0] * (10**-9))
            setattr(self.constants, "lamda_error", beam_params[1] * (10**-9))

        beam_params = (self.constants.lamda, self.constants.lamda_error)

        merged_data = self.derive_ct2(dataframe=input_data, wavelength=beam_params[0])
        merged_data = self.kinematic_shf(dataframe=merged_data, z_eff=effective_height)
        merged_data = self.free_convection_shf(dataframe=merged_data)

        return merged_data
