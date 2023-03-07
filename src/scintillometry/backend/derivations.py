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

Derives heat fluxes.
"""

from scintillometry.backend.constants import AtmosConstants


def derive_ct2(dataframe, wavelength=880):
    """Derives the structure parameter of temperature, |CT2|.

    Derives |CT2| from measured structure parameter of refractive index
    |Cn2| and weather data. The equation to derive |CT2| is precise to
    |10^-5| but doesn't account for humidity fluctuations - errors are
    within 3% of inverse Bowen ratio (Moene 2003).

    Typical transmission beam wavelengths::

        * Ward et al., (2013), Scintillometer Theory Manual: 880 nm
        * BLS Manual: 850 nm (± 20 nm)
        * SLS Manual: 670 nm (± 10 nm)

    See specifications in the corresponding hardware maintenance manual.

    Args:
        dataframe (pd.DataFrame): Parsed and localised data, containing
            at least |Cn2|, temperature, and pressure.
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


def kinematic_shf(dataframe, z_eff):
    """Calculates kinematic sensible heat fluxes.

    Args:
        dataframe (pd.DataFrame): Must contain at least |CT2| and
            temperature data.
        z_eff (float): Effective path height, |z_eff| [m].

    Returns:
        pd.DataFrame: Dataframe with new column for kinematic sensible
        heat flux |Q_0| [|Kms^-1|].
    """

    constants = AtmosConstants()
    dataframe["Q_0"] = (
        1.165
        * constants.k  # von Kármán's constant
        * z_eff
        * (dataframe["CT2"] ** (3 / 4))
        * (constants.g / dataframe["temperature_2m"]) ** (1 / 2)
    )

    return dataframe


def free_convection_shf(dataframe):
    """Calculates surface sensible heat flux under free convection.

    Args:
        dataframe (pd.DataFrame): Must contain at least kinematic
            sensible heat flux, pressure, and temperature data.

    Returns:
        pd.DataFrame: Dataframe with new columns for air density and
        surface sensible heat flux under free convection,
        |H_free| [|Wm^-2|].
    """

    constants = AtmosConstants()
    r_d = constants.r_dry  # J kg^-1 K^-1, specific gas constant for dry air
    cp = constants.cp  # J kg^-1 K^-1, heat capacity of air

    dataframe["rho_air"] = (  # Air density
        100 * dataframe["pressure"] / (r_d * dataframe["temperature_2m"])
    )
    # Free convection
    dataframe["H_free"] = dataframe["Q_0"] * cp * dataframe["rho_air"]

    return dataframe


def compute_fluxes(input_data, effective_height, beam_params=None):
    """Compute kinematic and surface sensible heat fluxes.

    The surface sensible heat flux |H_free| is calculated under free
    convection (little wind shear, high instability).

    Args:
        input_data (pd.DataFrame): Parsed and localised scintillometry
            and weather data. This must contain at least |Cn2|,
            temperature, and pressure.
        effective_height (np.floating): Effective path height,
            |z_eff| [m].
        beam_params (tuple[int, int]): Wavelength and wavelength error
            interval in nanometres. For BLS this is typically (850, 20).

    Returns:
        pd.DataFrame: Updated dataframe containing derived values for
        |CT2|, [|K^2m^-2/3|].
    """

    constants = AtmosConstants()
    if beam_params:  # modify the class
        setattr(constants, "lamda", beam_params[0] * (10**-9))
        setattr(constants, "lamda_error", beam_params[1] * (10**-9))

    beam_params = (constants.lamda, constants.lamda_error)

    merged_data = derive_ct2(dataframe=input_data, wavelength=beam_params[0])
    merged_data = kinematic_shf(dataframe=merged_data, z_eff=effective_height)
    merged_data = free_convection_shf(dataframe=merged_data)

    return merged_data
