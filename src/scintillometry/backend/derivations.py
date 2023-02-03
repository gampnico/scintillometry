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


def get_switch_time(dataframe, local_time=None):
    """Gets local time of switch between stability conditions.

    This should be determined in order of priority::
        * potential temperature profile (NotImplemented)
        * eddy covariance methods (NotImplemented)
        * global irradiance (i.e. sunrise)

    Args:
        dataframe (pd.DataFrame): Parsed and localised data, containing
            data to construct a potential temperature profile, or eddy
            covariance data, or pressure and temperature.
        local_time (str): Local time of switch between stability
            conditions. Overrides calculations from <dataframe>.

    Returns:
        str: Local time of switch between stability conditions.

    Raises:
        KeyError: No data to calculate switch time. Set manually.
    """

    if not local_time:
        # potential temperature profile
        # eddy covariance

        if "global_irradiance" in dataframe.keys():  # ~sunrise
            local_time = dataframe[dataframe["global_irradiance"] > 20].index[0]
            local_time = local_time.strftime("%H:%M")
        else:
            raise KeyError("No data to calculate switch time. Set manually.")

    return local_time


def derive_ct2(dataframe, wavelength=880):
    """Derives the structure parameter of temperature |CT2|.

    Derives |CT2| from measured structure parameter of temperature |Cn2|
    and weather data. The equation to derive |CT2| is precise to |10^-5|
    but doesn't account for humidity fluctuations - errors are within 3%
    of inverse Bowen ratio (Moene 2003).

    Typical transmission beam wavelengths::

        * Ward et al., (2013), Scintillometer Theory Manual: 880 nm
        * BLS Manual: 850 nm (± 20 nm)
        * SLS Manual: 670 nm (± 10 nm)

    See specifications in the corresponding hardware maintenance manual.

    Args:
        dataframe (pd.DataFrame): Parsed and localised data, containing
            at least |Cn2|, temperature, and pressure.
        wavelength (int): Wavelength of transmitter beam, in
            nanometres. Default 880.

    Returns:
        pd.DataFrame: Input dataframe with updated |CT2| values.
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
