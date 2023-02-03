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
