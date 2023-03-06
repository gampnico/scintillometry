"""Copyright 2019-2023 Nicolas Gampierakis, Josef Zink.

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

Constructs profiles from vertical measurements.
"""

import numpy as np
import pandas as pd

from scintillometry.backend.constants import AtmosConstants


class ProfileConstructor(AtmosConstants):
    def __init__(self):
        super().__init__()
        self.constants = AtmosConstants()

    def get_water_vapour_pressure(self, abs_humidity, temperature):
        """Calculate water vapour pressure.

        Args:
            abs_humidity (pd.DataFrame): Absolute humidity,
                |rho_v| [kgm^-3].
            temperature (pd.DataFrame): Temperature, [K].

        Returns:
            pd.DataFrame: Water vapour pressure, |e| [Pa].
        """

        wvp = abs_humidity.multiply(temperature).multiply(self.r_vapour)

        return wvp

    def get_air_pressure(self, pressure, ref_z, alt_z, air_temp):
        """Calculate air pressure at specific altitude.

        Uses hypsometric equation.

        Args:
            pressure (pd.Series): Base pressure, |P_0| [Pa].
            ref_z (int): Base measurement altitude, |z_0| [m].
            alt_z (int): Desired altitude, |z| [m].
            air_temp (pd.Series): Air temperature, [K].

        Returns:
            pd.Series: Air pressure at desired altitude, |P| [Pa].
        """

        alt_pressure = pressure * np.exp(
            -self.g * (alt_z - ref_z) / (self.r_dry * air_temp)
        )

        return alt_pressure

    def extrapolate_air_pressure(self, surface_pressure, temperature):
        """Extrapolates pressure measurements to scan levels.

        Input dataframes should have matching indices.

        Args:
            surface_pressure (pd.Series): Surface pressure, |P_0| [hPa].
            temperature (pd.DataFrame): Temperature at target altitudes.

        Returns:
            pd.DataFrame: Air pressure for each target altitude.
        """

        air_pressure = pd.DataFrame(
            index=temperature.index, columns=temperature.columns
        )
        air_pressure[0] = surface_pressure * 100  # convert hPa to Pa
        target_cols = air_pressure.columns.difference([0])
        for col_idx in target_cols:
            air_pressure[col_idx] = self.get_air_pressure(
                pressure=air_pressure[0],
                ref_z=0,
                alt_z=col_idx,
                air_temp=temperature[col_idx],
            )

        return air_pressure

    def get_mixing_ratio(self, wv_pressure, d_pressure):
        """Calculate mixing ratio for dry air pressure.

        Args:
            wv_pressure (pd.DataFrame): Water vapour pressure, |e| [Pa].
            d_pressure (pd.DataFrame): Dry air pressure, [Pa].

        Returns:
            pd.DataFrame: Mixing ratio, r |kgkg^-1|.
        """

        m_ratio = (wv_pressure.multiply(self.r_dry)).divide(
            (d_pressure).multiply(self.r_vapour)
        )

        return m_ratio

    def get_virtual_temperature(self, temperature, mixing_ratio):
        """Calculate virtual temperature.

        Args:
            temperature (pd.DataFrame): Air temperature, [K].
            d_pressure (pd.DataFrame): Dry air pressure, [Pa].

        Returns:
            pd.DataFrame: Virtual temperature, |Tv| [K].
        """

        v_temp = temperature * (1 + 0.61 * mixing_ratio)

        return v_temp

    def get_reduced_pressure(self, station_pressure, virtual_temperature, elevation):
        """Reduce station pressure to mean sea-level pressure.

        Args:
            station_pressure (pd.DataFrame): Pressure at station
                elevation, |P_0| [Pa].
            virtual_temperature (pd.DataFrame): Virtual temperature at
                station elevation, |T_v| [K].
            elevation (float): Altitude above sea level, [m].

        Returns:
            pd.DataDrame: Mean sea-level pressure, |P_MSLP| [Pa].
        """

        mslp = station_pressure.multiply(
            np.exp(
                (virtual_temperature.multiply(np.abs(self.g)))
                .rdiv(self.r_dry)
                .rdiv(elevation)
            )
        )

        return mslp
