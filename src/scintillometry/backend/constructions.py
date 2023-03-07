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

These definitions avoid confusion with interchangeable terms:

- **Elevation**: Vertical distance between mean sea-level and station.
- **Height**: Vertical distance between station elevation and
  measurement point.
- **Altitude**: Vertical distance between mean sea-level and measurement
  point.
"""

import numpy as np
import pandas as pd

from scintillometry.backend.constants import AtmosConstants


class ProfileConstructor(AtmosConstants):
    def __init__(self):
        super().__init__()
        self.constants = AtmosConstants()

    def get_water_vapour_pressure(self, abs_humidity, temperature):
        """Derive water vapour pressure from vertical measurements.

        Args:
            abs_humidity (pd.DataFrame): Vertical measurements, absolute
                humidity, |rho_v| [|kgm^-3|].
            temperature (pd.DataFrame): Vertical measurements,
                temperature, |T| [K].

        Returns:
            pd.DataFrame: Derived vertical measurements for water vapour
            pressure, |e| [Pa].
        """

        wvp = abs_humidity.multiply(temperature).multiply(self.r_vapour)

        return wvp

    def get_air_pressure(self, pressure, air_temperature, z_target, z_ref=0):
        """Derive air pressure at specific height.

        Uses hypsometric equation. By default, heights are relative to
        the station elevation, i.e. <z_target> - <z_ref> = <z_target>,
        <z_ref> = 0.

        Args:
            pressure (pd.Series): Air pressure at reference measurement
                height, |P_0| [Pa].
            air_temperature (pd.Series): Air temperatures at target
                measurement height, |T_z| [K].
            z_target (int): Target measurement height, |z| [m].
            z_ref (int): Reference measurement height, |z_0| [m].
                Default 0.

        Returns:
            pd.Series: Air pressure at target heights, |P_z| [Pa].
        """

        alt_pressure = pressure * np.exp(
            -self.g * (z_target - z_ref) / (self.r_dry * air_temperature)
        )

        return alt_pressure

    def extrapolate_air_pressure(self, surface_pressure, temperature):
        """Extrapolates reference pressure measurements to scan levels.

        Input dataframes must have matching indices. Converts hPa to Pa.

        Args:
            surface_pressure (pd.Series): Air pressure measurements at
                reference measurement height, |P_0| [hPa].
            temperature (pd.DataFrame): Vertical measurements, air
                temperature at target heights, |T| [K].

        Returns:
            pd.DataFrame: Derived vertical measurements for pressure,
            |P| [Pa].
        """

        air_pressure = pd.DataFrame(
            index=temperature.index, columns=temperature.columns
        )
        air_pressure[0] = surface_pressure * 100  # convert hPa to Pa
        target_cols = air_pressure.columns.difference([0])
        for col_idx in target_cols:
            air_pressure[col_idx] = self.get_air_pressure(
                pressure=air_pressure[0],
                air_temperature=temperature[col_idx],
                z_target=col_idx,
                z_ref=0,
            )

        return air_pressure

    def get_mixing_ratio(self, wv_pressure, d_pressure):
        """Calculate mixing ratio for dry air pressure.

        Args:
            wv_pressure (pd.DataFrame): Vertical measurements, water
                vapour pressure, |e| [Pa].
            d_pressure (pd.DataFrame): Vertical measurements, dry air
                pressure, |P_dry| [Pa].

        Returns:
            pd.DataFrame: Derived vertical measurements for mixing
            ratio, |r| [|kgkg^-1|].
        """

        m_ratio = (wv_pressure.multiply(self.r_dry)).divide(
            (d_pressure).multiply(self.r_vapour)
        )

        return m_ratio

    def get_virtual_temperature(self, temperature, mixing_ratio):
        """Derive virtual temperature from vertical measurements.

        Args:
            temperature (pd.DataFrame): Vertical measurements, air
                temperature, |T| [K].
            d_pressure (pd.DataFrame): Vertical measurements, dry air
                pressure, |P_dry| [Pa].

        Returns:
            pd.DataFrame: Derived vertical measurements for virtual
            temperature, |T_v| [K].
        """

        v_temp = temperature * (1 + 0.61 * mixing_ratio)

        return v_temp

    def get_reduced_pressure(self, station_pressure, virtual_temperature, elevation):
        """Reduce station pressure to mean sea-level pressure.

        Args:
            station_pressure (pd.DataFrame): Vertical measurements,
                pressure relative to station elevation, |P_z| [Pa].
            virtual_temperature (pd.DataFrame): Vertical measurements,
                virtual temperature, |T_v| [K].
            elevation (float): Station elevation above sea level,
                |z| [m].

        Returns:
            pd.DataDrame: Derived vertical measurements for mean
            sea-level pressure, |P_MSL| [Pa].
        """

        mslp = station_pressure.multiply(
            np.exp(
                (virtual_temperature.multiply(np.abs(self.g)))
                .rdiv(self.r_dry)
                .rdiv(elevation)
            )
        )

        return mslp

    def get_potential_temperature(self, virtual_temperature, pressure):
        """Calculates potential temperature.

        Args:
            virtual_temperature (pd.DataFrame): Vertical measurements,
                virtual temperature, |T_v| [K].
            pressure (pd.DataFrame): Vertical measurements, mean
                sea-level pressure, |P_MSL| [Pa].

        Returns:
            pd.DataFrame: Derived vertical measurements for potential
            temperature, |theta| [K].
        """

        potential_temperature = virtual_temperature.multiply(
            (pressure.rdiv(self.ref_pressure)).pow((self.r_dry / self.cp))
        )

        return potential_temperature
