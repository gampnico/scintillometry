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

        Input series and dataframe must have matching indices.

        Args:
            surface_pressure (pd.Series): Air pressure measurements at
                reference measurement height, |P_0| [Pa].
            temperature (pd.DataFrame): Vertical measurements, air
                temperature at target heights, |T| [K].

        Returns:
            pd.DataFrame: Derived vertical measurements for pressure,
            |P| [Pa].
        """

        air_pressure = pd.DataFrame(
            index=temperature.index, columns=temperature.columns
        )
        air_pressure.isetitem(0, surface_pressure)
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

        alpha = self.r_dry / np.abs(self.g)
        mslp = station_pressure.multiply(
            (np.exp(virtual_temperature.multiply(alpha).rdiv(elevation)))
        )

        return mslp

    def get_potential_temperature(self, temperature, pressure):
        """Calculates potential temperature.

        Args:
            temperature (pd.DataFrame): Vertical measurements,
                temperature, |T| [K].
            pressure (pd.DataFrame): Vertical measurements, mean
                sea-level pressure, |P_MSL| [Pa].

        Returns:
            pd.DataFrame: Derived vertical measurements for potential
            temperature, |theta| [K].
        """

        potential_temperature = temperature.multiply(
            (pressure.rdiv(self.ref_pressure)).pow(self.r_dry / self.cp)
        )

        return potential_temperature

    def non_uniform_differencing(self, dataframe):
        """Computes gradient of data from a non-uniform mesh.

        The gradient is calculated using a 1-D centred-differencing
        scheme for a non-uniform mesh. For the boundary conditions:
        :math:`y_{{(x=0)}} = 0`, and :math:`y_{{(x=\\max(x))}}` is
        calculated by backwards-differencing - i.e. no open boundaries.

        If the size of :math:`\\Delta x_{{i}}` is very different to
        :math:`\\Delta x_{{i\\pm 1}}`, then accuracy deteriorates from
        :math:`O((\\Delta x)^{{2}})` to :math:`O(\\Delta x)`.

        The scheme is as follows:

        .. math::

            \\left.\\frac{{dy}}{{dx}} \\right |_{{0}} = 0 \\\\

        .. math::

            \\left.\\frac{{dy}}{{dx}} \\right |_{{i=\\max(i)}}
                = \\frac{{
                    y_{{i}} - y_{{i-1}}
                }}{{
                    \\Delta x_{{i}}
                }}

        .. math::

            \\left.\\frac{{dy}}{{dx}} \\right |_{{i>0, i <\\max(i)}}
                = \\frac{{
                    y_{{i+1}} (\\Delta x_{{i-1}})^{{2}}
                    - y_{{i-1}} (\\Delta x_{{i}})^{{2}}
                    + y_{{i}} \\left [
                        (\\Delta x_{{i-1}})^{{2}}
                        - (\\Delta x_{{i}})^{{2}}
                        \\right ]
                }}{{
                    (\\Delta x_{{i-1}})
                    (\\Delta x_{{i}})
                    (\\Delta x_{{i-1}} + \\Delta x_{{i-1}})
                    + O(\\Delta x_{{i}})^{{2}}
                }}

        Args:
            dataframe (pd.DataFrame): Non-uniformly spaced measurements for
                single variable.

        Returns:
            pd.DataFrame: Derivative of variable with respect to
            non-uniform intervals.
        """

        array = dataframe.copy(deep=True)
        delta_x = dataframe.columns.to_series().diff()
        delta_x[0] = dataframe.columns[1]
        derivative = pd.DataFrame(columns=dataframe.columns, index=dataframe.index)

        # Set boundary conditions
        derivative.isetitem(0, 0)  # y_0 = 0
        derivative.isetitem(  # y_max(x) from backwards-differencing
            -1, (array[array.columns[-1]] - array[array.columns[-2]]) / delta_x.iloc[-1]
        )

        for i in np.arange(1, len(array.columns) - 1):
            derivative[derivative.columns[i]] = (
                array.iloc[:, i + 1].multiply(delta_x.iloc[i - 1] ** 2)
                - array.iloc[:, i - 1].multiply(delta_x.iloc[i] ** 2)
                + array.iloc[:, i].multiply(
                    ((delta_x.iloc[i - 1] ** 2) - (delta_x.iloc[i] ** 2))
                )
            ).divide(
                delta_x.iloc[i - 1]
                * delta_x.iloc[i]
                * (delta_x.iloc[i - 1] + delta_x.iloc[i])
            )

        return derivative

    def get_gradient(self, data, method="uneven"):
        """Computes spatial gradient of a set of vertical measurements.

        Calculates |dy/dx| at each value of independent variable x for
        each time step t(n): e.g. an input dataframe of temperatures |T|
        for heights |z| with time index t, returns a dataframe of
        :math:`\\partial T/\\partial z` for heights |z| with time
        index t.

        By default the gradient is calculated using a 1-D
        centred-differencing scheme for non-uniform meshes, since
        vertical measurements are rarely made at uniform intervals.

        Args:
            data (pd.DataFrame): Vertical measurements for a single
                variable.
            method (str): Finite differencing method. Supports "uneven"
                for centred-differencing over a non-uniform mesh, and
                "backward" for backward-differencing. Default "uneven".

        Returns:
            pd.DataFrame: Derived spatial gradients |dy/dx| for each
            value x at each time step.

        Raises:
            NotImplementedError: '<method>' is not an implemented
                differencing scheme. Use 'uneven' or 'backward'.
        """

        if method.lower() == "backward":
            gradients = data.diff(periods=1, axis=1) / data.columns.to_series().diff()
            gradients[0] = 0  # set boundary condition
        elif method.lower() == "uneven":
            gradients = self.non_uniform_differencing(dataframe=data)
        else:
            error_msg = (
                f"'{method}' is not an implemented differencing scheme.",
                "Use 'uneven' or 'backward'.",
            )
            raise NotImplementedError(" ".join(error_msg))

        return gradients
