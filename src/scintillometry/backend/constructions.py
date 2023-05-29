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

Constructs profiles from vertical measurements.

These definitions avoid confusion with interchangeable terms:

- **Elevation**: Vertical distance between mean sea-level and station.
- **Height**: Vertical distance between station elevation and
  measurement point.
- **Altitude**: Vertical distance between mean sea-level and measurement
  point.

Functions are written for performance. More legible versions are
included as inline comments, and are used for tests.
"""

import numpy as np
import pandas as pd

from scintillometry.backend.constants import AtmosConstants


class ProfileConstructor:
    """Constructs vertical profiles.

    Attributes:
        constants (AtmosConstants): Inherited atmospheric constants.
    """

    def __init__(self):
        super().__init__()
        self.constants = AtmosConstants()

    def get_water_vapour_pressure(self, abs_humidity, temperature):
        """Derive water vapour pressure from vertical measurements.

        .. math::

            e = \\mathfrak{{R}}_{{v}} \\cdot \\rho_{{v}} T

        Args:
            abs_humidity (pd.DataFrame): Vertical measurements, absolute
                humidity, |rho_v| [|kgm^-3|].
            temperature (pd.DataFrame): Vertical measurements,
                temperature, |T| [K].

        Returns:
            pd.DataFrame: Derived vertical measurements for water vapour
            pressure, |e| [Pa].
        """

        # abs_humidity * temperature * self.r_vapour
        wvp = abs_humidity.multiply(temperature).multiply(self.constants.r_vapour)

        return wvp

    def get_air_pressure(self, pressure, air_temperature, z_target, z_ref=0):
        """Derive air pressure at specific height.

        Uses hypsometric equation. By default, heights are relative to
        the station elevation, i.e. <z_target> - <z_ref> = <z_target>,
        <z_ref> = 0.

        .. math::

            P_{{z}} = P_{{0}} \\cdot \\exp{{
                \\left (
                \\frac{{-g}}{{\\mathfrak{{R}}_{{d}}}}
                \\frac{{\\Delta z}}{{T_{{z}}}}
                \\right )
                }}

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

        # pressure * np.exp(
        #     (-self.g * (z_target - z_ref)) / (air_temperature * self.r_dry)
        # )
        gdz = -self.constants.g * (z_target - z_ref)
        alt_pressure = pressure.multiply(
            np.exp((air_temperature.multiply(self.constants.r_dry)).rdiv(gdz))
        )

        return alt_pressure

    def extrapolate_column(self, dataframe, gradient):
        """Extrapolates measurements from reference column.

        Applies gradient to the first column of a dataframe to produce
        data for the remaining columns.

        Args:
            dataframe (pd.DataFrame): Numeric data with integer column
                labels.
            gradient (pd.DataFrame or float): Either a gradient or
                dataframe of gradients.

        Returns:
            pd.DataFrame: Extrapolated measurements. The first column
            remains unchanged.
        """

        extrapolated = dataframe.copy(deep=True)

        if isinstance(gradient, pd.DataFrame):
            delta_col = -extrapolated.columns.to_series().diff(periods=-1)
            extrapolated = extrapolated.add(gradient.multiply(delta_col))
        else:
            for col_idx in extrapolated.columns.difference([0]):
                extrapolated[col_idx] = extrapolated[0] + gradient * col_idx

        return extrapolated

    def extrapolate_air_pressure(self, surface_pressure, temperature):
        """Extrapolates reference pressure measurements to scan levels.

        Input series and dataframe do not need matching indices. Data
        for <surface_pressure> (1D time series) is mapped to the index
        and columns of <temperature> (2D time series), i.e. no
        interpolation occurs.

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

        .. math::

            r = \\frac{{\\mathfrak{{R}}_{{d}}}}{{\\mathfrak{{R}}_{{v}}}}
                \\frac{{e}}{{P_{{dry}}}}

        Args:
            wv_pressure (pd.DataFrame): Vertical measurements, water
                vapour pressure, |e| [Pa].
            d_pressure (pd.DataFrame): Vertical measurements, dry air
                pressure, |P_dry| [Pa].

        Returns:
            pd.DataFrame: Derived vertical measurements for mixing
            ratio, |r| [|kgkg^-1|].
        """

        # (wv_pressure * self.r_dry) / (d_pressure * self.r_vapour)
        m_ratio = (wv_pressure.multiply(self.constants.r_dry)).divide(
            d_pressure.multiply(self.constants.r_vapour)
        )

        return m_ratio

    def get_virtual_temperature(self, temperature, mixing_ratio):
        """Derive virtual temperature from vertical measurements.

        .. math::

            T_{{v}} = T(1 + 0.61r)

        Args:
            temperature (pd.DataFrame): Vertical measurements, air
                temperature, |T| [K].
            mixing_ratio (pd.DataFrame): Vertical measurements, mixing
                ratio, |r| [|kgkg^-1|].

        Returns:
            pd.DataFrame: Derived vertical measurements for virtual
            temperature, |T_v| [K].
        """

        # temperature * (1 + 0.61 * mixing_ratio)
        v_temp = temperature.multiply((mixing_ratio.multiply(0.61)).radd(1))

        return v_temp

    def get_reduced_pressure(self, station_pressure, virtual_temperature, elevation):
        """Reduce station pressure to mean sea-level pressure.

        .. math::

            P_{{MSL}} = P_{{z}} \\cdot \\exp \\left (
                \\frac{{|g|}}{{\\mathfrak{{R}}_{{d}}}}
                \\frac{{z_{{stn}}}}{{T_{{v}}}}
            \\right )

        Args:
            station_pressure (pd.DataFrame): Vertical measurements,
                pressure relative to station elevation, |P_z| [Pa].
            virtual_temperature (pd.DataFrame): Vertical measurements,
                virtual temperature, |T_v| [K].
            elevation (float): Station elevation, |z_stn| [m].

        Returns:
            pd.DataFrame: Derived vertical measurements for mean
            sea-level pressure, |P_MSL| [Pa].
        """

        # station_pressure * np.exp(
        #     elevation / (virtual_temperature * (self.r_dry / np.abs(self.g)))
        # )
        alpha = self.constants.r_dry / np.abs(self.constants.g)
        mslp = station_pressure.multiply(
            (np.exp(virtual_temperature.multiply(alpha).rdiv(elevation)))
        )

        return mslp

    def get_potential_temperature(self, temperature, pressure):
        """Calculates potential temperature.

        .. math::

            \\theta = T \\left ( \\frac{{P_{{b}}}}{{P}}
                \\right ) ^{{(\\mathfrak{{R}}_{{d}}/c_{{p}})}}

        Args:
            temperature (pd.DataFrame): Vertical measurements,
                temperature, |T| [K].
            pressure (pd.DataFrame): Vertical measurements, mean
                sea-level pressure, |P_MSL| [Pa].

        Returns:
            pd.DataFrame: Derived vertical measurements for potential
            temperature, |theta| [K].
        """

        # temperature * (self.ref_pressure / pressure) ** (self.r_dry / self.cp)
        factor = self.constants.r_dry / self.constants.cp
        potential_temperature = temperature.multiply(
            (pressure.rdiv(self.constants.ref_pressure)).pow(factor)
        )

        return potential_temperature

    def get_environmental_lapse_rate(self, temperature):
        """Computes environmental lapse rate.

        Lapse rate is inversely proportional to the temperature
        gradient.

        Args:
            temperature (pd.DataFrame): Vertical measurements,
                temperature, |T| [K].

        Returns:
            pd.DataFrame: Derived vertical measurements for
            environmental lapse rate, |Gamma_e| [|Km^-1|]. Values for
            the last column are all NaN.
        """

        delta_z = -temperature.columns.to_series().diff(periods=-1)
        lapse_rate = temperature.diff(periods=-1, axis=1).divide(delta_z)

        return lapse_rate

    def get_moist_adiabatic_lapse_rate(self, mixing_ratio, temperature):
        """Computes moist adiabatic lapse rate.

        Lapse rate is inversely proportional to the temperature
        gradient.

        .. math::

            \\Gamma_{{m}} = g \\frac{{
                \\left ( 1 +
                \\frac{{H_{{v}}r}}
                {{\\mathfrak{{R}}_{{d}}T}} \\right )
            }}
            {{
                \\left ( c_{{p}} +
                \\frac{{H_{{v}}^{{2}}r}}
                {{\\mathfrak{{R}}_{{d}}T^{{2}}}} \\right)
            }}

        Args:
            temperature (pd.DataFrame): Vertical measurements,
                temperature, |T| [K].
            mixing_ratio (pd.DataFrame): Vertical measurements, mixing
                ratio, |r| [|kgkg^-1|].

        Returns:
            pd.DataFrame: Derived vertical measurements for moist
            adiabatic lapse rate, |Gamma_m| [|Km^-1|].
        """

        # 1 + (self.latent_vapour * mixing_ratio) / (self.r_dry * temperature)
        numerator = (
            (mixing_ratio.multiply(self.constants.latent_vapour)).divide(
                (temperature.multiply(self.constants.r_dry))
            )
        ).radd(1)
        # self.cp_dry + (mixing_ratio * (self.latent_vapour**2)) / (
        #     self.r_vapour * (temperature**2)
        # )
        denominator = (
            (
                mixing_ratio.multiply(
                    self.constants.ratio_rmm * self.constants.latent_vapour**2
                )
            ).divide(((temperature.pow(2)).multiply(self.constants.r_dry)))
        ).radd(self.constants.cp_dry)

        lapse_rate = (numerator.divide(denominator)).multiply(self.constants.g)

        return lapse_rate

    def get_lapse_rates(self, temperature, mixing_ratio):
        """Calculate lapse rates.

        Lapse rates are inversely proportional to the temperature
        gradient:

        .. math::

            \\Gamma = -\\frac{{\\delta T}}{{\\delta z}}

        Args:
            temperature (pd.DataFrame): Vertical measurements,
                temperature, |T| [K].
            mixing_ratio (pd.DataFrame): Vertical measurements, mixing
                ratio, |r| [|kgkg^-1|].

        Returns:
            dict[str, pd.DataFrame]: Derived vertical measurements for
            the environmental and moist adiabatic lapse rates, |Gamma_e|
            and |Gamma_m| [|Km^-1|].
        """

        environmental_lapse = self.get_environmental_lapse_rate(temperature=temperature)
        moist_adiabatic_lapse = self.get_moist_adiabatic_lapse_rate(
            mixing_ratio=mixing_ratio,
            temperature=temperature,
        )

        unsaturated_temperature = self.extrapolate_column(
            dataframe=temperature, gradient=-self.constants.dalr
        )
        saturated_temperature = self.extrapolate_column(
            dataframe=temperature, gradient=-moist_adiabatic_lapse
        )
        lapse_rates = {
            "environmental": environmental_lapse,
            "moist_adiabatic": moist_adiabatic_lapse,
            "unsaturated": unsaturated_temperature,
            "saturated": saturated_temperature,
        }

        return lapse_rates

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
                    (\\Delta x_{{i-1}} + \\Delta x_{{i}})
                    + O(\\Delta x_{{i}})^{{2}}
                }}

        Args:
            dataframe (pd.DataFrame): Non-uniformly spaced measurements
                for single variable.

        Returns:
            pd.DataFrame: Derivative of variable with respect to
            non-uniform intervals.
        """

        array = dataframe.copy(deep=True)
        delta_x = dataframe.columns.to_series().diff()
        delta_x.iloc[0] = delta_x.iloc[1]
        derivative = pd.DataFrame(columns=dataframe.columns, index=dataframe.index)

        # Set boundary conditions
        derivative.isetitem(0, 0)  # y_0 = 0
        # (array[array.columns[-1]] - array[array.columns[-2]]) / delta_x.iloc[-1]
        derivative.isetitem(  # y_max(x) from backwards-differencing
            -1,
            (array[array.columns[-1]].subtract(array[array.columns[-2]])).divide(
                delta_x.iloc[-1]
            ),
        )

        # In pseudo-code:
        # for i in np.arange(1, len(array.columns) - 1):
        #     derivative[i] = (
        #         (array[i + 1] * delta_x[i - 1] ** 2)
        #         - (array[i - 1] * delta_x[i] ** 2)
        #         + (array[i] * ((delta_x[i - 1] ** 2) - (delta_x[i] ** 2)))
        #     ) / (delta_x[i - 1] * delta_x[i] * (delta_x[i - 1] + delta_x[i]))

        for i in np.arange(1, len(array.columns) - 1):
            derivative[derivative.columns[i]] = (
                array.iloc[:, i + 1]
                .multiply(delta_x.iloc[i - 1] ** 2)
                .subtract(array.iloc[:, i - 1].multiply(delta_x.iloc[i] ** 2))
                .add(
                    array.iloc[:, i].multiply(
                        ((delta_x.iloc[i - 1] ** 2) - (delta_x.iloc[i] ** 2))
                    )
                )
            ).divide(
                delta_x.iloc[i - 1]
                * delta_x.iloc[i]
                * (delta_x.iloc[i - 1] + delta_x.iloc[i])
            )

        return derivative

    def get_gradient(self, data, method="backward"):
        """Computes spatial gradient of a set of vertical measurements.

        Calculates |dy/dx| at each value of independent variable x for
        each time step t(n): e.g. an input dataframe of temperatures |T|
        for heights |z| with time index t, returns a dataframe of
        :math:`\\partial T/\\partial z` for heights |z| with time
        index t.

        By default, the gradient is calculated using a 1-D
        centred-differencing scheme for non-uniform meshes, since
        vertical measurements are rarely made at uniform intervals.

        Args:
            data (pd.DataFrame): Vertical measurements for a single
                variable.
            method (str): Finite differencing method. Supports "uneven"
                for centred-differencing over a non-uniform mesh, and
                "backward" for backward-differencing. Default
                "backward".

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

    def get_static_stability(self, potential_temperature, scheme="backward"):
        """Determines static stability of atmosphere.

        Args:
            potential_temperature (pd.DataFrame): Contains vertical
                measurements for potential temperature.
            scheme (str): Finite differencing method. Supports "uneven"
                for centred-differencing over a non-uniform mesh, and
                "backward" for backward-differencing. Default
                "backward".

        Returns:
            pd.DataFrame: Derived vertical measurements for static
            stability.
        """

        grad_potential_temperature = self.get_gradient(
            data=potential_temperature, method=scheme
        )

        return grad_potential_temperature

    def get_bulk_richardson(self, potential_temperature, meteo_data):
        """Calculate bulk Richardson number.

        As there are no vertical wind observations available to
        calculate |Delta| |u|, it is assumed mean wind speed vanishes at
        surface. This severely limits its accuracy for representing
        dynamic stability.

        .. math::

            Ri_{{b}} = \\frac{{g}}{{\\bar{{\\theta}}}}
            \\frac{{\\Delta \\theta \\Delta z}}{{(\\Delta u)^{{2}}}}

        Args:
            potential_temperature (pd.DataFrame): Vertical measurements,
                potential temperature |theta| [K].
            meteo_data (pd.DataFrame): Meteorological data.
        """

        heights = potential_temperature.columns[potential_temperature.columns <= 6000]
        delta_theta = potential_temperature[heights[-1]].subtract(
            potential_temperature[heights[0]]
        )

        # delta_theta * delta_z * self.g
        numerator = delta_theta.multiply((heights[-1] - heights[0])).multiply(
            self.constants.g
        )
        # mean_potential_temperature * (wind_speed ** 2)
        denominator = (
            potential_temperature[heights]
            .mean(axis=1, skipna=True)
            .multiply((meteo_data["wind_speed"] ** 2))
        )

        bulk_ri = numerator.divide(denominator)

        return bulk_ri

    def get_vertical_variables(self, vertical_data, meteo_data, station_elevation=None):
        """Derives data from vertical measurements.

        For several days or more of data, the returned dictionary may be
        quite large.

        Args:
            vertical_data (dict): Contains vertical measurements for
                absolute humidity and temperature.
            meteo_data (pd.DataFrame): Meteorological data from surface
                measurements.
            station_elevation (float): Weather station elevation,
                |z_stn| [m]. This is the station taking vertical
                measurements, not the scintillometer. If None, the
                elevation is detected from stored dataframe attributes.

        Returns:
            dict: Derived vertical data for water vapour pressure, air
            pressure, mixing ratio, virtual temperature, mean sea-level
            pressure, and potential temperature.
        """

        if not station_elevation:
            station_elevation = vertical_data["temperature"].attrs["elevation"]

        vertical_data["temperature"] = self.constants.convert_temperature(
            temperature=vertical_data["temperature"], base=True
        )
        meteo_pressure = self.constants.convert_pressure(
            meteo_data["pressure"], base=True
        )
        wvp = self.get_water_vapour_pressure(
            abs_humidity=vertical_data["humidity"],
            temperature=vertical_data["temperature"],
        )
        z_pressure = self.extrapolate_air_pressure(
            surface_pressure=meteo_pressure,
            temperature=vertical_data["temperature"],
        )
        m_ratio = self.get_mixing_ratio(wv_pressure=wvp, d_pressure=z_pressure)
        v_temperature = self.get_virtual_temperature(
            temperature=vertical_data["temperature"], mixing_ratio=m_ratio
        )
        reduced_pressure = self.get_reduced_pressure(
            station_pressure=z_pressure,
            virtual_temperature=v_temperature,
            elevation=station_elevation,
        )
        potential_temperature = self.get_potential_temperature(
            temperature=vertical_data["temperature"], pressure=reduced_pressure
        )
        grad_potential_temperature = self.get_static_stability(
            potential_temperature=potential_temperature,
            scheme="backward",
        )

        lapse_rates = self.get_lapse_rates(
            temperature=vertical_data["temperature"], mixing_ratio=m_ratio
        )

        derived_measurements = {
            "temperature": vertical_data["temperature"],
            "humidity": vertical_data["humidity"],
            "water_vapour_pressure": wvp,
            "air_pressure": z_pressure,
            "mixing_ratio": m_ratio,
            "virtual_temperature": v_temperature,
            "msl_pressure": reduced_pressure,
            "potential_temperature": potential_temperature,
            "grad_potential_temperature": grad_potential_temperature,
            "environmental_lapse_rate": lapse_rates["environmental"],
            "moist_adiabatic_lapse_rate": lapse_rates["moist_adiabatic"],
            "unsaturated_temperature": lapse_rates["unsaturated"],
            "saturated_temperature": lapse_rates["saturated"],
        }

        return derived_measurements

    def get_n_squared(self, potential_temperature, scheme="backward"):
        """Calculates Brunt-Väisälä frequency, squared.

        .. math::

            N^{{2}} = \\frac{{g}}{{\\bar{{\\theta}}}}
            \\frac{{\\Delta \\theta}}{{\\Delta z}}

        Args:
            potential_temperature (pd.DataFrame): Vertical measurements,
                potential temperature |theta| [K].
            scheme (str): Finite differencing method. Supports "uneven"
                for centred-differencing over a non-uniform mesh, and
                "backward" for backward-differencing. Default
                "backward".

        Returns:
            pd.DataFrame: Derived vertical measurements for
            Brunt-Väisälä frequency squared, |N^2| [Hz].
        """

        grad_pot_temperature = self.get_gradient(
            data=potential_temperature, method=scheme
        )

        n_squared = potential_temperature.rdiv(self.constants.g).multiply(
            grad_pot_temperature
        )

        return n_squared
