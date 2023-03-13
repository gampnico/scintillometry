"""Copyright 2023 Scintillometry-Tools Contributors.

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

Tests profile construction from vertical measurements.

Methods in ProfileConstructor use pascals and kelvins, whereas fixtures
for meteorological data are in hectopascals and Celsius.

Do not use pandas' vectorisation when applying test formulas to objects
that are compared to the output of a module's function::

    test_foobar = test_foo.multiply(test_bar)
    compare_foobar = self.get_foobar(foo=test_foo, bar=test_bar)

should be written as::

    test_foobar = test_foo * test_bar
    compare_foobar = self.get_foobar(foo=test_foo, bar=test_bar)

The module being tested uses these pandas methods, so duplicating them
in tests makes comparisons pointless.
"""

from typing import Any

import numpy as np
import pandas as pd
import pandas.api.types as ptypes
import pytest

import scintillometry.backend.constants
import scintillometry.backend.constructions


class TestBackendProfileConstructor:
    """Test class for profile construction objects.

    The `check_dataframe` and `setup_extrapolated` functions are
    recommended instead of repetitive boilerplate.

    Attributes:
        test_profile (ProfileConstructor): An initialised
            ProfileConstructor object.
        test_levels (list): Mocked measurement heights.
        test_index (pd.DatetimeIndex): Mocked TZ-naive datetime index
            for dataframe.
        test_elevation (float): Mocked station elevation.
    """

    test_profile = scintillometry.backend.constructions.ProfileConstructor()
    test_levels = [0, 10, 30, 50, 75, 100]
    test_index = pd.to_datetime(
        ["2020-06-03 03:10:00", "2020-06-03 03:20:00", "2020-06-03 03:30:00"], utc=False
    )
    test_elevation = 600.0

    def check_dataframe(self, dataframe: Any):
        """Boilerplate tests for dataframes.

        Checks input is a dataframe, all columns have numeric dtypes,
        index is a DatetimeIndex, and there are no NaN or infinite
        values.
        """

        assert isinstance(dataframe, pd.DataFrame)
        assert all(ptypes.is_numeric_dtype(dataframe[i]) for i in dataframe.columns)
        assert ptypes.is_datetime64_any_dtype(dataframe.index)
        assert not dataframe.isnull().values.any()
        assert not np.isinf(dataframe).values.any()

    @pytest.mark.dependency(
        name="TestBackendProfileConstructor::test_check_dataframe_error"
    )
    @pytest.mark.parametrize(
        "arg_test", ["type", "ptype", "numeric", "null", "inf", "index"]
    )
    def test_check_dataframe_error(self, arg_test):
        """Raise AssertionError if dataframe fails check_dataframe.

        Check AssertionError is raised if input is not a dataframe, any
        column does not have a numeric dtype, the index is not a
        DatetimeIndex, or there are any NaN or infinite values.
        """

        test_data = {0: [1, 2, 3], 10: [4.0, 5.0, 6.0], 20: [7, 8.0, -9]}
        test_frame = pd.DataFrame(
            data=test_data, columns=[0, 10, 20], index=self.test_index
        )

        if arg_test == "ptype":
            test_frame[test_frame.columns[0]] = "error"
            assert isinstance(test_frame, pd.DataFrame)
        elif arg_test == "numeric":
            test_frame[test_frame.columns[0]] = "error"
            assert isinstance(test_frame, pd.DataFrame)
        elif arg_test == "null":
            test_frame[test_frame.columns[0]] = np.nan
        elif arg_test == "inf":
            test_frame[test_frame.columns[0]] = np.inf
        elif arg_test == "index":
            test_frame = test_frame.reset_index()
        else:
            test_frame = test_frame[test_frame.columns[0]]

        with pytest.raises(AssertionError):
            self.check_dataframe(dataframe=test_frame)

    @pytest.mark.dependency(
        name="TestBackendProfileConstructor::test_check_dataframe",
        depends=["TestBackendProfileConstructor::test_check_dataframe_error"],
    )
    def test_check_dataframe(self):
        """Check dataframe can pass boilerplate tests."""

        test_data = {0: [1, 2, 3], 10: [4.0, 5.0, 6.0], 20: [7, 8.0, -9]}
        test_frame = pd.DataFrame(
            data=test_data, columns=[0, 10, 20], index=self.test_index
        )
        self.check_dataframe(dataframe=test_frame)

    def setup_extrapolated(self, series: pd.Series, levels: list):
        """Extrapolate series to mock vertical measurements.

        Args:
            series: Mocked surface measurements.
            levels: Mocked vertical measurement intervals.

        Returns:
            pd.DataFrame: Mocked vertical measurements.
        """

        reference = series.copy(deep=True)
        values = {}
        for i in levels:  # for tests, extrapolation can be very simple
            values[i] = reference.multiply(1 / (i + 1))
        dataset = pd.DataFrame(
            data=values,
            columns=levels,
            index=reference.index,
        )
        dataset.attrs["elevation"] = self.test_elevation

        self.check_dataframe(dataframe=dataset)
        assert "elevation" in dataset.attrs
        assert np.isclose(dataset.attrs["elevation"], self.test_elevation)

        return dataset

    @pytest.mark.dependency(
        name="TestBackendProfileConstructor::setup_extrapolated",
        depends=["TestBackendProfileConstructor::test_check_dataframe"],
    )
    def test_setup_extrapolated(
        self, conftest_mock_weather_dataframe_tz, conftest_mock_hatpro_scan_levels
    ):
        """Extrapolate series to dataframe."""

        test_series = conftest_mock_weather_dataframe_tz["pressure"].copy(deep=True)
        compare_extrapolate = self.setup_extrapolated(
            series=test_series, levels=conftest_mock_hatpro_scan_levels
        )
        self.check_dataframe(dataframe=compare_extrapolate)
        assert compare_extrapolate.index.equals(test_series.index)
        assert "elevation" in compare_extrapolate.attrs
        assert np.isclose(compare_extrapolate.attrs["elevation"], self.test_elevation)

    @pytest.mark.dependency(
        name="TestBackendProfileConstructor::test_constructor_init",
        depends=["TestBackendConstants::test_constants_init"],
        scope="session",
    )
    def test_profile_constructor_init(self):
        """Initialise constants as class attributes."""

        test_profile = scintillometry.backend.constructions.ProfileConstructor()
        assert test_profile.constants
        assert isinstance(
            test_profile.constants, scintillometry.backend.constants.AtmosConstants
        )

    @pytest.mark.dependency(
        name="TestBackendProfileConstructor::test_get_water_vapour_pressure",
        depends=["TestBackendProfileConstructor::test_constructor_init"],
    )
    def test_get_water_vapour_pressure(
        self,
        conftest_mock_hatpro_humidity_dataframe,
        conftest_mock_hatpro_temperature_dataframe,
        conftest_mock_hatpro_scan_levels,
    ):
        """Calculate water vapour pressure."""

        assert isinstance(self.test_profile.constants.r_vapour, (float, int))
        test_wvp = (
            conftest_mock_hatpro_humidity_dataframe
            * conftest_mock_hatpro_temperature_dataframe
            * self.test_profile.constants.r_vapour
        )
        compare_wvp = self.test_profile.get_water_vapour_pressure(
            abs_humidity=conftest_mock_hatpro_humidity_dataframe,
            temperature=conftest_mock_hatpro_temperature_dataframe,
        )

        self.check_dataframe(dataframe=compare_wvp)
        assert all(
            key in compare_wvp.columns for key in conftest_mock_hatpro_scan_levels
        )
        assert compare_wvp.index.name == "rawdate"
        assert np.allclose(compare_wvp, test_wvp)

    @pytest.mark.dependency(
        name="TestBackendProfileConstructor::test_get_air_pressure",
        depends=["TestBackendProfileConstructor::test_constructor_init"],
    )
    def test_get_air_pressure(
        self,
        conftest_mock_weather_dataframe_tz,
        conftest_mock_hatpro_temperature_dataframe_tz,
        conftest_mock_hatpro_scan_levels,
    ):
        """Calculate air pressure at specific height."""

        test_temperature = conftest_mock_hatpro_temperature_dataframe_tz.copy(deep=True)
        test_idx = conftest_mock_hatpro_scan_levels[-1]
        ref_pressure = conftest_mock_weather_dataframe_tz["pressure"].multiply(100)

        test_pressure = ref_pressure * np.exp(
            -self.test_profile.constants.g
            * (test_idx - 0)
            / (self.test_profile.constants.r_dry * test_temperature[test_idx])
        )

        compare_pressure = self.test_profile.get_air_pressure(
            pressure=ref_pressure,
            air_temperature=test_temperature[test_idx],
            z_target=test_idx,
            z_ref=0,
        )
        assert isinstance(compare_pressure, pd.Series)
        assert compare_pressure.index.equals(test_temperature.index)
        assert not np.allclose(ref_pressure, compare_pressure)
        assert (ref_pressure > compare_pressure).all()
        assert (compare_pressure > 1000).all()
        assert not compare_pressure.isnull().values.any()
        assert not np.isinf(compare_pressure).values.any()
        assert np.allclose(test_pressure, compare_pressure)

    @pytest.mark.dependency(
        name="TestBackendProfileConstructor::test_extrapolate_air_pressure",
        depends=[
            "TestBackendProfileConstructor::test_constructor_init",
            "TestBackendProfileConstructor::test_get_air_pressure",
        ],
    )
    def test_extrapolate_air_pressure(
        self,
        conftest_mock_weather_dataframe_tz,
        conftest_mock_hatpro_temperature_dataframe_tz,
    ):
        """Extrapolate air pressure across all scan heights."""

        test_temperature = conftest_mock_hatpro_temperature_dataframe_tz.copy(deep=True)
        test_pressure = conftest_mock_weather_dataframe_tz["pressure"].copy(deep=True)
        test_pressure = test_pressure.multiply(100)  # hPa -> Pa

        compare_pressure = self.test_profile.extrapolate_air_pressure(
            surface_pressure=test_pressure, temperature=test_temperature
        )

        self.check_dataframe(dataframe=compare_pressure)
        assert compare_pressure.index.equals(test_temperature.index)
        assert np.allclose(compare_pressure[compare_pressure.columns[0]], test_pressure)
        for col in compare_pressure.columns.difference([0]):
            assert not np.allclose(test_pressure, compare_pressure[col])
        assert ((compare_pressure.ge(1000)).all()).all()
        assert not np.allclose(
            compare_pressure[compare_pressure.columns[-1]],
            compare_pressure[compare_pressure.columns[0]],
        )

    @pytest.mark.dependency(
        name="TestBackendProfileConstructor::test_get_mixing_ratio",
        depends=[
            "TestBackendProfileConstructor::test_constructor_init",
            "TestBackendProfileConstructor::setup_extrapolated",
        ],
    )
    def test_get_mixing_ratio(
        self,
        conftest_mock_weather_dataframe_tz,
        conftest_mock_hatpro_humidity_dataframe_tz,
        conftest_mock_hatpro_temperature_dataframe_tz,
        conftest_mock_hatpro_scan_levels,
    ):
        """Calculate mixing ratio for dry air pressure."""

        test_weather = conftest_mock_weather_dataframe_tz.copy(deep=True)
        test_pressure = self.setup_extrapolated(
            series=test_weather["pressure"].multiply(100),
            levels=conftest_mock_hatpro_scan_levels,
        )
        test_wvp = conftest_mock_hatpro_humidity_dataframe_tz.multiply(
            conftest_mock_hatpro_temperature_dataframe_tz
        ).multiply(self.test_profile.constants.r_vapour)
        test_ratio = (test_wvp * self.test_profile.constants.r_dry) / (
            self.test_profile.constants.r_vapour * test_pressure
        )

        compare_ratio = self.test_profile.get_mixing_ratio(
            wv_pressure=test_wvp, d_pressure=test_pressure
        )

        self.check_dataframe(dataframe=compare_ratio)
        assert compare_ratio.index.equals(test_wvp.index)
        assert np.allclose(compare_ratio, test_ratio)

    @pytest.mark.dependency(
        name="TestBackendProfileConstructor::test_get_virtual_temperature"
    )
    def test_get_virtual_temperature(
        self,
        conftest_mock_weather_dataframe_tz,
        conftest_mock_hatpro_temperature_dataframe_tz,
        conftest_mock_hatpro_scan_levels,
    ):
        """Calculate virtual temperature."""

        ref_temperature = conftest_mock_hatpro_temperature_dataframe_tz.copy(deep=True)
        test_pressure = (
            conftest_mock_weather_dataframe_tz["pressure"].copy(deep=True).multiply(100)
        )

        test_ratio = self.setup_extrapolated(
            series=test_pressure.divide(test_pressure + 1),
            levels=conftest_mock_hatpro_scan_levels,
        )
        test_temperature = ref_temperature * (1 + (0.61 * test_ratio))

        compare_temperature = self.test_profile.get_virtual_temperature(
            temperature=ref_temperature, mixing_ratio=test_ratio
        )
        self.check_dataframe(dataframe=compare_temperature)
        assert compare_temperature.index.equals(test_temperature.index)
        assert not np.allclose(compare_temperature, ref_temperature)
        assert np.allclose(compare_temperature, test_temperature)

    @pytest.mark.dependency(
        name="TestBackendProfileConstructor::test_get_reduced_pressure",
        depends=["TestBackendProfileConstructor::test_constructor_init"],
    )
    def test_get_reduced_pressure(
        self, conftest_mock_weather_dataframe_tz, conftest_mock_hatpro_scan_levels
    ):
        """Reduce station pressure to mean sea-level pressure."""

        test_weather = conftest_mock_weather_dataframe_tz.copy(deep=True)
        test_pressure = self.setup_extrapolated(
            series=test_weather["pressure"],
            levels=conftest_mock_hatpro_scan_levels,
        )
        test_temperature = self.setup_extrapolated(
            series=test_weather["temperature_2m"],
            levels=conftest_mock_hatpro_scan_levels,
        )
        assert isinstance(test_temperature, pd.DataFrame)
        alpha = self.test_profile.constants.r_dry / np.abs(
            self.test_profile.constants.g
        )
        assert np.isclose(alpha, 29.26)
        test_factor = self.test_elevation / (alpha * test_temperature)
        test_exp = np.exp(test_factor)
        self.check_dataframe(dataframe=test_exp)
        assert np.allclose(test_exp, np.e**test_factor)
        test_mslp = test_pressure * test_exp
        assert isinstance(test_mslp, pd.DataFrame)

        compare_mslp = self.test_profile.get_reduced_pressure(
            station_pressure=test_pressure,
            virtual_temperature=test_temperature,
            elevation=self.test_elevation,
        )

        self.check_dataframe(dataframe=compare_mslp)
        assert np.allclose(compare_mslp, test_mslp)

    @pytest.mark.dependency(
        name="TestBackendProfileConstructor::test_get_potential_temperature",
        depends=[
            "TestBackendProfileConstructor::test_constructor_init",
            "TestBackendProfileConstructor::setup_extrapolated",
        ],
    )
    def test_get_potential_temperature(
        self,
        conftest_mock_hatpro_scan_levels,
        conftest_mock_weather_dataframe_tz,
        conftest_mock_hatpro_temperature_dataframe_tz,
    ):
        """Calculate potential temperature."""

        test_weather = conftest_mock_weather_dataframe_tz.copy(deep=True)
        test_pressure = self.setup_extrapolated(
            series=test_weather["pressure"].multiply(100),
            levels=conftest_mock_hatpro_scan_levels,
        )
        test_temperature = conftest_mock_hatpro_temperature_dataframe_tz.copy(deep=True)
        test_potential = test_temperature * (
            self.test_profile.constants.ref_pressure / test_pressure
        ) ** (self.test_profile.constants.r_dry / self.test_profile.constants.cp)
        for frame in (test_temperature, test_pressure):
            assert isinstance(frame, pd.DataFrame)
            assert frame.index.equals(test_weather.index)
            assert all(i in conftest_mock_hatpro_scan_levels for i in frame.columns)
            self.check_dataframe(dataframe=frame)
            assert frame.gt(0).values.all()

        compare_potential = self.test_profile.get_potential_temperature(
            temperature=test_temperature, pressure=test_pressure
        )
        self.check_dataframe(dataframe=compare_potential)
        assert compare_potential.gt(0).values.all()

        assert np.allclose(compare_potential, test_potential)

    @pytest.mark.dependency(
        name="TestBackendProfileConstructor::test_non_uniform_differencing"
    )
    def test_non_uniform_differencing(
        self,
        conftest_mock_hatpro_temperature_dataframe_tz,
        conftest_mock_hatpro_scan_levels,
    ):
        """Compute centred-differencing scheme for non-uniform data."""

        test_dataframe = conftest_mock_hatpro_temperature_dataframe_tz.copy(deep=True)
        test_cols = test_dataframe.columns
        test_diff = test_cols.to_series().diff()
        assert isinstance(test_diff, pd.Series)
        assert np.isnan(test_diff[0])
        assert not test_diff[1:].isnull().all()

        test_gradient = test_dataframe.copy(deep=True)
        for i in range(1, len(test_cols)):
            test_gradient.iloc[:, i] = (
                test_dataframe.iloc[:, i] - test_dataframe.iloc[:, i - 1]
            ) / test_diff.iloc[i]
        test_gradient.isetitem(0, 0)
        test_gradient.isetitem(
            -1,
            (test_dataframe[test_cols[-1]] - test_dataframe[test_cols[-2]])
            / test_diff.iloc[-1],
        )

        compare_gradient = self.test_profile.non_uniform_differencing(
            dataframe=test_dataframe
        )

        self.check_dataframe(dataframe=compare_gradient)
        assert all(
            key in compare_gradient.columns for key in conftest_mock_hatpro_scan_levels
        )
        assert compare_gradient.index.name == "rawdate"

        # Test boundary conditions
        assert (compare_gradient[compare_gradient.columns[0]] == 0).all()
        assert np.allclose(
            compare_gradient[compare_gradient.columns[-1]],
            test_gradient[test_gradient.columns[-1]],
        )
        assert all(np.sign(compare_gradient)) == all(np.sign(test_gradient))

    @pytest.mark.dependency(
        name="TestBackendProfileConstructor::test_get_gradient_error"
    )
    def test_get_gradient_error(self, conftest_mock_hatpro_temperature_dataframe_tz):
        """Raise error for missing finite-differencing scheme."""

        test_error = conftest_mock_hatpro_temperature_dataframe_tz.copy(deep=True)
        error_msg = (
            "'missing scheme' is not an implemented differencing scheme.",
            "Use 'uneven' or 'backward'.",
        )
        with pytest.raises(NotImplementedError, match=" ".join(error_msg)):
            self.test_profile.get_gradient(data=test_error, method="missing scheme")

    @pytest.mark.dependency(
        name="TestBackendProfileConstructor::test_get_gradient",
        depends=[
            "TestBackendProfileConstructor::test_get_gradient_error",
            "TestBackendProfileConstructor::test_non_uniform_differencing",
        ],
    )
    @pytest.mark.parametrize("arg_method", ["backward", "uneven"])
    def test_get_gradient(
        self,
        conftest_mock_hatpro_temperature_dataframe_tz,
        conftest_mock_hatpro_scan_levels,
        arg_method,
    ):
        """Calculate spatial gradient."""

        test_dataframe = conftest_mock_hatpro_temperature_dataframe_tz.copy(deep=True)
        test_cols = test_dataframe.columns
        test_diff = test_cols.to_series().diff()
        assert isinstance(test_diff, pd.Series)
        assert np.isnan(test_diff[0])
        assert not test_diff[1:].isnull().all()

        test_gradient = test_dataframe.copy(deep=True)
        for i in range(1, len(test_cols)):
            test_gradient.iloc[:, i] = (
                test_dataframe.iloc[:, i] - test_dataframe.iloc[:, i - 1]
            ) / test_diff.iloc[i]
        test_gradient.isetitem(0, 0)

        compare_gradient = self.test_profile.get_gradient(
            data=test_dataframe, method=arg_method
        )

        self.check_dataframe(dataframe=compare_gradient)
        assert all(
            key in compare_gradient.columns for key in conftest_mock_hatpro_scan_levels
        )
        assert compare_gradient.index.name == "rawdate"
        assert (compare_gradient[test_cols[0]] == 0).all()

        if arg_method == "uneven":
            test_gradient.isetitem(
                -1,
                (test_dataframe[test_cols[-1]] - test_dataframe[test_cols[-2]])
                / test_diff.iloc[-1],
            )
            assert np.allclose(
                compare_gradient[test_cols[-1]], test_gradient[test_cols[-1]]
            )
        else:
            assert np.allclose(compare_gradient, test_gradient)

    @pytest.mark.dependency(
        name="TestBackendProfileConstructor::test_get_static_stability",
        depends=[
            "TestBackendProfileConstructor::test_constructor_init",
            "TestBackendProfileConstructor::test_get_water_vapour_pressure",
            "TestBackendProfileConstructor::test_extrapolate_air_pressure",
            "TestBackendProfileConstructor::test_get_mixing_ratio",
            "TestBackendProfileConstructor::test_get_reduced_pressure",
            "TestBackendProfileConstructor::test_get_virtual_temperature",
            "TestBackendProfileConstructor::test_get_potential_temperature",
            "TestBackendProfileConstructor::test_get_gradient",
        ],
    )
    @pytest.mark.parametrize("arg_method", ["backward", "uneven"])
    def test_get_static_stability(
        self,
        conftest_mock_hatpro_temperature_dataframe_tz,
        conftest_mock_hatpro_humidity_dataframe_tz,
        conftest_mock_weather_dataframe_tz,
        conftest_mock_hatpro_scan_levels,
        arg_method,
    ):
        """Calculate static stability."""

        test_hatpro = {
            "temperature": conftest_mock_hatpro_temperature_dataframe_tz.copy(
                deep=True
            ),
            "humidity": conftest_mock_hatpro_humidity_dataframe_tz.copy(deep=True),
        }
        test_weather = conftest_mock_weather_dataframe_tz.copy(deep=True)
        assert isinstance(test_hatpro, dict)
        for frame in test_hatpro.values():
            self.check_dataframe(dataframe=frame)
            assert frame.index.equals(test_weather.index)

        compare_stability = self.test_profile.get_static_stability(
            vertical_data=test_hatpro,
            meteo_data=test_weather,
            station_elevation=self.test_elevation,
            scheme=arg_method,
        )

        self.check_dataframe(dataframe=compare_stability)
        assert all(
            key in compare_stability.columns for key in conftest_mock_hatpro_scan_levels
        )
        assert np.allclose(compare_stability[compare_stability.columns[0]], 0)

    @pytest.mark.dependency(
        name="TestBackendProfileConstructor::test_get_bulk_richardson",
        depends=["TestBackendProfileConstructor::test_constructor_init"],
    )
    # @pytest.mark.parametrize("arg_method", ["backward", "uneven"])
    def test_get_bulk_richardson(
        self,
        conftest_mock_hatpro_temperature_dataframe_tz,
        conftest_mock_weather_dataframe_tz,
    ):
        """Calculate bulk Richardson number."""

        test_weather = conftest_mock_weather_dataframe_tz.copy(deep=True)
        test_temperature = conftest_mock_hatpro_temperature_dataframe_tz.copy(deep=True)

        cols = test_temperature.columns
        delta_theta = test_temperature[cols[-1]] - test_temperature[cols[0]]
        numerator = self.test_profile.constants.g * delta_theta * (cols[-1] - cols[0])
        denominator = test_temperature.mean(axis=1, skipna=True) * (
            test_weather["wind_speed"] ** 2
        )
        test_bulk = numerator / denominator

        compare_bulk = self.test_profile.get_bulk_richardson(
            potential_temperature=test_temperature, meteo_data=test_weather
        )

        assert isinstance(compare_bulk, pd.Series)
        assert compare_bulk.index.equals(test_temperature.index)
        assert not compare_bulk.isnull().values.any()
        assert not np.isinf(compare_bulk).values.any()
        assert np.allclose(compare_bulk, test_bulk)
