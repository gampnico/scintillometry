"""Copyright 2023 Nicolas Gampierakis.

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
"""

import numpy as np
import pandas as pd
import pandas.api.types as ptypes
import pytest

import scintillometry.backend.constants
import scintillometry.backend.constructions


class TestBackendProfileConstructor:
    """Test class for profile construction objects.

    Attributes:
        TestProfile (ProfileConstructor): An initialised
            ProfileConstructor object.
        test_levels (list): Test measurement heights.
        test_index (pd.DatetimeIndex): TZ-naive datetime index for
            dataframe.
        test_constants (AtmosConstants): Various atmospheric constants.
    """

    TestProfile = scintillometry.backend.constructions.ProfileConstructor()
    test_levels = [0, 10, 30, 50, 75, 100]
    test_index = pd.to_datetime(
        ["2020-06-03 03:10:00", "2020-06-03 03:20:00", "2020-06-03 03:30:00"], utc=False
    )
    test_constants = scintillometry.backend.constants.AtmosConstants()

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

        assert isinstance(self.TestProfile.constants.r_vapour, (float, int))
        test_wvp = (
            conftest_mock_hatpro_humidity_dataframe
            * conftest_mock_hatpro_temperature_dataframe
            * self.TestProfile.constants.r_vapour
        )
        compare_wvp = self.TestProfile.get_water_vapour_pressure(
            abs_humidity=conftest_mock_hatpro_humidity_dataframe,
            temperature=conftest_mock_hatpro_temperature_dataframe,
        )

        assert isinstance(compare_wvp, pd.DataFrame)
        for key in conftest_mock_hatpro_scan_levels:
            assert key in compare_wvp.columns
            assert ptypes.is_numeric_dtype(compare_wvp[key])
        assert ptypes.is_datetime64_any_dtype(compare_wvp.index)
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
        test_pressure = conftest_mock_weather_dataframe_tz["pressure"].multiply(100)

        compare_pressure = self.TestProfile.get_air_pressure(
            pressure=test_pressure,
            air_temperature=test_temperature[test_idx],
            z_target=test_idx,
            z_ref=0,
        )
        assert isinstance(compare_pressure, pd.Series)
        assert compare_pressure.index.equals(test_temperature.index)
        assert not np.allclose(test_pressure, compare_pressure)
        assert (test_pressure > compare_pressure).all()
        assert (compare_pressure > 1000).all()

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

        compare_pressure = self.TestProfile.extrapolate_air_pressure(
            surface_pressure=test_pressure, temperature=test_temperature
        )

        assert isinstance(compare_pressure, pd.DataFrame)
        assert compare_pressure.index.equals(test_temperature.index)
        assert np.allclose(compare_pressure[compare_pressure.columns[0]], test_pressure)
        for col in compare_pressure.columns.difference([0]):
            assert not np.allclose(test_pressure, compare_pressure[col])
        assert not (test_pressure.isna()).any()
        assert ((compare_pressure.ge(1000)).all()).all()
        assert not np.allclose(
            compare_pressure[compare_pressure.columns[-1]],
            compare_pressure[compare_pressure.columns[0]],
        )

    @pytest.mark.dependency(
        name="TestBackendProfileConstructor::test_get_mixing_ratio",
        depends=[
            "TestBackendProfileConstructor::test_constructor_init",
        ],
    )
    def test_get_mixing_ratio(
        self,
        conftest_mock_weather_dataframe_tz,
        conftest_mock_hatpro_humidity_dataframe_tz,
        conftest_mock_hatpro_temperature_dataframe_tz,
    ):
        """Calculate mixing ratio for dry air pressure."""

        test_wvp = conftest_mock_hatpro_humidity_dataframe_tz.multiply(
            conftest_mock_hatpro_temperature_dataframe_tz
        ).multiply(self.TestProfile.constants.r_vapour)
        test_pressure = (
            conftest_mock_weather_dataframe_tz["pressure"].copy(deep=True).multiply(100)
        )
        test_data = {}
        for level in test_wvp.columns:
            test_data[level] = test_pressure * (1 / (level + 1))
        test_pressure = pd.DataFrame(
            data=test_data, columns=test_wvp.columns, index=test_wvp.index
        )
        test_ratio = (self.TestProfile.constants.r_dry * test_wvp) / (
            self.TestProfile.constants.r_vapour * test_pressure
        )

        compare_ratio = self.TestProfile.get_mixing_ratio(
            wv_pressure=test_wvp, d_pressure=test_pressure
        )

        assert isinstance(compare_ratio, pd.DataFrame)
        assert compare_ratio.index.equals(test_wvp.index)
        assert np.allclose(compare_ratio, test_ratio)

    @pytest.mark.dependency(
        name="TestBackendProfileConstructor::test_get_virtual_temperature"
    )
    def test_get_virtual_temperature(
        self,
        conftest_mock_weather_dataframe_tz,
        conftest_mock_hatpro_temperature_dataframe_tz,
    ):
        """Calculate virtual temperature."""

        test_temperature = conftest_mock_hatpro_temperature_dataframe_tz.copy(deep=True)
        test_pressure = (
            conftest_mock_weather_dataframe_tz["pressure"].copy(deep=True).multiply(100)
        )
        test_data = {}
        for level in test_temperature.columns:  # placeholder values for mixing ratio
            test_data[level] = test_pressure / (test_pressure * (level + 1))
        test_ratio = pd.DataFrame(
            data=test_data,
            columns=test_temperature.columns,
            index=test_temperature.index,
        )

        compare_temperature = self.TestProfile.get_virtual_temperature(
            temperature=test_temperature, mixing_ratio=test_ratio
        )
        assert isinstance(compare_temperature, pd.DataFrame)
        assert compare_temperature.index.equals(test_temperature.index)
        assert not np.allclose(compare_temperature, test_temperature)

    @pytest.mark.dependency(
        name="TestBackendProfileConstructor::test_get_reduced_pressure",
        depends=["TestBackendProfileConstructor::test_constructor_init"],
    )
    def test_get_reduced_pressure(self, conftest_mock_weather_dataframe_tz):
        """Reduce station pressure to mean sea-level pressure."""

        test_pressure = pd.DataFrame(
            data={10: conftest_mock_weather_dataframe_tz["pressure"].copy(deep=True)},
            columns=[10],
            index=conftest_mock_weather_dataframe_tz.index,
        )
        test_temperature = pd.DataFrame(
            data={
                10: conftest_mock_weather_dataframe_tz["temperature_2m"].copy(deep=True)
            },
            columns=[10],
            index=conftest_mock_weather_dataframe_tz.index,
        )
        assert isinstance(test_temperature, pd.DataFrame)
        test_elevation = 600
        test_station = test_elevation / (
            self.TestProfile.constants.r_dry
            / (test_temperature * np.abs(self.TestProfile.constants.g))
        )
        test_exp = np.exp(test_station)
        assert isinstance(test_exp, pd.DataFrame)
        assert np.allclose(test_exp, np.e**test_station)
        test_mslp = test_pressure * test_exp
        assert isinstance(test_mslp, pd.DataFrame)

        compare_mslp = self.TestProfile.get_reduced_pressure(
            station_pressure=test_pressure,
            virtual_temperature=test_temperature,
            elevation=test_elevation,
        )
        assert isinstance(compare_mslp, pd.DataFrame)
        assert np.allclose(compare_mslp, test_mslp)

    @pytest.mark.dependency(
        name="TestBackendProfileConstructor::test_get_potential_temperature",
        depends=["TestBackendProfileConstructor::test_constructor_init"],
    )
    def test_get_potential_temperature(self, conftest_mock_weather_dataframe_tz):
        """Calculate potential temperature."""

        test_pressure = pd.DataFrame(
            data={10: conftest_mock_weather_dataframe_tz["pressure"].copy(deep=True)},
            columns=[10],
            index=conftest_mock_weather_dataframe_tz.index,
        )
        test_temperature = pd.DataFrame(
            data={
                10: conftest_mock_weather_dataframe_tz["temperature_2m"].copy(deep=True)
            },
            columns=[10],
            index=conftest_mock_weather_dataframe_tz.index,
        )
        assert isinstance(test_temperature, pd.DataFrame)
        test_potential = test_temperature * (
            self.TestProfile.constants.ref_pressure / test_pressure
        ) ** (self.TestProfile.constants.r_dry / self.TestProfile.constants.cp)
        assert isinstance(test_potential, pd.DataFrame)

        compare_potential = self.TestProfile.get_potential_temperature(
            virtual_temperature=test_temperature, pressure=test_pressure
        )
        assert isinstance(compare_potential, pd.DataFrame)
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

        compare_gradient = self.TestProfile.non_uniform_differencing(
            dataframe=test_dataframe
        )
        assert isinstance(compare_gradient, pd.DataFrame)
        for key in conftest_mock_hatpro_scan_levels:
            assert key in compare_gradient.columns
            assert ptypes.is_numeric_dtype(compare_gradient[key])
        assert ptypes.is_datetime64_any_dtype(compare_gradient.index)
        assert compare_gradient.index.name == "rawdate"

        # Test boundary conditions
        assert (compare_gradient[compare_gradient.columns[0]] == 0).all()
        assert np.allclose(
            compare_gradient[compare_gradient.columns[-1]],
            test_gradient[test_gradient.columns[-1]],
        )

        assert not all(
            compare_gradient[compare_gradient.columns.difference([0])].isnull().any()
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
            self.TestProfile.get_gradient(data=test_error, method="missing scheme")

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

        compare_gradient = self.TestProfile.get_gradient(
            data=test_dataframe, method=arg_method
        )
        assert isinstance(compare_gradient, pd.DataFrame)
        for key in conftest_mock_hatpro_scan_levels:
            assert key in compare_gradient.columns
            assert ptypes.is_numeric_dtype(compare_gradient[key])
        assert ptypes.is_datetime64_any_dtype(compare_gradient.index)
        assert compare_gradient.index.name == "rawdate"

        assert not all(
            compare_gradient[compare_gradient.columns.difference([0])].isnull().any()
        )
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
        conftest_mock_hatpro_scan_levels,
        arg_method,
    ):
        """Calculate spatial gradient."""

        assert True
