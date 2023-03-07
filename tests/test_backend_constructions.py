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
        test_class (ProfileConstructor): An initialised
            ProfileConstructor object.
        test_levels (list): Test measurement heights.
        test_index (pd.DatetimeIndex): TZ-naive datetime index for
            dataframe.
        test_constants (AtmosConstants): Various atmospheric constants.
    """

    test_class = scintillometry.backend.constructions.ProfileConstructor()
    test_levels = [0, 10, 30]
    test_index = pd.to_datetime(
        ["2020-06-03 03:10:00", "2020-06-03 03:20:00", "2020-06-03 03:30:00"], utc=False
    )
    test_constants = scintillometry.backend.constants.AtmosConstants()

    @pytest.mark.dependency(
        name="TestBackendConstructProfile::test_constructor_init",
        depends=["TestBackendConstants::test_constants_init"],
        scope="session",
    )
    def test_profile_constructor_init(self):
        test_profile = scintillometry.backend.constructions.ProfileConstructor()
        assert test_profile.constants
        assert isinstance(
            test_profile.constants, scintillometry.backend.constants.AtmosConstants
        )

    @pytest.mark.dependency(
        name="TestBackendConstructProfile::test_get_water_vapour_pressure",
        depends=["TestBackendConstructProfile::test_constructor_init"],
    )
    def test_get_water_vapour_pressure(
        self,
        conftest_mock_hatpro_humidity_dataframe,
        conftest_mock_hatpro_temperature_dataframe,
    ):
        """Calculate water vapour pressure."""

        assert isinstance(self.test_class.constants.r_vapour, (float, int))
        test_wvp = (
            conftest_mock_hatpro_humidity_dataframe
            * conftest_mock_hatpro_temperature_dataframe
            * self.test_class.constants.r_vapour
        )
        compare_wvp = self.test_class.get_water_vapour_pressure(
            abs_humidity=conftest_mock_hatpro_humidity_dataframe,
            temperature=conftest_mock_hatpro_temperature_dataframe,
        )

        assert isinstance(compare_wvp, pd.DataFrame)
        for key in self.test_levels:
            assert key in compare_wvp.columns
            assert ptypes.is_numeric_dtype(compare_wvp[key])
        assert ptypes.is_datetime64_any_dtype(compare_wvp.index)
        assert compare_wvp.index.name == "rawdate"

        assert np.allclose(compare_wvp, test_wvp)

    @pytest.mark.dependency(
        name="TestBackendConstructProfile::test_get_air_pressure",
        depends=["TestBackendConstructProfile::test_constructor_init"],
    )
    def test_get_air_pressure(
        self,
        conftest_mock_weather_dataframe_tz,
        conftest_mock_hatpro_temperature_dataframe_tz,
    ):
        """Calculate air pressure at specific height."""

        test_temperature = conftest_mock_hatpro_temperature_dataframe_tz.copy(deep=True)
        test_idx = self.test_levels[-1]
        test_pressure = conftest_mock_weather_dataframe_tz["pressure"] * 100  # to Pa

        compare_pressure = self.test_class.get_air_pressure(
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
        name="TestBackendConstructProfile::test_extrapolate_air_pressure",
        depends=[
            "TestBackendConstructProfile::test_constructor_init",
            "TestBackendConstructProfile::test_get_air_pressure",
        ],
    )
    def test_extrapolate_air_pressure(
        self,
        conftest_mock_weather_dataframe_tz,
        conftest_mock_hatpro_temperature_dataframe_tz,
    ):
        """Extrapolate air pressure across all scan heights."""

        test_temperature = conftest_mock_hatpro_temperature_dataframe_tz.copy(deep=True)
        test_pressure = conftest_mock_weather_dataframe_tz["pressure"]

        compare_pressure = self.test_class.extrapolate_air_pressure(
            surface_pressure=test_pressure, temperature=test_temperature
        )

        assert isinstance(compare_pressure, pd.DataFrame)
        assert compare_pressure.index.equals(test_temperature.index)
        assert np.allclose(compare_pressure[0], test_pressure * 100)
        assert ((compare_pressure.ge(1000)).all()).all()
        assert not np.allclose(test_pressure, compare_pressure)
        assert not (test_pressure.isna()).any()
        assert not compare_pressure[compare_pressure.columns[-1]].equals(
            compare_pressure[0]  # Int64 index doesn't support indexing directly with -1
        )

    @pytest.mark.dependency(
        name="TestBackendConstructProfile::test_get_mixing_ratio",
        depends=[
            "TestBackendConstructProfile::test_constructor_init",
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
        ).multiply(self.test_class.constants.r_vapour)
        test_pressure = 100 * conftest_mock_weather_dataframe_tz["pressure"].copy(
            deep=True
        )
        test_data = {}
        for level in test_wvp.columns:
            test_data[level] = test_pressure * (1 / (level + 1))
        test_pressure = pd.DataFrame(
            data=test_data, columns=test_wvp.columns, index=test_wvp.index
        )
        test_ratio = (self.test_class.constants.r_dry * test_wvp) / (
            self.test_class.constants.r_vapour * test_pressure
        )

        compare_ratio = self.test_class.get_mixing_ratio(
            wv_pressure=test_wvp, d_pressure=test_pressure
        )

        assert isinstance(compare_ratio, pd.DataFrame)
        assert compare_ratio.index.equals(test_wvp.index)
        assert np.allclose(compare_ratio, test_ratio)

    @pytest.mark.dependency(
        name="TestBackendConstructProfile::test_get_virtual_temperature"
    )
    def test_get_virtual_temperature(
        self,
        conftest_mock_weather_dataframe_tz,
        conftest_mock_hatpro_temperature_dataframe_tz,
    ):
        """Calculate virtual temperature."""

        test_temperature = conftest_mock_hatpro_temperature_dataframe_tz.copy(deep=True)
        test_pressure = 100 * conftest_mock_weather_dataframe_tz["pressure"].copy(
            deep=True
        )
        test_data = {}
        for level in test_temperature.columns:  # placeholder values for mixing ratio
            test_data[level] = test_pressure / (test_pressure * (level + 1))
        test_ratio = pd.DataFrame(
            data=test_data,
            columns=test_temperature.columns,
            index=test_temperature.index,
        )

        compare_temperature = self.test_class.get_virtual_temperature(
            temperature=test_temperature, mixing_ratio=test_ratio
        )
        assert isinstance(compare_temperature, pd.DataFrame)
        assert compare_temperature.index.equals(test_temperature.index)
        assert not np.allclose(compare_temperature, test_temperature)

    @pytest.mark.dependency(
        name="TestBackendConstructProfile::test_get_reduced_temperature",
        depends=["TestBackendConstructProfile::test_constructor_init"],
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
            self.test_class.constants.r_dry
            / (test_temperature * np.abs(self.test_class.constants.g))
        )
        test_exp = np.exp(test_station)
        assert isinstance(test_exp, pd.DataFrame)
        assert np.allclose(test_exp, np.e**test_station)
        test_mslp = test_pressure * test_exp
        assert isinstance(test_mslp, pd.DataFrame)

        compare_mslp = self.test_class.get_reduced_pressure(
            station_pressure=test_pressure,
            virtual_temperature=test_temperature,
            elevation=test_elevation,
        )
        assert isinstance(compare_mslp, pd.DataFrame)
        assert np.allclose(compare_mslp, test_mslp)

    @pytest.mark.dependency(
        name="TestBackendConstructProfile::test_get_potential_temperature",
        depends=["TestBackendConstructProfile::test_constructor_init"],
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
            self.test_class.constants.ref_pressure / test_pressure
        ) ** (self.test_class.constants.r_dry / self.test_class.constants.cp)
        assert isinstance(test_potential, pd.DataFrame)

        compare_potential = self.test_class.get_potential_temperature(
            virtual_temperature=test_temperature, pressure=test_pressure
        )
        assert isinstance(compare_potential, pd.DataFrame)
        assert np.allclose(compare_potential, test_potential)
