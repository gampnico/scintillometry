"""Copyright 2023 Scintillometry Contributors.

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

Use the `conftest_boilerplate` fixture to avoid duplicating tests.
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
        test_profile (ProfileConstructor): An initialised
            ProfileConstructor object.
        test_elevation (float): Mocked station elevation.
    """

    test_profile = scintillometry.backend.constructions.ProfileConstructor()
    test_elevation = 600.0

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
        conftest_mock_hatpro_dataset,
        conftest_mock_hatpro_scan_levels,
        conftest_boilerplate,
    ):
        """Calculate water vapour pressure."""

        assert isinstance(self.test_profile.constants.r_vapour, (float, int))
        test_wvp = (
            conftest_mock_hatpro_dataset["humidity"]
            * conftest_mock_hatpro_dataset["temperature"]
            * self.test_profile.constants.r_vapour
        )
        compare_wvp = self.test_profile.get_water_vapour_pressure(
            abs_humidity=conftest_mock_hatpro_dataset["humidity"],
            temperature=conftest_mock_hatpro_dataset["temperature"],
        )

        conftest_boilerplate.check_dataframe(dataframe=compare_wvp)
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
        ref_pressure = conftest_mock_weather_dataframe_tz["pressure"].copy(deep=True)
        ref_pressure = ref_pressure.asfreq("10T").multiply(100)  # same indices
        assert isinstance(ref_pressure, pd.Series)

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
        pd.testing.assert_index_equal(
            compare_pressure.index, test_temperature.index, check_names=False
        )
        assert not np.allclose(compare_pressure, ref_pressure)
        assert (compare_pressure < ref_pressure).all()
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
        conftest_boilerplate,
    ):
        """Extrapolate air pressure across all scan heights."""

        test_temperature = conftest_mock_hatpro_temperature_dataframe_tz.copy(deep=True)
        test_pressure = conftest_mock_weather_dataframe_tz["pressure"].copy(deep=True)
        test_pressure = test_pressure.multiply(100)  # hPa -> Pa, do not resample

        compare_pressure = self.test_profile.extrapolate_air_pressure(
            surface_pressure=test_pressure, temperature=test_temperature
        )

        conftest_boilerplate.check_dataframe(dataframe=compare_pressure)
        # verify resampling
        pd.testing.assert_index_equal(compare_pressure.index, test_temperature.index)
        conftest_boilerplate.index_not_equal(
            compare_pressure.index, test_pressure.index
        )
        assert np.allclose(
            compare_pressure[compare_pressure.columns[0]], test_pressure.asfreq("10T")
        )
        for col in compare_pressure.columns.difference([0]):
            assert not np.allclose(test_pressure.asfreq("10T"), compare_pressure[col])
        assert ((compare_pressure.ge(1000)).all()).all()
        assert not np.allclose(
            compare_pressure[compare_pressure.columns[-1]],
            compare_pressure[compare_pressure.columns[0]],
        )

    @pytest.mark.dependency(
        name="TestBackendProfileConstructor::test_get_mixing_ratio",
        depends=["TestBackendProfileConstructor::test_constructor_init"],
    )
    def test_get_mixing_ratio(
        self,
        conftest_mock_weather_dataframe_tz,
        conftest_mock_hatpro_dataset,
        conftest_mock_hatpro_scan_levels,
        conftest_boilerplate,
    ):
        """Calculate mixing ratio for dry air pressure."""

        test_weather = conftest_mock_weather_dataframe_tz.copy(deep=True).asfreq("10T")
        test_pressure = conftest_boilerplate.setup_extrapolated(
            series=test_weather["pressure"].multiply(100),
            levels=conftest_mock_hatpro_scan_levels,
        )
        test_wvp = (
            conftest_mock_hatpro_dataset["humidity"]
            .multiply(conftest_mock_hatpro_dataset["temperature"])
            .multiply(self.test_profile.constants.r_vapour)
        )
        test_ratio = (test_wvp * self.test_profile.constants.r_dry) / (
            self.test_profile.constants.r_vapour * test_pressure
        )

        compare_ratio = self.test_profile.get_mixing_ratio(
            wv_pressure=test_wvp, d_pressure=test_pressure
        )

        conftest_boilerplate.check_dataframe(dataframe=compare_ratio)
        pd.testing.assert_index_equal(compare_ratio.index, test_wvp.index)
        assert np.allclose(compare_ratio, test_ratio)

    @pytest.mark.dependency(
        name="TestBackendProfileConstructor::test_get_virtual_temperature",
    )
    def test_get_virtual_temperature(
        self,
        conftest_mock_weather_dataframe_tz,
        conftest_mock_hatpro_temperature_dataframe_tz,
        conftest_mock_hatpro_scan_levels,
        conftest_boilerplate,
    ):
        """Calculate virtual temperature."""

        ref_temperature = conftest_mock_hatpro_temperature_dataframe_tz.copy(deep=True)
        test_pressure = conftest_mock_weather_dataframe_tz["pressure"].copy(deep=True)
        test_pressure = test_pressure.asfreq("10T").multiply(100)

        test_ratio = conftest_boilerplate.setup_extrapolated(
            series=test_pressure.divide(test_pressure + 1),
            levels=conftest_mock_hatpro_scan_levels,
        )
        test_temperature = ref_temperature * (1 + (0.61 * test_ratio))

        compare_temperature = self.test_profile.get_virtual_temperature(
            temperature=ref_temperature, mixing_ratio=test_ratio
        )
        conftest_boilerplate.check_dataframe(dataframe=compare_temperature)
        pd.testing.assert_index_equal(compare_temperature.index, test_temperature.index)
        assert not np.allclose(compare_temperature, ref_temperature)
        assert np.allclose(compare_temperature, test_temperature)

    @pytest.mark.dependency(
        name="TestBackendProfileConstructor::test_get_reduced_pressure",
        depends=["TestBackendProfileConstructor::test_constructor_init"],
    )
    def test_get_reduced_pressure(
        self,
        conftest_mock_weather_dataframe_tz,
        conftest_mock_hatpro_scan_levels,
        conftest_boilerplate,
    ):
        """Reduce station pressure to mean sea-level pressure."""

        test_weather = conftest_mock_weather_dataframe_tz.copy(deep=True).asfreq("10T")
        test_pressure = conftest_boilerplate.setup_extrapolated(
            series=test_weather["pressure"].multiply(100),
            levels=conftest_mock_hatpro_scan_levels,
        )
        test_temperature = conftest_boilerplate.setup_extrapolated(
            series=test_weather["temperature_2m"].add(273.15),
            levels=conftest_mock_hatpro_scan_levels,
        )
        assert isinstance(test_temperature, pd.DataFrame)
        alpha = self.test_profile.constants.r_dry / np.abs(
            self.test_profile.constants.g
        )
        assert np.isclose(alpha, 29.26)
        test_factor = self.test_elevation / (alpha * test_temperature)
        test_exp = np.exp(test_factor)
        conftest_boilerplate.check_dataframe(dataframe=test_exp)
        assert np.allclose(test_exp, np.e**test_factor)
        test_mslp = test_pressure * test_exp
        assert isinstance(test_mslp, pd.DataFrame)

        compare_mslp = self.test_profile.get_reduced_pressure(
            station_pressure=test_pressure,
            virtual_temperature=test_temperature,
            elevation=self.test_elevation,
        )

        conftest_boilerplate.check_dataframe(dataframe=compare_mslp)
        assert np.allclose(compare_mslp, test_mslp)

    @pytest.mark.dependency(
        name="TestBackendProfileConstructor::test_get_potential_temperature",
        depends=["TestBackendProfileConstructor::test_constructor_init"],
    )
    def test_get_potential_temperature(
        self,
        conftest_mock_hatpro_scan_levels,
        conftest_mock_weather_dataframe_tz,
        conftest_mock_hatpro_temperature_dataframe_tz,
        conftest_boilerplate,
    ):
        """Calculate potential temperature."""

        test_weather = conftest_mock_weather_dataframe_tz.copy(deep=True).asfreq("10T")
        test_pressure = conftest_boilerplate.setup_extrapolated(
            series=test_weather["pressure"].multiply(100),
            levels=conftest_mock_hatpro_scan_levels,
        )
        test_temperature = conftest_mock_hatpro_temperature_dataframe_tz.copy(deep=True)
        test_potential = test_temperature * (
            self.test_profile.constants.ref_pressure / test_pressure
        ) ** (self.test_profile.constants.r_dry / self.test_profile.constants.cp)
        for frame in (test_temperature, test_pressure):
            assert isinstance(frame, pd.DataFrame)
            pd.testing.assert_index_equal(
                frame.index, test_weather.index, check_names=False
            )
            assert all(i in conftest_mock_hatpro_scan_levels for i in frame.columns)
            conftest_boilerplate.check_dataframe(dataframe=frame)
            assert frame.gt(0).values.all()

        compare_potential = self.test_profile.get_potential_temperature(
            temperature=test_temperature, pressure=test_pressure
        )
        conftest_boilerplate.check_dataframe(dataframe=compare_potential)
        assert compare_potential.gt(0).values.all()

        assert np.allclose(compare_potential, test_potential)

    @pytest.mark.dependency(
        name="TestBackendProfileConstructor::test_get_environmental_lapse_rate"
    )
    def test_get_environmental_lapse_rate(
        self,
        conftest_mock_hatpro_dataset,
        conftest_boilerplate,
    ):
        """Calculate environmental lapse rate."""

        test_temperature = conftest_mock_hatpro_dataset["temperature"].copy(deep=True)
        test_dz = -test_temperature.columns.to_series().diff(periods=-1)
        assert np.isnan(test_dz.iloc[-1])
        assert not np.isnan(test_dz.iloc[:-1]).any()
        assert (test_dz.iloc[:-1] > 0).all().all()
        assert len(test_dz) == len(test_temperature.columns)
        test_dt = test_temperature.diff(periods=-1, axis=1)
        conftest_boilerplate.check_dataframe(dataframe=test_dt.iloc[:, :-1])
        test_lapse = test_dt / test_dz
        conftest_boilerplate.check_dataframe(dataframe=test_lapse.iloc[:, :-1])

        compare_lapse = self.test_profile.get_environmental_lapse_rate(
            temperature=test_temperature
        )
        conftest_boilerplate.check_dataframe(dataframe=compare_lapse.iloc[:, :-1])
        assert np.allclose(compare_lapse.iloc[:, :-1], test_lapse.iloc[:, :-1])

    @pytest.mark.dependency(
        name="TestBackendProfileConstructor::test_get_moist_adiabatic_lapse_rate"
    )
    def test_get_moist_adiabatic_lapse_rate(
        self,
        conftest_mock_hatpro_dataset,
        conftest_mock_hatpro_scan_levels,
        conftest_mock_weather_dataframe_tz,
        conftest_boilerplate,
    ):
        """Calculate moist adiabatic lapse rate."""

        test_weather = conftest_mock_weather_dataframe_tz.copy(deep=True)
        test_pressure = conftest_boilerplate.setup_extrapolated(
            series=test_weather["pressure"].asfreq("10T").multiply(100),
            levels=conftest_mock_hatpro_scan_levels,
        )
        test_hatpro = conftest_mock_hatpro_dataset.copy()
        test_wvp = self.test_profile.get_water_vapour_pressure(
            abs_humidity=test_hatpro["humidity"], temperature=test_hatpro["temperature"]
        )
        test_ratio = self.test_profile.get_mixing_ratio(
            wv_pressure=test_wvp, d_pressure=test_pressure
        )
        for frame in [test_wvp, test_ratio]:
            conftest_boilerplate.check_dataframe(dataframe=frame)
        numerator = 1 + (
            (self.test_profile.constants.latent_vapour * test_ratio)
            / (self.test_profile.constants.r_dry * test_hatpro["temperature"])
        )
        denominator = self.test_profile.constants.cp + (
            (test_ratio * self.test_profile.constants.latent_vapour**2)
            / (self.test_profile.constants.r_vapour * test_hatpro["temperature"] ** 2)
        )
        test_malr = self.test_profile.constants.g * (numerator / denominator)

        compare_malr = self.test_profile.get_moist_adiabatic_lapse_rate(
            mixing_ratio=test_ratio, temperature=test_hatpro["temperature"]
        )

        conftest_boilerplate.check_dataframe(dataframe=compare_malr)
        assert np.allclose(compare_malr, test_malr)

    @pytest.mark.dependency(name="TestBackendProfileConstructor::test_get_lapse_rates")
    def test_get_lapse_rates(
        self,
        conftest_mock_weather_dataframe_tz,
        conftest_mock_hatpro_dataset,
        conftest_mock_hatpro_scan_levels,
        conftest_boilerplate,
    ):
        """Calculate lapse rates."""

        test_dataset = {
            "weather": conftest_mock_weather_dataframe_tz.copy(deep=True),
            "vertical": conftest_mock_hatpro_dataset.copy(),
        }
        test_pressure = test_dataset["weather"]["pressure"].asfreq("10T").multiply(100)
        test_ratio = conftest_boilerplate.setup_extrapolated(
            series=test_pressure.divide(test_pressure + 1),
            levels=conftest_mock_hatpro_scan_levels,
        )
        test_dataset["vertical"]["mixing_ratio"] = test_ratio

        compare_rates = self.test_profile.get_lapse_rates(
            temperature=test_dataset["vertical"]["temperature"],
            mixing_ratio=test_dataset["vertical"]["mixing_ratio"],
        )
        assert isinstance(compare_rates, dict)
        for key in ["environmental", "moist_adiabatic"]:
            assert key in compare_rates
        conftest_boilerplate.check_dataframe(
            dataframe=compare_rates["environmental"].iloc[:, :-1]
        )
        conftest_boilerplate.check_dataframe(dataframe=compare_rates["moist_adiabatic"])

    @pytest.mark.dependency(
        name="TestBackendProfileConstructor::test_non_uniform_differencing"
    )
    def test_non_uniform_differencing(
        self,
        conftest_mock_hatpro_temperature_dataframe_tz,
        conftest_mock_hatpro_scan_levels,
        conftest_boilerplate,
    ):
        """Compute centred-differencing scheme for non-uniform data."""

        test_dataframe = conftest_mock_hatpro_temperature_dataframe_tz.copy(deep=True)
        test_cols = test_dataframe.columns
        test_diff = -test_cols.to_series().diff(periods=-1)
        assert isinstance(test_diff, pd.Series)
        assert np.isnan(test_diff.iloc[-1])
        test_diff.iloc[-1] = test_diff.iloc[-2]
        assert not test_diff.isnull().any()

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

        conftest_boilerplate.check_dataframe(dataframe=compare_gradient)
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
        conftest_boilerplate,
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
        for i in compare_gradient.columns:
            assert ptypes.is_numeric_dtype(compare_gradient[i])
        conftest_boilerplate.check_dataframe(dataframe=compare_gradient)
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
        conftest_mock_hatpro_scan_levels,
        conftest_boilerplate,
        arg_method,
    ):
        """Calculate static stability."""

        test_temperature = conftest_mock_hatpro_temperature_dataframe_tz.copy(deep=True)

        compare_stability = self.test_profile.get_static_stability(
            potential_temperature=test_temperature, scheme=arg_method
        )

        conftest_boilerplate.check_dataframe(dataframe=compare_stability)
        pd.testing.assert_index_equal(compare_stability.index, test_temperature.index)
        assert all(
            key in compare_stability.columns for key in conftest_mock_hatpro_scan_levels
        )
        assert np.allclose(compare_stability[compare_stability.columns[0]], 0)

    @pytest.mark.dependency(
        name="TestBackendProfileConstructor::test_get_bulk_richardson",
        depends=["TestBackendProfileConstructor::test_constructor_init"],
    )
    def test_get_bulk_richardson(
        self,
        conftest_mock_hatpro_temperature_dataframe_tz,
        conftest_mock_weather_dataframe_tz,
    ):
        """Calculate bulk Richardson number."""

        test_weather = conftest_mock_weather_dataframe_tz.copy(deep=True).asfreq("10T")
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
        pd.testing.assert_index_equal(compare_bulk.index, test_temperature.index)
        assert not compare_bulk.isnull().values.any()
        assert not np.isinf(compare_bulk).values.any()
        assert np.allclose(compare_bulk, test_bulk)

    @pytest.mark.dependency(
        name="TestBackendProfileConstructor::test_get_n_squared",
        depends=[
            "TestBackendProfileConstructor::test_constructor_init",
            "TestBackendProfileConstructor::test_get_gradient",
        ],
    )
    def test_get_n_squared(
        self, conftest_mock_hatpro_temperature_dataframe_tz, conftest_boilerplate
    ):
        """Calculate Brunt-Väisälä frequency, squared."""

        test_temperature = conftest_mock_hatpro_temperature_dataframe_tz.copy(deep=True)
        test_grad_pot_temperature = self.test_profile.get_gradient(
            data=test_temperature, method="backward"
        )

        test_brunt = (self.test_profile.constants.g / test_temperature) * (
            test_grad_pot_temperature
        )

        compare_brunt = self.test_profile.get_n_squared(
            potential_temperature=test_temperature, scheme="backward"
        )

        conftest_boilerplate.check_dataframe(dataframe=compare_brunt)
        assert np.allclose(compare_brunt, test_brunt)

    @pytest.mark.dependency(
        name="TestBackendProfileConstructor::test_get_vertical_variables",
        depends=[
            "TestBackendConstants::test_convert_pressure",
            "TestBackendConstants::test_convert_pressure",
        ],
        scope="session",
    )
    @pytest.mark.parametrize("arg_elevation", [None, 600.0])
    def test_get_vertical_variables(
        self,
        conftest_mock_hatpro_dataset,
        conftest_mock_weather_dataframe_tz,
        conftest_boilerplate,
        arg_elevation,
    ):
        """Derive data from vertical measurements."""

        test_vertical = conftest_mock_hatpro_dataset
        test_weather = conftest_mock_weather_dataframe_tz.copy(deep=True)
        test_keys = [
            "temperature",
            "humidity",
            "water_vapour_pressure",
            "air_pressure",
            "mixing_ratio",
            "virtual_temperature",
            "msl_pressure",
            "potential_temperature",
            "grad_potential_temperature",
            "environmental_lapse_rate",
            "moist_adiabatic_lapse_rate",
        ]

        compare_dataset = self.test_profile.get_vertical_variables(
            vertical_data=test_vertical,
            meteo_data=test_weather,
            station_elevation=arg_elevation,
        )

        assert isinstance(compare_dataset, dict)
        assert all(key in compare_dataset for key in test_keys)
        for key in test_keys:
            if key != "environmental_lapse_rate":
                conftest_boilerplate.check_dataframe(compare_dataset[key])
            else:
                conftest_boilerplate.check_dataframe(compare_dataset[key].iloc[:, :-1])
            assert isinstance(compare_dataset[key].index, pd.DatetimeIndex)

            pd.testing.assert_index_equal(
                compare_dataset[key].index, test_vertical["temperature"].index
            )
            conftest_boilerplate.index_not_equal(
                compare_dataset[key].index, test_weather.index
            )
