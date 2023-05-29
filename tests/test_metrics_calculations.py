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

Tests metrics calculated from datasets.

Any test that creates a plot should be explicitly appended with
`plt.close("all")`, otherwise the plots remain open in memory.

Only patch mocks for dependencies that have already been tested. When
patching several mocks via decorators, parameters are applied in the
opposite order::

    # decorators passed correctly
    @patch("lib.bar")
    @patch("lib.foo")
    def test_foobar(self, foo_mock, bar_mock):

        foo_mock.return_value = 1
        bar_mock.return_value = 2

        foo_val, bar_val = foobar(...)  # foo_val = 1, bar_val = 2

    # decorators passed in wrong order
    @patch("lib.foo")
    @patch("lib.bar")
    def test_foobar(self, foo_mock, bar_mock):

        foo_mock.return_value = 1
        bar_mock.return_value = 2

        foo_val, bar_val = foobar(...)  # foo_val = 2, bar_val = 1

"""

import argparse

import kneed
import matplotlib.pyplot as plt
import mpmath
import numpy as np
import pandas as pd
import pandas.api.types as ptypes
import pytest
import sklearn

import scintillometry.backend.constants
import scintillometry.backend.constructions
import scintillometry.backend.derivations
import scintillometry.backend.iterations
import scintillometry.metrics.calculations
import scintillometry.visuals.plotting


class TestMetricsTopography:
    """Test class for topography metrics."""

    @pytest.mark.dependency(name="TestMetricsTopography::test_metrics_topography_init")
    def test_metrics_topography_init(self):
        """Boilerplate implementation in case of future changes."""

        test_class = scintillometry.metrics.calculations.MetricsTopography()
        assert test_class

    @pytest.mark.dependency(
        name="TestMetricsTopography::test_get_path_height_parameters",
        depends=[
            "TestBackendTransectParameters::test_get_all_path_heights",
            "TestBackendTransectParameters::test_print_path_heights",
            "TestMetricsTopography::test_metrics_topography_init",
        ],
        scope="session",
    )
    @pytest.mark.parametrize("arg_regime", ["stable", "unstable", None])
    def test_get_path_height_parameters(
        self, capsys, conftest_mock_transect_dataframe, arg_regime
    ):
        """Get effective and mean path heights of transect."""

        test_metrics = scintillometry.metrics.calculations.MetricsTopography()

        compare_metrics = test_metrics.get_path_height_parameters(
            transect=conftest_mock_transect_dataframe, regime=arg_regime
        )

        assert isinstance(compare_metrics, dict)
        for key in ["stable", "unstable", "None"]:
            assert key in compare_metrics
        for values in compare_metrics.values():
            assert isinstance(values, tuple)
            assert len(values) == 2
            assert all(isinstance(z, float) for z in values)
        for key in ["stable", "unstable"]:
            assert compare_metrics[key][0] < compare_metrics[key][1]
        assert compare_metrics["None"][0] > compare_metrics["None"][1]

        compare_print = capsys.readouterr()
        if arg_regime is None:
            assert "Selected no height dependency:" in compare_print.out
        else:
            assert str(arg_regime) in compare_print.out
        assert (
            str(f"Effective path height:\t{compare_metrics[str(arg_regime)][0]:>0.2f}")
            in compare_print.out
        )
        assert (
            str(f"Mean path height:\t{compare_metrics[str(arg_regime)][1]:>0.2f}")
            in compare_print.out
        )


class TestMetricsFlux:
    """Test class for flux metrics.

    Attributes:
        test_metrics (MetricsFlux): Heat flux calculation class.
        test_date (str): Placeholder for date of data collection.
        test_timestamp (pd.Timestamp): Placeholder for index timestamp.
    """

    test_metrics = scintillometry.metrics.calculations.MetricsFlux()
    test_date = "03 June 2020"
    test_timestamp = pd.Timestamp(f"{test_date} 05:10", tz="CET")

    @pytest.mark.dependency(name="TestMetricsFlux::test_metrics_flux_init")
    def test_metrics_flux_init(self):
        """Boilerplate implementation in case of future changes."""

        test_class = scintillometry.metrics.calculations.MetricsFlux()
        assert test_class.constants
        assert isinstance(
            test_class.constants, scintillometry.backend.constants.AtmosConstants
        )
        assert test_class.derivation
        assert isinstance(
            test_class.derivation,
            scintillometry.backend.derivations.DeriveScintillometer,
        )
        assert test_class.construction
        assert isinstance(
            test_class.construction,
            scintillometry.backend.constructions.ProfileConstructor,
        )
        assert test_class.iteration
        assert isinstance(
            test_class.iteration, scintillometry.backend.iterations.IterationMost
        )
        assert test_class.plotting
        assert isinstance(
            test_class.plotting, scintillometry.visuals.plotting.FigurePlotter
        )

    @pytest.mark.dependency(
        name="TestMetricsFlux::test_construct_flux_dataframe",
        depends=["TestBackendDerivations::test_compute_fluxes"],
        scope="session",
    )
    @pytest.mark.parametrize("arg_kwargs", [(880, 20), None])
    def test_construct_flux_dataframe(self, conftest_mock_merged_dataframe, arg_kwargs):
        """Compute sensible heat flux for free convection."""

        test_frame = conftest_mock_merged_dataframe[["CT2", "H_convection"]]
        if isinstance(arg_kwargs, tuple):
            test_kwargs = {
                "beam_wavelength": arg_kwargs[0],
                "beam_error": arg_kwargs[1],
            }
        else:
            test_kwargs = {}

        compare_metrics = self.test_metrics.construct_flux_dataframe(
            interpolated_data=conftest_mock_merged_dataframe,
            z_eff=(100, 200),
            **test_kwargs,
        )
        for key in ["CT2", "H_free", "H_convection"]:
            assert key in compare_metrics.columns
            assert ptypes.is_numeric_dtype(compare_metrics[key])
        assert not np.allclose(compare_metrics["CT2"], test_frame["CT2"])
        assert np.allclose(compare_metrics["H_convection"], test_frame["H_convection"])

    @pytest.mark.dependency(name="TestMetricsFlux::test_get_nearest_time_index")
    @pytest.mark.parametrize("arg_time", ["05:19", "05:20", "05:21"])
    def test_get_nearest_time_index(
        self, conftest_mock_hatpro_humidity_dataframe_tz, arg_time
    ):
        """Get index of data nearest to time stamp."""

        test_frame = conftest_mock_hatpro_humidity_dataframe_tz.copy(deep=True)
        test_time = pd.Timestamp(f"{self.test_date} {arg_time}", tz="CET")
        assert test_time.strftime("%H:%M") == arg_time
        assert test_time.strftime("%d %B %Y") == self.test_date
        assert test_time.tz.zone == "CET"

        compare_index = self.test_metrics.get_nearest_time_index(
            data=test_frame, time_stamp=test_time
        )
        assert isinstance(compare_index, pd.Timestamp)
        assert compare_index.strftime("%H:%M") == "05:20"
        assert compare_index.strftime("%d %B %Y") == self.test_date

    @pytest.mark.dependency(
        name="TestMetricsFlux::test_append_vertical_variables_missing"
    )
    def test_append_vertical_variables_missing(self):
        """Pass unmodified dataset if vertical data is missing."""

        test_dataset = {"weather": pd.DataFrame()}
        compare_dataset = self.test_metrics.append_vertical_variables(data=test_dataset)

        assert isinstance(compare_dataset, dict)
        assert "vertical" not in compare_dataset
        assert "weather" in compare_dataset
        assert isinstance(compare_dataset["weather"], pd.DataFrame)

    @pytest.mark.dependency(name="TestMetricsFlux::test_append_vertical_variables")
    def test_append_vertical_variables(
        self,
        conftest_mock_weather_dataframe_tz,
        conftest_mock_hatpro_dataset,
        conftest_boilerplate,
    ):
        """Derive and append vertical measurements to dataset."""

        test_dataset = {
            "weather": conftest_mock_weather_dataframe_tz.copy(deep=True),
            "vertical": conftest_mock_hatpro_dataset,
        }

        test_keys = [
            "temperature",
            "humidity",
            "water_vapour_pressure",
            "air_pressure",
            "mixing_ratio",
            "virtual_temperature",
            "msl_pressure",
            "potential_temperature",
        ]
        compare_dataset = self.test_metrics.append_vertical_variables(data=test_dataset)

        assert isinstance(compare_dataset, dict)
        assert all(key in compare_dataset for key in ["vertical", "weather"])
        assert isinstance(compare_dataset["weather"], pd.DataFrame)
        assert isinstance(compare_dataset["vertical"], dict)
        assert all(key in compare_dataset["vertical"] for key in test_keys)
        for key in test_keys:
            conftest_boilerplate.check_dataframe(compare_dataset["vertical"][key])
            assert isinstance(compare_dataset["vertical"][key].index, pd.DatetimeIndex)

            pd.testing.assert_index_equal(
                compare_dataset["vertical"][key].index,
                test_dataset["vertical"]["temperature"].index,
            )
            conftest_boilerplate.index_not_equal(
                compare_dataset["vertical"][key].index, test_dataset["weather"].index
            )

    @pytest.mark.dependency(name="TestMetricsFlux::test_match_time_at_threshold")
    @pytest.mark.parametrize("arg_lessthan", [True, False])
    @pytest.mark.parametrize("arg_empty", [True, False])
    @pytest.mark.parametrize("arg_timestamp", [True, False])
    def test_match_time_at_threshold(
        self, conftest_mock_weather_dataframe_tz, arg_lessthan, arg_empty, arg_timestamp
    ):
        """Derive and append vertical measurements to dataset."""

        test_weather = conftest_mock_weather_dataframe_tz.copy(deep=True)
        if arg_empty:
            test_weather["global_irradiance"] = 20
        if arg_timestamp:
            test_timestamp = self.test_timestamp
        else:
            test_timestamp = None

        compare_time = self.test_metrics.match_time_at_threshold(
            series=test_weather["global_irradiance"],
            threshold=20,
            lessthan=arg_lessthan,
            min_time=test_timestamp,
        )

        if not arg_empty:
            assert isinstance(compare_time, pd.Timestamp)
            if arg_lessthan:
                assert compare_time.strftime("%H:%M") == "05:10"
            else:
                assert compare_time.strftime("%H:%M") == "05:19"
        else:
            assert compare_time is None

    @pytest.mark.dependency(name="TestMetricsFlux::test_get_regression")
    @pytest.mark.parametrize("arg_intercept", [True, False])
    @pytest.mark.parametrize("arg_mismatch_index", [True, False])
    def test_get_regression(
        self, conftest_boilerplate, arg_intercept, arg_mismatch_index
    ):
        """Perform regression on labelled data."""

        rng = np.random.default_rng()
        test_index = pd.date_range(start=self.test_timestamp, periods=100, freq="T")
        test_data = rng.random(size=len(test_index))

        test_x = pd.Series(name="obukhov", data=test_data, index=test_index)
        test_y = pd.Series(name="other_obukhov", data=test_data + 0.5, index=test_index)
        if arg_mismatch_index:
            test_y = test_y[:-5]
            conftest_boilerplate.index_not_equal(test_x.index, test_y.index)
            assert test_y.shape == (95,)
        assert isinstance(test_x, pd.Series)
        assert test_x.shape == (100,)
        test_keys = ["fit", "score", "regression_line"]

        compare_regression = self.test_metrics.get_regression(
            x_data=test_x, y_data=test_y, intercept=arg_intercept
        )
        assert isinstance(compare_regression, dict)
        assert all(key in compare_regression for key in test_keys)
        assert isinstance(
            compare_regression["fit"], sklearn.linear_model.LinearRegression
        )
        if arg_intercept:
            assert compare_regression["fit"].fit_intercept
        else:
            assert not compare_regression["fit"].fit_intercept
        assert isinstance(compare_regression["score"], float)
        assert isinstance(compare_regression["regression_line"], np.ndarray)
        assert not (np.isnan(compare_regression["regression_line"])).any()
        assert len(test_y.index) == len(compare_regression["regression_line"])

    @pytest.mark.dependency(name="TestMetricsFlux::test_get_elbow_point")
    @pytest.mark.parametrize("arg_min_index", [None, 0, 50])
    @pytest.mark.parametrize("arg_max_index", [None, 180, 190])
    @pytest.mark.parametrize("arg_curve", [1, -1])
    def test_get_elbow_point(self, arg_min_index, arg_max_index, arg_curve):
        """Estimate elbow point of curve."""

        test_heights = np.arange(0, 200, 10)
        test_data = pd.DataFrame(columns=test_heights)
        test_data.loc[self.test_timestamp] = 10 / np.arange(0.1, 2.1, 0.1) ** 2
        test_curve = test_data.loc[self.test_timestamp]
        assert np.allclose(test_curve.index, test_heights)
        if arg_curve > 0:
            test_direction = "decreasing"
        else:
            test_direction = "increasing"
        if not arg_max_index:
            max_index = test_curve.index[-1]
        else:
            max_index = arg_max_index
        if not arg_min_index:
            min_index = test_curve.index[0]
        else:
            min_index = arg_min_index

        test_indices = test_curve.index[
            (test_curve.index >= min_index) & (test_curve.index <= max_index)
        ]
        test_knee = kneed.KneeLocator(
            test_curve[test_indices] * arg_curve,
            test_indices,
            S=1.5,
            curve="convex",
            online=True,
            direction=test_direction,
            interp_method="interp1d",
        )
        elbows = test_curve[test_knee.all_elbows_y][
            test_curve[test_knee.all_elbows_y] < test_curve[test_indices].mean()
        ]
        if not elbows.empty:
            test_elbow = min(elbows.index)
        else:
            test_elbow = min(test_knee.all_elbows_y)

        compare_elbow = self.test_metrics.get_elbow_point(
            series=test_curve * arg_curve,
            min_index=arg_min_index,
            max_index=arg_max_index,
        )
        assert ptypes.is_int64_dtype(compare_elbow)
        assert np.isclose(compare_elbow, test_elbow)

    @pytest.mark.dependency(name="TestMetricsFlux::test_get_boundary_height")
    def test_get_boundary_height(self, capsys):
        """Estimate boundary layer height."""

        test_heights = np.arange(0, 200, 10)
        test_curve = pd.DataFrame(columns=test_heights)
        test_curve.loc[self.test_timestamp] = 10 / np.arange(0.1, 2.1, 0.1) ** 2
        test_intersect = 90

        compare_height = self.test_metrics.get_boundary_height(
            grad_potential=test_curve,
            time_index=self.test_timestamp,
            max_height=180,
        )
        compare_print = capsys.readouterr()
        assert "Estimated boundary layer height: 90 m." in compare_print.out
        assert isinstance(compare_height, np.int64)
        assert compare_height == test_intersect

    @pytest.mark.dependency(name="TestMetricsFlux::test_get_boundary_height_error")
    def test_get_boundary_height_error(self, capsys):
        """Warn if boundary layer height not found."""

        test_heights = np.arange(0, 200, 10)
        test_curve = pd.DataFrame(columns=test_heights)
        test_curve.loc[self.test_timestamp] = 10 / np.arange(0.1, 2.1, 0.1) ** 2

        with pytest.warns(RuntimeWarning):
            compare_height = self.test_metrics.get_boundary_height(
                grad_potential=test_curve,
                time_index=self.test_timestamp,
                max_height=50,
            )
            compare_print = capsys.readouterr()
            assert compare_height is None
            assert "Failed to estimate boundary layer height." in compare_print.out

    @pytest.mark.dependency(
        name="TestMetricsFlux::test_compare_lapse_rates",
        depends=["TestMetricsFlux::test_append_vertical_variables"],
    )
    def test_compare_lapse_rates(
        self, conftest_mock_weather_dataframe_tz, conftest_mock_hatpro_dataset
    ):
        """Find instability from lapse rates."""

        test_dataset = {
            "weather": conftest_mock_weather_dataframe_tz,
            "vertical": conftest_mock_hatpro_dataset.copy(),
        }
        test_dataset = self.test_metrics.append_vertical_variables(data=test_dataset)

        compare_stabilities = self.test_metrics.compare_lapse_rates(
            air_temperature=test_dataset["vertical"]["temperature"],
            saturated=test_dataset["vertical"]["saturated_temperature"],
            unsaturated=test_dataset["vertical"]["unsaturated_temperature"],
        )

        assert isinstance(compare_stabilities, tuple)
        for series in compare_stabilities:
            assert isinstance(series, pd.Series)
            assert ptypes.is_bool_dtype(series)
        plt.close("all")

    @pytest.mark.dependency(name="TestMetricsFlux::test_get_switch_time_error_method")
    @pytest.mark.parametrize("arg_method", ["incorrect_method", "eddy", None])
    def test_get_switch_time_error_method(
        self,
        conftest_mock_weather_dataframe_tz,
        conftest_mock_hatpro_dataset,
        conftest_mock_save_figure,
        arg_method,
    ):
        """Raise error if incorrect switch time algorithm specified."""

        _ = conftest_mock_save_figure

        test_error = {
            "weather": conftest_mock_weather_dataframe_tz.copy(deep=True),
            "vertical": conftest_mock_hatpro_dataset,
        }
        error_msg = f"Switch time algorithm not implemented for '{arg_method}'."

        with pytest.raises(NotImplementedError, match=error_msg):
            self.test_metrics.get_switch_time(
                data=test_error, method=arg_method, local_time=None
            )

    @pytest.mark.dependency(name="TestMetricsFlux::test_get_switch_time_error_data")
    @pytest.mark.parametrize("arg_method", ["sun", "bulk"])
    def test_get_switch_time_error_data(
        self, conftest_mock_derived_dataframe, conftest_mock_save_figure, arg_method
    ):
        """Raise error if no data available to calculate switch time."""

        _ = conftest_mock_save_figure

        test_error = {
            "weather": conftest_mock_derived_dataframe[["CT2", "temperature_2m"]]
        }

        error_msg = (
            "No data to calculate switch time.",
            "Set <local_time> manually with `--switch-time`.",
        )
        with pytest.raises(UnboundLocalError, match=" ".join(error_msg)):
            self.test_metrics.get_switch_time(
                data=test_error, method=arg_method, local_time=None, ri_crit=-1000.0
            )

    @pytest.mark.dependency(
        name="TestMetricsFlux::test_get_switch_time_fallback",
        depends=[
            "TestMetricsFlux::test_get_switch_time_error_data",
            "TestMetricsFlux::test_get_switch_time_error_method",
        ],
    )
    def test_get_switch_time_fallback(self, conftest_mock_weather_dataframe_tz):
        """Get switch time by falling back to global irradiance."""

        test_dataset = {
            "weather": conftest_mock_weather_dataframe_tz.copy(deep=True),
        }
        compare_switch = self.test_metrics.get_switch_time(
            data=test_dataset, method="sun"
        )
        assert isinstance(compare_switch, pd.Timestamp)
        assert compare_switch.strftime("%H:%M") == "05:19"
        assert compare_switch.tz.zone == "CET"

    @pytest.mark.dependency(name="TestMetricsFlux::test_get_switch_time_convert")
    @pytest.mark.parametrize("arg_string", [True, False])
    def test_get_switch_time_convert(
        self, conftest_mock_weather_dataframe_tz, arg_string
    ):
        """Convert local time string to timestamp."""

        test_dataset = {
            "weather": conftest_mock_weather_dataframe_tz.copy(deep=True),
            "timestamp": self.test_timestamp,
        }
        if not arg_string:
            test_time = pd.Timestamp(f"{self.test_date} 05:19", tz="CET")
        else:
            test_time = "05:19"

        compare_switch = self.test_metrics.get_switch_time(
            data=test_dataset, method="sun", local_time=test_time
        )
        assert isinstance(compare_switch, pd.Timestamp)
        assert compare_switch.strftime("%H:%M") == "05:19"
        assert compare_switch.strftime("%d %B %Y") == self.test_date
        assert compare_switch.tz.zone == "CET"

    @pytest.mark.dependency(
        name="TestMetricsFlux::test_get_switch_time_vertical",
        depends=[
            "TestMetricsFlux::test_append_vertical_variables",
            "TestMetricsFlux::test_get_switch_time_fallback",
        ],
    )
    @pytest.mark.parametrize("arg_method", ["static", "lapse", "bulk"])
    def test_get_switch_time_vertical(
        self,
        conftest_mock_weather_dataframe_tz,
        conftest_mock_hatpro_dataset,
        conftest_boilerplate,
        arg_method,
    ):
        """Get stability switch time using vertical data."""

        test_dataset = {
            "weather": conftest_mock_weather_dataframe_tz.copy(deep=True),
            "vertical": conftest_mock_hatpro_dataset,
            "timestamp": self.test_timestamp,
        }
        test_dataset = self.test_metrics.append_vertical_variables(data=test_dataset)

        for key in ["grad_potential_temperature", "potential_temperature"]:
            assert key in test_dataset["vertical"]
            conftest_boilerplate.check_dataframe(
                dataframe=test_dataset["vertical"][key]
            )

        compare_switch = self.test_metrics.get_switch_time_vertical(
            data=test_dataset, method=arg_method, ri_crit=0.25
        )

        assert isinstance(compare_switch, pd.Timestamp)
        assert compare_switch.strftime("%H:%M") == "05:10"

    @pytest.mark.dependency(
        name="TestMetricsFlux::test_get_switch_time_vertical_wrapper",
        depends=["TestMetricsFlux::test_get_switch_time_vertical"],
    )
    @pytest.mark.parametrize("arg_local_time", ["05:19", None])
    def test_get_switch_time_vertical_wrapper(
        self,
        conftest_mock_weather_dataframe_tz,
        conftest_mock_hatpro_dataset,
        conftest_boilerplate,
        arg_local_time,
    ):
        """Use wrapper to get switch time from vertical data."""

        test_dataset = {
            "weather": conftest_mock_weather_dataframe_tz.copy(deep=True),
            "vertical": conftest_mock_hatpro_dataset,
            "timestamp": self.test_timestamp,
        }
        test_dataset = self.test_metrics.append_vertical_variables(data=test_dataset)

        for key in ["grad_potential_temperature", "potential_temperature"]:
            assert key in test_dataset["vertical"]
            conftest_boilerplate.check_dataframe(
                dataframe=test_dataset["vertical"][key]
            )

        compare_switch = self.test_metrics.get_switch_time(
            data=test_dataset, method="static", local_time=arg_local_time, ri_crit=0.25
        )

        assert isinstance(compare_switch, pd.Timestamp)
        if arg_local_time:
            assert compare_switch.strftime("%H:%M") == arg_local_time
        else:
            assert compare_switch.strftime("%H:%M") == "05:10"

    @pytest.mark.dependency(
        name="TestMetricsFlux::test_plot_switch_time_stability",
        depends=[
            "TestMetricsFlux::test_append_vertical_variables",
            "TestMetricsFlux::test_get_switch_time_vertical",
        ],
    )
    @pytest.mark.parametrize("arg_gradient", ["grad_potential_temperature", False])
    @pytest.mark.parametrize("arg_location", [None, "Test Location"])
    def test_plot_switch_time_stability(
        self,
        conftest_mock_weather_dataframe_tz,
        conftest_mock_hatpro_dataset,
        conftest_mock_save_figure,
        arg_gradient,
        arg_location,
    ):
        """Plot potential temperature profiles at switch time."""

        _ = conftest_mock_save_figure

        test_dataset = {
            "weather": conftest_mock_weather_dataframe_tz.copy(deep=True),
            "vertical": conftest_mock_hatpro_dataset.copy(),
            "timestamp": self.test_timestamp,
        }
        test_dataset = self.test_metrics.append_vertical_variables(data=test_dataset)
        test_time = self.test_metrics.get_switch_time(
            data=test_dataset, local_time="05:19", method="sun"
        )
        assert isinstance(test_time, pd.Timestamp)
        test_vertical = test_dataset["vertical"]
        if arg_location:
            test_location = f"\nat {arg_location}, "
        else:
            test_location = ",\n"
        test_labels = ["Potential Temperature, [K]"]
        if arg_gradient:
            assert "grad_potential_temperature" in test_vertical
            test_title = (
                "Vertical Profiles of Potential Temperature ",
                f"and Gradient of Potential Temperature{test_location}",
                f"{self.test_date} 05:20 CET",
            )
            test_labels.append(r"Gradient of Potential Temperature, [K$\cdot$m$^{-1}$]")
        else:
            test_vertical.pop("grad_potential_temperature", None)
            assert "grad_potential_temperature" not in test_vertical
            test_title = (
                f"Vertical Profile of Potential Temperature{test_location}",
                f"{self.test_date} 05:20 CET",
            )

        compare_plots = self.test_metrics.plot_switch_time_stability(
            data=test_vertical, local_time=test_time, location=arg_location
        )
        assert isinstance(compare_plots, list)
        for compare_tuple in compare_plots:
            assert isinstance(compare_tuple, tuple)
            assert isinstance(compare_tuple[0], plt.Figure)
            compare_ax = compare_tuple[1]
            if arg_gradient:
                assert isinstance(compare_ax, np.ndarray)
                assert all(isinstance(ax, plt.Axes) for ax in compare_ax)
                assert compare_tuple[0].texts[0].get_text() == "".join(test_title)
                assert compare_ax[0].yaxis.get_label_text() == "Height [m]"
                for i in range(len(compare_ax)):
                    assert compare_ax[i].xaxis.get_label_text() == test_labels[i]
            else:
                assert isinstance(compare_ax, plt.Axes)
                assert compare_ax.get_title() == "".join(test_title)
                assert compare_ax.yaxis.get_label_text() == "Height [m]"

        plt.close("all")

    @pytest.mark.dependency(name="TestMetricsFlux::test_plot_lapse_rates")
    @pytest.mark.parametrize("arg_height", [None, 100])
    @pytest.mark.parametrize("arg_location", [None, "Test Location"])
    def test_plot_lapse_rates(
        self,
        conftest_mock_weather_dataframe_tz,
        conftest_mock_hatpro_dataset,
        conftest_mock_save_figure,
        conftest_boilerplate,
        arg_location,
        arg_height,
    ):
        """Plot comparison of lapse rates and boundary layer height."""

        _ = conftest_mock_save_figure

        test_dataset = {
            "weather": conftest_mock_weather_dataframe_tz.copy(deep=True),
            "vertical": conftest_mock_hatpro_dataset.copy(),
            "timestamp": self.test_timestamp,
        }
        test_dataset = self.test_metrics.append_vertical_variables(data=test_dataset)
        for key in ["grad_potential_temperature", "environmental_lapse_rate"]:
            assert key in test_dataset["vertical"]
        if arg_location:
            test_location = f"\nat {arg_location}, "
        else:
            test_location = ",\n"
        test_title = f"{test_location}{self.test_date} 05:10 CET"

        compare_plots = self.test_metrics.plot_lapse_rates(
            vertical_data=test_dataset["vertical"],
            dry_adiabat=self.test_metrics.constants.dalr,
            local_time=self.test_timestamp,
            location=arg_location,
            bl_height=arg_height,
        )
        assert isinstance(compare_plots, list)
        assert all(isinstance(compare_tuple, tuple) for compare_tuple in compare_plots)

        compare_params = {
            "lapse": {
                "title": "Temperature Lapse Rates",
                "x_label": r"Lapse Rate, [Km$^{-1}$]",
                "y_label": "Height [m]",
                "plot": (compare_plots[0]),
            },
            "parcel": {
                "title": "Vertical Profiles of Parcel Temperature",
                "x_label": "Temperature, [K]",
                "y_label": "Height [m]",
                "plot": (compare_plots[1]),
            },
        }

        for params in compare_params.values():
            conftest_boilerplate.check_plot(plot_params=params, title=test_title)

        plt.close("all")

    @pytest.mark.dependency(
        name="TestMetricsFlux::test_calculate_switch_time_no_vertical",
    )
    def test_calculate_switch_time_no_vertical(
        self, conftest_mock_weather_dataframe_tz, conftest_mock_save_figure
    ):
        """Plot potential temperature profiles at switch time."""

        _ = conftest_mock_save_figure

        test_dataset = {
            "weather": conftest_mock_weather_dataframe_tz.copy(deep=True),
            "timestamp": self.test_timestamp.replace(hour=5, minute=10),
        }

        compare_switch = self.test_metrics.calculate_switch_time(
            datasets=test_dataset, method="sun", switch_time="05:10", location=""
        )
        assert isinstance(compare_switch, pd.Timestamp)
        assert compare_switch.strftime("%H:%M") == "05:10"
        assert 1 not in plt.get_fignums()

        plt.close("all")

    @pytest.mark.dependency(
        name="TestMetricsFlux::test_calculate_switch_time",
        depends=[
            "TestMetricsFlux::test_get_switch_time_vertical",
            "TestMetricsFlux::test_plot_switch_time_stability",
            "TestMetricsFlux::test_plot_lapse_rates",
        ],
    )
    @pytest.mark.filterwarnings("ignore:No knee/elbow found")
    @pytest.mark.parametrize("arg_potential", [True, False])
    @pytest.mark.parametrize("arg_lapse", [True, False])
    def test_calculate_switch_time(
        self,
        conftest_mock_weather_dataframe_tz,
        conftest_mock_hatpro_dataset,
        conftest_mock_save_figure,
        arg_potential,
        arg_lapse,
    ):
        """Calculate and plot switch time."""

        _ = conftest_mock_save_figure

        test_dataset = {
            "weather": conftest_mock_weather_dataframe_tz.copy(deep=True),
            "timestamp": self.test_timestamp.replace(hour=5, minute=10),
            "vertical": conftest_mock_hatpro_dataset.copy(),
        }
        test_dataset = self.test_metrics.append_vertical_variables(data=test_dataset)

        if not arg_potential:
            test_dataset["vertical"].pop("grad_potential_temperature")
            assert "grad_potential_temperature" not in test_dataset["vertical"]
        else:
            assert "grad_potential_temperature" in test_dataset["vertical"]

        if not arg_lapse:
            test_dataset["vertical"].pop("environmental_lapse_rate")
            assert "environmental_lapse_rate" not in test_dataset["vertical"]
        else:
            assert "environmental_lapse_rate" in test_dataset["vertical"]

        plt.close("all")
        compare_switch = self.test_metrics.calculate_switch_time(
            datasets=test_dataset, method="sun", switch_time="05:10", location=""
        )
        assert isinstance(compare_switch, pd.Timestamp)
        assert compare_switch.strftime("%H:%M") == "05:10"

        compare_fignums = len(plt.get_fignums())
        if not arg_potential:
            assert 1 not in plt.get_fignums()
        elif arg_potential and arg_lapse:
            assert compare_fignums == 5
        else:
            assert compare_fignums == 3

        plt.close("all")

    @pytest.mark.dependency(name="TestMetricsFlux::test_plot_derived_metrics")
    @pytest.mark.parametrize("arg_regime", ["stable", "unstable", None])
    def test_plot_derived_metrics(
        self,
        conftest_mock_save_figure,
        conftest_mock_derived_dataframe,
        conftest_boilerplate,
        arg_regime,
    ):
        """Plot time series of heat fluxes for free convection."""

        _ = conftest_mock_save_figure

        test_frame = conftest_mock_derived_dataframe
        if arg_regime is not None:
            test_conditions = f"{arg_regime.capitalize()} Conditions"
        else:
            test_conditions = "No Height Dependency"
        test_title = (
            "Sensible Heat Fluxes from On-Board Software and",
            f"for Free Convection ({test_conditions}),\n{self.test_date}",
        )

        compare_plots = self.test_metrics.plot_derived_metrics(
            derived_data=test_frame,
            time_id=test_frame.index[0],
            regime=arg_regime,
            location="",
        )
        assert isinstance(compare_plots, list)
        assert all(isinstance(compare_tuple, tuple) for compare_tuple in compare_plots)

        compare_params = {
            "plot": (compare_plots[0]),
            "x_label": "Time, CET",
            "y_label": r"Sensible Heat Flux, [W$\cdot$m$^{-2}$]",
            "title": " ".join(test_title),
        }
        conftest_boilerplate.check_plot(plot_params=compare_params)

        plt.close("all")

    @pytest.mark.dependency(name="TestMetricsFlux::test_plot_iterated_metrics")
    @pytest.mark.parametrize("arg_location", [None, "", "Test Location"])
    def test_plot_iterated_metrics(
        self,
        conftest_mock_save_figure,
        conftest_mock_iterated_dataframe,
        conftest_boilerplate,
        arg_location,
    ):
        """Plot time series of iteration and comparison to free convection."""

        _ = conftest_mock_save_figure

        test_frame = conftest_mock_iterated_dataframe.copy(deep=True)
        test_stamp = conftest_mock_iterated_dataframe.index[0]
        if arg_location:
            test_frame.attrs["name"] = arg_location
            assert "name" in test_frame.attrs
        if arg_location:
            test_location = f" at {arg_location}"
        else:
            test_location = ""
        test_title = f"{test_location},\n{self.test_date}"

        compare_plots = self.test_metrics.plot_iterated_metrics(
            iterated_data=test_frame,
            time_stamp=test_stamp,
            location=arg_location,
        )
        assert isinstance(compare_plots, list)
        assert all(isinstance(compare_tuple, tuple) for compare_tuple in compare_plots)

        compare_params = {
            "iteration": {
                "plot": (compare_plots[0]),
                "title": "Sensible Heat Flux",
                "x_label": "Time, CET",
                "y_label": r"Sensible Heat Flux, [W$\cdot$m$^{-2}$]",
            },
            "comparison": {
                "plot": (compare_plots[1]),
                "title": "Sensible Heat Flux from Free Convection and Iteration",
                "x_label": "Time, CET",
                "y_label": r"Sensible Heat Flux, [W$\cdot$m$^{-2}$]",
            },
        }
        for params in compare_params.values():
            conftest_boilerplate.check_plot(plot_params=params, title=test_title)

        plt.close("all")

        # site_location is pending deprecation
        with pytest.warns(PendingDeprecationWarning):
            compare_plots = self.test_metrics.plot_iterated_metrics(
                iterated_data=test_frame,
                time_stamp=test_stamp,
                site_location=arg_location,  # pylint:disable=unexpected-keyword-arg
            )
        assert isinstance(compare_plots, list)
        assert all(isinstance(compare_tuple, tuple) for compare_tuple in compare_plots)
        for params in compare_params.values():
            conftest_boilerplate.check_plot(plot_params=params, title=test_title)

        plt.close("all")

    @pytest.mark.dependency(
        name="TestMetricsFlux::test_iterate_fluxes",
        depends=["TestMetricsFlux::test_append_vertical_variables"],
    )
    @pytest.mark.filterwarnings("ignore:No knee/elbow found")
    def test_iterate_fluxes(
        self,
        conftest_mock_save_figure,
        conftest_mock_merged_dataframe,
        conftest_mock_weather_dataframe_tz,
        conftest_mock_hatpro_dataset,
    ):
        """Compute sensible heat fluxes with MOST through iteration."""

        _ = conftest_mock_save_figure  # otherwise figure is saved to disk

        test_dataset = {
            "weather": conftest_mock_weather_dataframe_tz.copy(deep=True),
            "interpolated": conftest_mock_merged_dataframe.copy(deep=True),
            "vertical": conftest_mock_hatpro_dataset,
            "timestamp": self.test_timestamp,
        }
        test_dataset = self.test_metrics.append_vertical_variables(data=test_dataset)
        compare_metrics = self.test_metrics.iterate_fluxes(
            z_parameters={"stable": (100, 150), "unstable": (2, 150)},
            datasets=test_dataset,
            most_id="an1988",
            algorithm="sun",
            switch_time="05:20",
            location="",
        )
        compare_keys = [
            "u_star",
            "theta_star",
            "f_ct2",
            "shf",
            "obukhov",
        ]
        assert isinstance(compare_metrics, pd.DataFrame)
        for key in compare_keys:
            assert not (compare_metrics[key].isnull()).any()
            assert key in compare_metrics.keys()
            assert all(isinstance(x, mpmath.mpf) for x in compare_metrics[key])
        plt.close("all")


class TestMetricsWorkflow:
    """Test class for metrics workflow.

    Attributes:
        test_metrics (MetricsFlux): Heat flux calculation class.
        test_workflow (MetricsWorkflow): Workflow for metrics.
        test_date (str): Placeholder for date of data collection.
    """

    test_metrics = scintillometry.metrics.calculations.MetricsFlux()
    test_workflow = scintillometry.metrics.calculations.MetricsWorkflow()
    test_date = "03 June 2020"

    @pytest.mark.dependency(name="TestMetricsWorkflow::test_metrics_workflow_init")
    def test_metrics_workflow_init(self):
        """Boilerplate implementation in case of future changes."""

        test_class = scintillometry.metrics.calculations.MetricsWorkflow()
        assert test_class
        assert isinstance(test_class, type(self.test_metrics))

    @pytest.mark.dependency(
        name="TestMetricsWorkflow::test_calculate_standard_metrics",
        depends=[
            "TestMetricsWorkflow::test_metrics_workflow_init",
            "TestMetricsTopography::test_get_path_height_parameters",
            "TestMetricsFlux::test_construct_flux_dataframe",
            "TestMetricsFlux::test_plot_derived_metrics",
            "TestMetricsFlux::test_append_vertical_variables",
            "TestMetricsFlux::test_iterate_fluxes",
            "TestMetricsFlux::test_plot_iterated_metrics",
        ],
        scope="module",
    )
    @pytest.mark.filterwarnings("ignore:No knee/elbow found")
    @pytest.mark.parametrize("arg_switch_time", [None, "05:20"])
    @pytest.mark.parametrize("arg_vertical", [True, False])
    def test_calculate_standard_metrics(
        self,
        conftest_mock_save_figure,
        conftest_mock_bls_dataframe_tz,
        conftest_mock_weather_dataframe_tz,
        conftest_mock_transect_dataframe,
        conftest_mock_merged_dataframe,
        conftest_mock_hatpro_dataset,
        conftest_boilerplate,
        arg_switch_time,
        arg_vertical,
    ):
        """Calculate and plot standard metrics.

        This test has a relatively long runtime.
        """

        _ = conftest_mock_save_figure

        test_data = {
            "bls": conftest_mock_bls_dataframe_tz.copy(deep=True),
            "weather": conftest_mock_weather_dataframe_tz.copy(deep=True),
            "transect": conftest_mock_transect_dataframe.copy(deep=True),
            "interpolated": conftest_mock_merged_dataframe.copy(deep=True),
            "timestamp": conftest_mock_bls_dataframe_tz.index[0],
        }

        if arg_vertical:
            test_data["vertical"] = conftest_mock_hatpro_dataset.copy()

        test_args = argparse.Namespace(
            regime="unstable",
            switch_time=arg_switch_time,
            beam_wavelength=880,
            beam_error=20,
            most_name="an1988",
            location="Test Location",
            algorithm="sun",
        )

        plt.close("all")

        compare_data = self.test_workflow.calculate_standard_metrics(
            arguments=test_args, data=test_data
        )

        assert 1 in plt.get_fignums()
        if arg_vertical:
            assert len(plt.get_fignums()) == 8
        else:
            assert len(plt.get_fignums()) == 3

        plt.close("all")

        assert isinstance(compare_data, dict)
        for key in ["derivation", "iteration"]:
            assert key in compare_data
            assert isinstance(compare_data[key], pd.DataFrame)
            conftest_boilerplate.check_timezone(
                dataframe=compare_data[key], tzone="CET"
            )

    @pytest.mark.dependency(
        name="TestMetricsWorkflow::test_calculate_standard_metrics_no_vertical",
        depends=["TestMetricsWorkflow::test_calculate_standard_metrics"],
    )
    def test_calculate_standard_metrics_no_vertical(
        self,
        conftest_mock_save_figure,
        conftest_mock_bls_dataframe_tz,
        conftest_mock_weather_dataframe_tz,
        conftest_mock_transect_dataframe,
        conftest_mock_merged_dataframe,
    ):
        """Raise error when referencing non-existent vertical data."""

        _ = conftest_mock_save_figure

        test_data = {
            "bls": conftest_mock_bls_dataframe_tz.copy(deep=True),
            "weather": conftest_mock_weather_dataframe_tz.copy(deep=True),
            "transect": conftest_mock_transect_dataframe.copy(deep=True),
            "interpolated": conftest_mock_merged_dataframe.copy(deep=True),
            "timestamp": conftest_mock_bls_dataframe_tz.index[0],
        }

        test_data = self.test_metrics.append_vertical_variables(data=test_data)
        assert "vertical" not in test_data
        error_msg = (
            "No data to calculate switch time.",
            "Set <local_time> manually with `--switch-time`.",
        )
        with pytest.raises(UnboundLocalError, match=" ".join(error_msg)):
            self.test_workflow.calculate_standard_metrics(
                data=test_data, method="static"
            )
        plt.close("all")

    @pytest.mark.dependency(
        name="TestMetricsWorkflow::test_compare_innflux",
        depends=["TestMetricsWorkflow::test_metrics_workflow_init"],
        scope="class",
    )
    @pytest.mark.parametrize("arg_location", [None, "", "Test Location"])
    def test_compare_innflux(
        self,
        conftest_mock_save_figure,
        conftest_boilerplate,
        conftest_generate_series,
        arg_location,
    ):
        """Compares input data to innFLUX data."""

        _ = conftest_mock_save_figure

        test_data, test_index = conftest_generate_series
        test_obukhov = pd.Series(data=test_data, index=test_index)
        test_shf = pd.Series(data=test_data, index=test_index)
        test_base_dataframe = pd.DataFrame(
            data={"obukhov": test_obukhov, "shf": test_shf},
        )
        test_ext_dataframe = test_base_dataframe.add(0.5)

        if arg_location:
            test_location = f" at {arg_location}"
        else:
            test_location = ""
        test_title = f"{test_location},\n{self.test_date}"
        test_regression_string = "Regression Between\nMOST Iteration and innFLUX"

        compare_plots = self.test_workflow.compare_innflux(
            own_data=test_base_dataframe,
            innflux_data=test_ext_dataframe,
            location=arg_location,
        )
        assert isinstance(compare_plots, list)
        assert all(isinstance(compare_tuple, tuple) for compare_tuple in compare_plots)
        compare_params = {
            "obukhov": {
                "title": "Obukhov Length from Scintillometer and innFLUX",
                "y_label": "Obukhov Length, [m]",
                "x_label": "Time, CET",
                "plot": (compare_plots[0]),
            },
            "shf": {
                "title": "Sensible Heat Flux from Scintillometer and innFLUX",
                "y_label": r"Sensible Heat Flux, [W$\cdot$m$^{-2}$]",
                "x_label": "Time, CET",
                "plot": (compare_plots[1]),
            },
            "obukhov_regression": {
                "title": f"Obukhov Length {test_regression_string}",
                "y_label": "Obukhov Length, [m] (innFLUX)",
                "x_label": "Obukhov Length, [m] (MOST Iteration)",
                "plot": (compare_plots[2]),
            },
            "shf_regression": {
                "title": f"Sensible Heat Flux {test_regression_string}",
                "y_label": r"Sensible Heat Flux, [W$\cdot$m$^{-2}$] (innFLUX)",
                "x_label": r"Sensible Heat Flux, [W$\cdot$m$^{-2}$] (MOST Iteration)",
                "plot": (compare_plots[3]),
            },
        }

        for params in compare_params.values():
            conftest_boilerplate.check_plot(plot_params=params, title=test_title)

        plt.close("all")

    @pytest.mark.dependency(name="TestMetricsWorkflow::test_compare_eddy_error")
    def test_compare_eddy_error(
        self,
        conftest_mock_save_figure,
        conftest_mock_innflux_dataframe_tz,
        conftest_mock_iterated_dataframe,
    ):
        """Compares input data to external eddy covariance data."""

        _ = conftest_mock_save_figure
        test_source = "Wrong Source"
        error_msg = f"{test_source} measurements are not supported. Use 'innflux'."

        with pytest.raises(NotImplementedError, match=error_msg):
            self.test_workflow.compare_eddy(
                own_data=conftest_mock_iterated_dataframe.copy(deep=True),
                ext_data=conftest_mock_innflux_dataframe_tz.copy(deep=True),
                source=test_source,
                location="Test Location",
            )
        plt.close("all")

    @pytest.mark.dependency(
        name="TestMetricsWorkflow::test_compare_eddy",
        depends=[
            "TestMetricsWorkflow::test_compare_innflux",
            "TestMetricsWorkflow::test_compare_eddy_error",
        ],
    )
    def test_compare_eddy(
        self, conftest_mock_save_figure, conftest_generate_series, conftest_boilerplate
    ):
        """Compares input data to external eddy covariance data."""

        _ = conftest_mock_save_figure

        test_data, test_index = conftest_generate_series
        test_obukhov = pd.Series(data=test_data, index=test_index)
        test_shf = pd.Series(data=test_data, index=test_index)
        test_base_dataframe = pd.DataFrame(
            data={"obukhov": test_obukhov, "shf": test_shf},
        )
        test_ext_dataframe = test_base_dataframe.add(0.5)
        test_location = "Test Location"
        test_title = f" at {test_location},\n{self.test_date}"

        compare_plots = self.test_workflow.compare_eddy(
            own_data=test_base_dataframe,
            ext_data=test_ext_dataframe,
            source="innflux",
            location="Test Location",
        )
        assert isinstance(compare_plots, list)
        assert all(isinstance(compare_tuple, tuple) for compare_tuple in compare_plots)

        compare_params = {
            "obukhov": {
                "title": "Obukhov Length from Scintillometer and innFLUX",
                "y_label": "Obukhov Length, [m]",
                "x_label": "Time, CET",
                "plot": (compare_plots[0]),
            },
            "shf": {
                "title": "Sensible Heat Flux from Scintillometer and innFLUX",
                "y_label": r"Sensible Heat Flux, [W$\cdot$m$^{-2}$]",
                "x_label": "Time, CET",
                "plot": (compare_plots[1]),
            },
        }
        for params in compare_params.values():
            conftest_boilerplate.check_plot(plot_params=params, title=test_title)

        plt.close("all")
