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

Tests metrics calculated from datasets.

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

import matplotlib.pyplot as plt
import mpmath
import numpy as np
import pandas as pd
import pandas.api.types as ptypes
import pytest

import scintillometry.metrics.calculations


class TestMetricsTopographyClass:
    """Test class for topography metrics."""

    @pytest.mark.dependency(
        name="TestMetricsTopographyClass::test_metrics_topography_init"
    )
    def test_metrics_topography_init(self):
        """Boilerplate implementation in case of future changes."""

        test_class = scintillometry.metrics.calculations.MetricsTopography()
        assert test_class

    @pytest.mark.dependency(
        name="TestMetricsTopographyClass::test_get_z_params",
        depends=[
            "TestBackendTransects::test_get_all_z_parameters",
            "TestBackendTransects::test_print_z_parameters",
            "TestMetricsTopographyClass::test_metrics_topography_init",
        ],
        scope="session",
    )
    @pytest.mark.parametrize("arg_regime", ["stable", "unstable", None])
    def test_get_z_params(self, capsys, conftest_mock_transect_dataframe, arg_regime):
        """Get effective and mean path heights of transect."""

        test_metrics = scintillometry.metrics.calculations.MetricsTopography()
        test_args = argparse.Namespace(regime=arg_regime)
        assert isinstance(test_args, argparse.Namespace)
        assert test_args.regime == arg_regime

        compare_metrics = test_metrics.get_z_params(
            user_args=test_args, transect=conftest_mock_transect_dataframe
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
        if not arg_regime:
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


class TestMetricsFluxClass:
    """Test class for flux metrics.

    Attributes:
        test_metrics (MetricsFlux): Heat flux calculation class.
        test_date (str): Placeholder for date of data collection.
    """

    test_metrics = scintillometry.metrics.calculations.MetricsFlux()
    test_date = "03 June 2020"
    test_timestamp = pd.Timestamp(f"{test_date} 05:10", tz="CET")

    @pytest.mark.dependency(name="TestMetricsFluxClass::test_metrics_flux_init")
    def test_metrics_flux_init(self):
        """Boilerplate implementation in case of future changes."""

        test_class = scintillometry.metrics.calculations.MetricsFlux()
        assert test_class

    @pytest.mark.dependency(
        name="TestMetricsFluxClass::test_construct_flux_dataframe",
        depends=["TestBackendDerivations::test_compute_fluxes"],
        scope="session",
    )
    @pytest.mark.parametrize("arg_regime", ["stable", "unstable", None])
    def test_construct_flux_dataframe(self, conftest_mock_merged_dataframe, arg_regime):
        """Compute sensible heat flux for free convection."""

        test_args = argparse.Namespace(
            regime=arg_regime, beam_wavelength=880, beam_error=20
        )
        test_frame = conftest_mock_merged_dataframe[["CT2", "H_convection"]]

        compare_metrics = self.test_metrics.construct_flux_dataframe(
            user_args=test_args,
            interpolated_data=conftest_mock_merged_dataframe,
            z_eff=(100, 200),
        )
        for key in ["CT2", "H_free", "H_convection"]:
            assert key in compare_metrics.columns
            assert ptypes.is_numeric_dtype(compare_metrics[key])
        assert not np.allclose(compare_metrics["CT2"], test_frame["CT2"])
        assert np.allclose(compare_metrics["H_convection"], test_frame["H_convection"])

    @pytest.mark.dependency(name="TestMetricsFluxClass::test_get_nearest_time_index")
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
        name="TestMetricsFluxClass::test_append_vertical_variables_missing"
    )
    def test_append_vertical_variables_missing(self):
        """Pass unmodified dataset if vertical data is missing."""

        test_dataset = {"weather": pd.DataFrame()}
        compare_dataset = self.test_metrics.append_vertical_variables(data=test_dataset)

        assert isinstance(compare_dataset, dict)
        assert "vertical" not in compare_dataset
        assert "weather" in compare_dataset
        assert isinstance(compare_dataset["weather"], pd.DataFrame)

    @pytest.mark.dependency(name="TestMetricsFluxClass::test_append_vertical_variables")
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

    @pytest.mark.dependency(name="TestMetricsFluxClass::test_match_time_at_threshold")
    @pytest.mark.parametrize("arg_lessthan", [True, False])
    @pytest.mark.parametrize("arg_empty", [True, False])
    def test_match_time_at_threshold(
        self, conftest_mock_weather_dataframe_tz, arg_lessthan, arg_empty
    ):
        """Derive and append vertical measurements to dataset."""

        test_weather = conftest_mock_weather_dataframe_tz.copy(deep=True)
        if arg_empty:
            test_weather["global_irradiance"] = 20

        compare_time = self.test_metrics.match_time_at_threshold(
            series=test_weather["global_irradiance"],
            threshold=20,
            lessthan=arg_lessthan,
        )

        if not arg_empty:
            assert isinstance(compare_time, pd.Timestamp)
            if arg_lessthan:
                assert compare_time.strftime("%H:%M") == "05:10"
            else:
                assert compare_time.strftime("%H:%M") == "05:19"
        else:
            assert compare_time is None

    @pytest.mark.dependency(
        name="TestMetricsFluxClass::test_get_switch_time_error_method"
    )
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

    @pytest.mark.dependency(
        name="TestMetricsFluxClass::test_get_switch_time_error_data"
    )
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
        name="TestMetricsFluxClass::test_get_switch_time_fallback",
        depends=[
            "TestMetricsFluxClass::test_get_switch_time_error_data",
            "TestMetricsFluxClass::test_get_switch_time_error_method",
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

    @pytest.mark.dependency(name="TestMetricsFluxClass::test_get_switch_time_convert")
    def test_get_switch_time_convert(self, conftest_mock_weather_dataframe_tz):
        """Convert local time string to timestamp."""

        test_dataset = {
            "weather": conftest_mock_weather_dataframe_tz.copy(deep=True),
            "timestamp": self.test_timestamp,
        }
        compare_switch = self.test_metrics.get_switch_time(
            data=test_dataset, method="sun", local_time="05:19"
        )
        assert isinstance(compare_switch, pd.Timestamp)
        assert compare_switch.strftime("%H:%M") == "05:19"
        assert compare_switch.strftime("%d %B %Y") == self.test_date
        assert compare_switch.tz.zone == "CET"

    @pytest.mark.dependency(
        name="TestMetricsFluxClass::test_get_switch_time_vertical",
        depends=[
            "TestMetricsFluxClass::test_append_vertical_variables",
            "TestMetricsFluxClass::test_get_switch_time_fallback",
        ],
    )
    @pytest.mark.parametrize("arg_local_time", ["05:19", None])
    @pytest.mark.parametrize("arg_method", ["sun", "static", "bulk"])
    def test_get_switch_time_vertical(
        self,
        conftest_mock_weather_dataframe_tz,
        conftest_mock_hatpro_dataset,
        conftest_boilerplate,
        arg_local_time,
        arg_method,
    ):
        """Get time where stability conditions change."""

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
            data=test_dataset,
            method=arg_method,
            local_time=arg_local_time,
            ri_crit=0.25,
        )

        assert isinstance(compare_switch, pd.Timestamp)
        if arg_local_time:
            assert compare_switch.strftime("%H:%M") == arg_local_time
        elif arg_method in ["static", "bulk"]:
            assert compare_switch.strftime("%H:%M") == "05:10"
        else:
            assert compare_switch.strftime("%H:%M") == "05:19"

    @pytest.mark.dependency(
        name="TestMetricsFluxClass::test_plot_switch_time_stability",
        depends=[
            "TestMetricsFluxClass::test_append_vertical_variables",
            "TestMetricsFluxClass::test_get_switch_time_vertical",
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
        if not arg_gradient:
            test_vertical.pop("grad_potential_temperature", None)
            assert "grad_potential_temperature" not in test_vertical
            test_title = (
                f"Vertical Profile of Potential Temperature{test_location}",
                f"{self.test_date} 05:20 CET",
            )
        else:
            assert "grad_potential_temperature" in test_vertical
            test_title = (
                "Vertical Profiles of Potential Temperature ",
                f"and Gradient of Potential Temperature{test_location}",
                f"{self.test_date} 05:20 CET",
            )
            test_labels.append(r"Gradient of Potential Temperature, [K$\cdot$m$^{-1}$]")

        compare_fig, compare_ax = self.test_metrics.plot_switch_time_stability(
            data=test_vertical, local_time=test_time, location=arg_location
        )
        assert isinstance(compare_fig, plt.Figure)

        if not arg_gradient:
            assert isinstance(compare_ax, plt.Axes)
            assert compare_ax.get_title() == "".join(test_title)
            assert compare_ax.yaxis.get_label().get_text() == "Height [m]"
        else:
            assert isinstance(compare_ax, np.ndarray)
            assert all(isinstance(ax, plt.Axes) for ax in compare_ax)
            assert compare_fig.texts[0].get_text() == "".join(test_title)
            assert compare_ax[0].yaxis.get_label().get_text() == "Height [m]"
            for i in range(len(compare_ax)):
                assert compare_ax[i].xaxis.get_label().get_text() == test_labels[i]

        plt.close("all")

    @pytest.mark.dependency(
        name="TestMetricsFluxClass::test_calculate_switch_time",
        depends=[
            "TestMetricsFluxClass::test_get_switch_time_vertical",
            "TestMetricsFluxClass::test_plot_switch_time_stability",
        ],
    )
    @pytest.mark.parametrize("arg_vertical", [True, False])
    @pytest.mark.parametrize("arg_potential", [True, False])
    def test_calculate_switch_time(
        self,
        conftest_mock_weather_dataframe_tz,
        conftest_mock_hatpro_dataset,
        conftest_mock_save_figure,
        arg_vertical,
        arg_potential,
    ):
        """Plot potential temperature profiles at switch time."""

        _ = conftest_mock_save_figure

        test_dataset = {
            "weather": conftest_mock_weather_dataframe_tz.copy(deep=True),
            "timestamp": self.test_timestamp.replace(hour=5, minute=10),
        }
        if arg_vertical:
            test_dataset["vertical"] = conftest_mock_hatpro_dataset.copy()
            if arg_potential:
                test_dataset = self.test_metrics.append_vertical_variables(
                    data=test_dataset
                )

        test_title = (
            "Vertical Profile of Potential Temperature,",
            f"\n{self.test_date} 05:10 CET",
        )

        compare_switch = self.test_metrics.calculate_switch_time(
            datasets=test_dataset, method="sun", switch_time="05:10", location=""
        )
        assert isinstance(compare_switch, pd.Timestamp)
        assert compare_switch.strftime("%H:%M") == "05:10"

        if arg_vertical and arg_potential:
            assert 1 in plt.get_fignums()
            compare_fig = plt.gca()
            assert compare_fig.title.get_text() == "".join(test_title)
        else:
            assert 1 not in plt.get_fignums()

        plt.close("all")

    @pytest.mark.dependency(name="TestMetricsFluxClass::test_plot_derived_metrics")
    @pytest.mark.parametrize("arg_regime", ["stable", "unstable", None])
    def test_plot_derived_metrics(
        self, conftest_mock_save_figure, conftest_mock_derived_dataframe, arg_regime
    ):
        """Plot time series of heat fluxes for free convection."""

        _ = conftest_mock_save_figure

        test_frame = conftest_mock_derived_dataframe
        test_args = test_args = argparse.Namespace(
            regime=arg_regime, beam_wavelength=880, beam_error=20
        )
        compare_fig = self.test_metrics.plot_derived_metrics(
            user_args=test_args, derived_data=test_frame, time_id=test_frame.index[0]
        )
        assert isinstance(compare_fig, plt.Figure)

        compare_ax = plt.gca()
        if arg_regime:
            compare_conditions = f"{arg_regime.capitalize()} Conditions"
        else:
            compare_conditions = "No Height Dependency"
        compare_title = (
            "Sensible Heat Fluxes from On-Board Software and",
            f"for Free Convection ({compare_conditions}),\n{self.test_date}",
        )
        assert compare_ax.get_title() == " ".join(compare_title)
        plt.close("all")

    @pytest.mark.dependency(name="TestMetricsFluxClass::test_plot_iterated_metrics")
    @pytest.mark.parametrize("arg_location", [None, "", "Test Location"])
    def test_plot_iterated_metrics(
        self, conftest_mock_save_figure, conftest_mock_iterated_dataframe, arg_location
    ):
        """Plot time series of iteration and comparison to free convection."""

        _ = conftest_mock_save_figure

        test_frame = conftest_mock_iterated_dataframe.copy(deep=True)
        test_stamp = conftest_mock_iterated_dataframe.index[0]
        if arg_location:
            test_frame.attrs["name"] = arg_location
            assert "name" in test_frame.attrs

        compare_iter, compare_comp = self.test_metrics.plot_iterated_metrics(
            iterated_data=test_frame,
            time_stamp=test_stamp,
            site_location=arg_location,
        )

        for fig in [compare_iter, compare_comp]:
            assert isinstance(fig, plt.Figure)

        if arg_location:
            test_location = f" at {arg_location}"
        else:
            test_location = ""
        test_iter_title = (
            "Sensible Heat Flux",
            f"{test_location},\n{self.test_date}",
        )
        assert compare_iter.gca().get_title() == "".join(test_iter_title)

        test_comp_title = (
            "Sensible Heat Flux from Free Convection and Iteration",
            f"{test_location},\n{self.test_date}",
        )
        assert compare_comp.gca().get_title() == "".join(test_comp_title)

        plt.close("all")

    @pytest.mark.dependency(
        name="TestMetricsFluxClass::test_iterate_fluxes",
        depends=["TestMetricsFluxClass::test_append_vertical_variables"],
    )
    def test_iterate_fluxes(
        self,
        conftest_mock_save_figure,
        conftest_mock_merged_dataframe,
        conftest_mock_weather_dataframe_tz,
        conftest_mock_hatpro_dataset,
    ):
        """Compute sensible heat fluxes with MOST through iteration."""

        _ = conftest_mock_save_figure  # otherwise figure is saved to disk

        test_args = argparse.Namespace(
            switch_time="05:20", beam_wavelength=880, beam_error=20, algorithm="sun"
        )
        test_dataset = {
            "weather": conftest_mock_weather_dataframe_tz.copy(deep=True),
            "interpolated": conftest_mock_merged_dataframe.copy(deep=True),
            "vertical": conftest_mock_hatpro_dataset,
            "timestamp": self.test_timestamp,
        }
        test_dataset = self.test_metrics.append_vertical_variables(data=test_dataset)
        compare_metrics = self.test_metrics.iterate_fluxes(
            user_args=test_args,
            z_parameters={"stable": (100, 150), "unstable": (2, 150)},
            datasets=test_dataset,
            most_id="an1988",
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
            assert all(isinstance(x, (mpmath.mpf)) for x in compare_metrics[key])
        plt.close("all")


class TestMetricsWorkflowClass:
    """Test class for metrics workflow.

    Attributes:
        test_metrics (MetricsFlux): Heat flux calculation class.
        test_date (str): Placeholder for date of data collection.
    """

    test_metrics = scintillometry.metrics.calculations.MetricsFlux()
    test_date = "03 June 2020"

    @pytest.mark.dependency(name="TestMetricsWorkflowClass::test_metrics_workflow_init")
    def test_metrics_workflow_init(self):
        """Boilerplate implementation in case of future changes."""

        test_class = scintillometry.metrics.calculations.MetricsWorkflow()
        assert test_class
        assert isinstance(test_class, type(self.test_metrics))

    @pytest.mark.dependency(
        name="TestMetricsWorkflow::test_calculate_standard_metrics",
        depends=[
            "TestMetricsWorkflowClass::test_metrics_workflow_init",
            "TestMetricsTopographyClass::test_get_z_params",
            "TestMetricsFluxClass::test_construct_flux_dataframe",
            "TestMetricsFluxClass::test_plot_derived_metrics",
            "TestMetricsFluxClass::test_append_vertical_variables",
            "TestMetricsFluxClass::test_iterate_fluxes",
            "TestMetricsFluxClass::test_plot_iterated_metrics",
        ],
        scope="module",
    )
    @pytest.mark.parametrize("arg_regime", [None, "stable", "unstable"])
    @pytest.mark.parametrize("arg_switch_time", [None, "05:20"])
    @pytest.mark.parametrize("arg_most_name", ["an1988", "li2012"])
    @pytest.mark.parametrize("arg_location", [None, "", "Test Location"])
    @pytest.mark.parametrize("arg_algorithm", ["sun", "static"])
    def test_calculate_standard_metrics(
        self,
        conftest_mock_save_figure,
        conftest_mock_bls_dataframe_tz,
        conftest_mock_weather_dataframe_tz,
        conftest_mock_transect_dataframe,
        conftest_mock_merged_dataframe,
        conftest_mock_hatpro_dataset,
        conftest_boilerplate,
        arg_regime,
        arg_switch_time,
        arg_most_name,
        arg_location,
        arg_algorithm,
    ):
        """Calculate and plot standard metrics.

        This test has a relatively long runtime.
        """

        _ = conftest_mock_save_figure

        test_class = scintillometry.metrics.calculations.MetricsWorkflow()
        test_data = {
            "bls": conftest_mock_bls_dataframe_tz.copy(deep=True),
            "weather": conftest_mock_weather_dataframe_tz.copy(deep=True),
            "transect": conftest_mock_transect_dataframe.copy(deep=True),
            "interpolated": conftest_mock_merged_dataframe.copy(deep=True),
            "timestamp": conftest_mock_bls_dataframe_tz.index[0],
            "vertical": conftest_mock_hatpro_dataset.copy(),
        }

        test_args = argparse.Namespace(
            regime=arg_regime,
            switch_time=arg_switch_time,
            beam_wavelength=880,
            beam_error=20,
            most_name=arg_most_name,
            location=arg_location,
            algorithm=arg_algorithm,
        )

        compare_data = test_class.calculate_standard_metrics(
            arguments=test_args, data=test_data
        )
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

        test_class = scintillometry.metrics.calculations.MetricsWorkflow()
        test_data = {
            "bls": conftest_mock_bls_dataframe_tz.copy(deep=True),
            "weather": conftest_mock_weather_dataframe_tz.copy(deep=True),
            "transect": conftest_mock_transect_dataframe.copy(deep=True),
            "interpolated": conftest_mock_merged_dataframe.copy(deep=True),
            "timestamp": conftest_mock_bls_dataframe_tz.index[0],
        }

        test_data = self.test_metrics.append_vertical_variables(data=test_data)
        test_args = argparse.Namespace(
            regime="unstable",
            switch_time=None,
            beam_wavelength=880,
            beam_error=20,
            most_name="an1988",
            location=None,
            algorithm="static",
        )
        error_msg = (
            "No data to calculate switch time.",
            "Set <local_time> manually with `--switch-time`.",
        )
        with pytest.raises(UnboundLocalError, match=" ".join(error_msg)):
            test_class.calculate_standard_metrics(arguments=test_args, data=test_data)
            plt.close("all")

    @pytest.mark.dependency(
        name="TestMetricsWorkflow::test_compare_innflux",
        depends=["TestMetricsWorkflowClass::test_metrics_workflow_init"],
        scope="class",
    )
    @pytest.mark.parametrize("arg_location", [None, "", "Test Location"])
    def test_compare_innflux(
        self,
        conftest_mock_save_figure,
        conftest_mock_innflux_dataframe_tz,
        conftest_mock_iterated_dataframe,
        arg_location,
    ):
        """Compares input data to InnFLUX data."""

        _ = conftest_mock_save_figure

        test_metrics = scintillometry.metrics.calculations.MetricsWorkflow()
        test_args = argparse.Namespace(location=arg_location)
        compare_obukhov, compare_shf = test_metrics.compare_innflux(
            arguments=test_args,
            innflux_data=conftest_mock_innflux_dataframe_tz,
            comparison_data=conftest_mock_iterated_dataframe,
        )

        for fig in [compare_obukhov, compare_shf]:
            assert isinstance(fig, plt.Figure)

        if arg_location:
            test_location = f" at {arg_location}"
        else:
            test_location = ""

        test_iter_title = (
            "Obukhov Length from Scintillometer and InnFLUX",
            f"{test_location},\n{self.test_date}",
        )
        assert compare_obukhov.gca().get_title() == "".join(test_iter_title)

        test_comp_title = (
            "Sensible Heat Flux from Scintillometer and InnFLUX",
            f"{test_location},\n{self.test_date}",
        )
        assert compare_shf.gca().get_title() == "".join(test_comp_title)

        plt.close("all")
