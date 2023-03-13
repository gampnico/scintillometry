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
    """Test class for flux metrics."""

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

        test_metrics = scintillometry.metrics.calculations.MetricsFlux()

        test_args = argparse.Namespace(
            regime=arg_regime, beam_wavelength=880, beam_error=20
        )
        test_frame = conftest_mock_merged_dataframe[["CT2", "H_convection"]]

        compare_metrics = test_metrics.construct_flux_dataframe(
            user_args=test_args,
            interpolated_data=conftest_mock_merged_dataframe,
            z_eff=(100, 200),
        )
        for key in ["CT2", "H_free", "H_convection"]:
            assert key in compare_metrics.columns
            assert ptypes.is_numeric_dtype(compare_metrics[key])
        assert not np.allclose(compare_metrics["CT2"], test_frame["CT2"])
        assert np.allclose(compare_metrics["H_convection"], test_frame["H_convection"])

    @pytest.mark.dependency(name="TestMetricsFluxClass::test_plot_derived_metrics")
    @pytest.mark.parametrize("arg_regime", ["stable", "unstable", None])
    def test_plot_derived_metrics(
        self, conftest_mock_save_figure, conftest_mock_derived_dataframe, arg_regime
    ):
        """Plot time series of heat fluxes for free convection."""

        _ = conftest_mock_save_figure
        test_metrics = scintillometry.metrics.calculations.MetricsFlux()
        test_frame = conftest_mock_derived_dataframe
        test_args = test_args = argparse.Namespace(
            regime=arg_regime, beam_wavelength=880, beam_error=20
        )
        compare_fig = test_metrics.plot_derived_metrics(
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
            f"for Free Convection ({compare_conditions}), 03 June 2020",
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
        test_metrics = scintillometry.metrics.calculations.MetricsFlux()
        test_frame = conftest_mock_iterated_dataframe.copy(deep=True)
        test_stamp = conftest_mock_iterated_dataframe.index[0]
        if arg_location:
            test_frame.attrs["name"] = arg_location
            assert "name" in test_frame.attrs

        compare_iter, compare_comp = test_metrics.plot_iterated_metrics(
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
            f"{test_location}, 03 June 2020",
        )
        assert compare_iter.gca().get_title() == "".join(test_iter_title)

        test_comp_title = (
            "Sensible Heat Flux from Free Convection and Iterated Flux",
            f"{test_location}, 03 June 2020",
        )
        assert compare_comp.gca().get_title() == "".join(test_comp_title)

        plt.close("all")

    @pytest.mark.dependency(name="TestMetricsFluxClass::test_iterate_fluxes")
    def test_iterate_fluxes(
        self,
        conftest_mock_save_figure,
        conftest_mock_derived_dataframe,
    ):
        """Compute sensible heat fluxes with MOST through iteration."""

        _ = conftest_mock_save_figure  # otherwise figure is saved to disk
        test_metrics = scintillometry.metrics.calculations.MetricsFlux()

        test_args = argparse.Namespace(
            switch_time="05:24", beam_wavelength=880, beam_error=20
        )

        compare_metrics = test_metrics.iterate_fluxes(
            user_args=test_args,
            z_parameters={"stable": (100, 150), "unstable": (2, 150)},
            interpolated_data=conftest_mock_derived_dataframe,
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

    Effectively serves as integration test for module.
    """

    @pytest.mark.dependency(name="TestMetricsWorkflowClass::test_metrics_workflow_init")
    def test_metrics_workflow_init(self):
        """Boilerplate implementation in case of future changes."""

        test_class = scintillometry.metrics.calculations.MetricsWorkflow()
        assert test_class

    @pytest.mark.dependency(
        name="TestMetricsWorkflow::test_calculate_standard_metrics",
        depends=[
            "TestMetricsWorkflowClass::test_metrics_workflow_init",
            "TestMetricsTopographyClass::test_get_z_params",
            "TestMetricsFluxClass::test_construct_flux_dataframe",
            "TestMetricsFluxClass::test_plot_derived_metrics",
            "TestMetricsFluxClass::test_iterate_fluxes",
            "TestMetricsFluxClass::test_plot_iterated_metrics",
        ],
        scope="module",
    )
    @pytest.mark.parametrize("arg_regime", [None, "stable", "unstable"])
    @pytest.mark.parametrize("arg_switch_time", [None, "05:24"])
    @pytest.mark.parametrize("arg_most_name", ["an1988", "li2012"])
    @pytest.mark.parametrize("arg_location", [None, "", "Test Location"])
    def test_calculate_standard_metrics(
        self,
        conftest_mock_save_figure,
        conftest_mock_bls_dataframe_tz,
        conftest_mock_weather_dataframe_tz,
        conftest_mock_transect_dataframe,
        conftest_mock_merged_dataframe,
        arg_regime,
        arg_switch_time,
        arg_most_name,
        arg_location,
    ):
        """Calculate and plot standard metrics."""

        _ = conftest_mock_save_figure
        test_class = scintillometry.metrics.calculations.MetricsWorkflow()
        test_data = {
            "bls": conftest_mock_bls_dataframe_tz.copy(deep=True),
            "weather": conftest_mock_weather_dataframe_tz.copy(deep=True),
            "transect": conftest_mock_transect_dataframe.copy(deep=True),
            "interpolated": conftest_mock_merged_dataframe.copy(deep=True),
            "timestamp": conftest_mock_bls_dataframe_tz.index[0],
        }

        test_args = argparse.Namespace(
            regime=arg_regime,
            switch_time=arg_switch_time,
            beam_wavelength=880,
            beam_error=20,
            most_name=arg_most_name,
            location=arg_location,
        )

        compare_data = test_class.calculate_standard_metrics(
            arguments=test_args, data=test_data
        )
        plt.close("all")

        assert isinstance(compare_data, dict)
        for key in ["derivation", "iteration"]:
            assert key in compare_data
            assert isinstance(compare_data[key], pd.DataFrame)

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
            f"{test_location}, 03 June 2020",
        )
        assert compare_obukhov.gca().get_title() == "".join(test_iter_title)

        test_comp_title = (
            "Sensible Heat Flux from Scintillometer and InnFLUX",
            f"{test_location}, 03 June 2020",
        )
        assert compare_shf.gca().get_title() == "".join(test_comp_title)

        plt.close("all")
