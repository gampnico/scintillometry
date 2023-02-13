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

Tests path weighting module.

Any test that creates a plot should be explicitly appended with
`plt.close()` if the test scope is outside the function, otherwise the
plots remain open in memory.
"""

import datetime

import matplotlib.pyplot as plt
import pandas as pd
import pytest
import pytz

import scintillometry.visuals.plotting


class TestVisualsFormatting:
    """Tests figure and axis formatting."""

    @pytest.mark.dependency(name="TestVisualsFormatting::test_label_selector")
    @pytest.mark.parametrize(
        "arg_label",
        [
            ["shf", ("Sensible Heat Flux", r"$Q_{H}$", r"[W$\cdot$m$^{-2}$]")],
            ["SHF", ("Sensible Heat Flux", r"$Q_{H}$", r"[W$\cdot$m$^{-2}$]")],
            ["missing key", ("Missing Key", r"$missing key$", "")],
        ],
    )
    def test_label_selector(self, arg_label):
        """Construct axis label and title from dependent variable name."""

        test_label = scintillometry.visuals.plotting.label_selector(
            dependent=arg_label[0]
        )

        assert isinstance(test_label, tuple)
        assert all(isinstance(label, str) for label in test_label)
        assert len(test_label) == 3
        for i in range(0, 3):
            assert test_label[i] == arg_label[1][i]

    @pytest.mark.dependency(name="TestVisualsFormatting::test_get_date_and_timezone")
    def test_get_date_and_timezone(self, conftest_mock_bls_dataframe_tz):
        """Get start date and timezone from dataframe."""

        test_data = conftest_mock_bls_dataframe_tz
        compare_times = scintillometry.visuals.plotting.get_date_and_timezone(
            data=test_data
        )
        assert isinstance(compare_times, dict)
        assert all(key in compare_times for key in ("date", "tzone"))

        assert isinstance(compare_times["date"], str)
        assert compare_times["date"] == "03 June 2020"

        assert isinstance(compare_times["tzone"], datetime.tzinfo)
        assert compare_times["tzone"].zone == "CET"

    @pytest.mark.dependency(name="TestVisualsFormatting::test_title_plot")
    @pytest.mark.parametrize("arg_location", ["", "Test Location", None])
    def test_title_plot(self, arg_location):
        """Construct title and legend."""

        test_fig = plt.figure(figsize=(26, 6))
        test_title = r"Test Title $X_{sub}$"
        test_date = "03 June 2020"

        compare_title = scintillometry.visuals.plotting.title_plot(
            title=test_title, timestamp=test_date, location=arg_location
        )

        assert isinstance(compare_title, str)
        assert compare_title[:20] == test_title
        assert compare_title[-12:] == test_date
        if arg_location:
            location_idx = -14 - len(arg_location)
            assert compare_title[location_idx:-14] == arg_location
        else:
            assert not arg_location
        assert test_fig.legend
        plt.close()

    @pytest.mark.dependency(
        name="TestVisualsFormatting::test_set_xy_labels",
        depends=["TestVisualsFormatting::test_label_selector"],
        scope="class",
    )
    @pytest.mark.parametrize("arg_name", ["shf", "test variable"])
    def test_set_xy_labels(self, arg_name):
        """Construct title and legend."""

        plt.figure(figsize=(26, 6))
        test_timezone = pytz.timezone(zone="CET")
        test_axis = plt.gca()
        compare_axis = scintillometry.visuals.plotting.set_xy_labels(
            ax=test_axis, timezone=test_timezone, name=arg_name
        )

        assert isinstance(compare_axis, plt.Axes)
        assert compare_axis.xaxis.get_label().get_text() == "Time, CET"

        compare_name = compare_axis.yaxis.get_label().get_text()
        if arg_name != "shf":
            assert compare_name == arg_name.title()
        else:
            assert compare_name == r"Sensible Heat Flux, [W$\cdot$m$^{-2}$]"
        plt.close()


class TestVisualsPlotting:
    """Tests time series plots."""

    @pytest.mark.dependency(
        name="TestVisualsPlotting::test_setup_plot_data",
        depends=["TestVisualsFormatting::test_get_date_and_timezone"],
        scope="module",
    )
    @pytest.mark.parametrize("arg_names", [["H_convection"], None])
    def test_setup_plot_data(self, conftest_mock_bls_dataframe_tz, arg_names):
        """Setup data for plotting."""

        test_data = conftest_mock_bls_dataframe_tz
        assert conftest_mock_bls_dataframe_tz.index.tz.zone == "CET"

        (
            compare_data,
            compare_mean,
            compare_time,
        ) = scintillometry.visuals.plotting.setup_plot_data(
            df=test_data, names=arg_names
        )

        for dataframe in [compare_data, compare_mean]:
            assert isinstance(dataframe, pd.DataFrame)
            assert dataframe.index.tz.zone == "CET"
            if arg_names:
                assert all(name in dataframe.columns for name in arg_names)

        assert isinstance(compare_time, dict)

    @pytest.mark.dependency(name="TestVisualsPlotting::test_plot_time_series")
    @pytest.mark.parametrize("arg_name", ["Time Series", "", "CT2"])
    @pytest.mark.parametrize("arg_colour", ["black", "red"])
    @pytest.mark.parametrize("arg_mean", [True, False])
    @pytest.mark.parametrize("arg_grey", [True, False])
    def test_plot_time_series(
        self, conftest_mock_bls_dataframe_tz, arg_name, arg_mean, arg_colour, arg_grey
    ):
        """Plot time series and mean."""

        test_data = conftest_mock_bls_dataframe_tz.copy(deep=True)
        if arg_mean:
            test_mean = test_data.dropna().resample("H", origin="start_day").mean()
            test_mean["CT2"] = (
                test_data["CT2"].dropna().resample("H", origin="start_day").mean()
            )

        else:
            test_mean = None
        test_fig = plt.figure(figsize=(6, 6))
        assert isinstance(test_fig, plt.Figure)
        scintillometry.visuals.plotting.plot_time_series(
            series_data=test_data["CT2"],
            series_mean=test_mean,
            name=arg_name,
            line_colour=arg_colour,
            grey=arg_grey,
        )
        plt.legend()
        compare_axes = test_fig.get_axes()
        for ax in compare_axes:
            assert ax.xaxis.label.get_text() == "time"
            compare_legend = ax.get_legend()
            for line in ax.get_lines():
                if not arg_grey:
                    assert line.get_color() == arg_colour
                else:
                    assert line.get_color() == "grey"
                if compare_legend and arg_name:
                    compare_idx = ax.get_lines().index(line)
                    assert compare_idx == 0
                    assert compare_legend.texts[0].get_text() == arg_name

        plt.close()  # otherwise the plots are kept in memory

    @pytest.mark.dependency(
        name="TestVisualsPlotting::test_plot_generic",
        depends=[
            "TestVisualsFormatting::test_title_plot",
            "TestVisualsFormatting::test_set_xy_labels",
            "TestVisualsPlotting::test_setup_plot_data",
            "TestVisualsPlotting::test_plot_time_series",
        ],
        scope="module",
    )
    def test_plot_generic(self, conftest_mock_bls_dataframe_tz):
        """Plot time series of variable."""

        test_data = conftest_mock_bls_dataframe_tz

        compare_fig, compare_ax = scintillometry.visuals.plotting.plot_generic(
            dataframe=test_data, name="pressure", site="Test"
        )

        assert isinstance(compare_fig, plt.Figure)
        assert isinstance(compare_ax, plt.Axes)
        assert compare_ax.xaxis.label.get_text() == "Time, CET"
        assert compare_ax.yaxis.label.get_text() == "Pressure, [mbar]"
        assert compare_ax.get_title() == "Pressure at Test, 03 June 2020"
        plt.close()

    @pytest.mark.dependency(
        name="TestVisualsPlotting::test_plot_convection",
        depends=[
            "TestVisualsFormatting::test_title_plot",
            "TestVisualsFormatting::test_set_xy_labels",
            "TestVisualsPlotting::test_setup_plot_data",
            "TestVisualsPlotting::test_plot_time_series",
        ],
        scope="module",
    )
    @pytest.mark.parametrize("arg_stability", ["unstable", None])
    def test_plot_convection(self, conftest_mock_bls_dataframe_tz, arg_stability):
        """Plot SHFs for scintillometer and free convection."""

        test_data = conftest_mock_bls_dataframe_tz
        test_data["H_free"] = pd.Series([4.4, 5.5], index=test_data.index)

        compare_fig, compare_ax = scintillometry.visuals.plotting.plot_convection(
            dataframe=test_data, stability=arg_stability
        )
        assert isinstance(compare_fig, plt.Figure)
        assert isinstance(compare_ax, plt.Axes)

        assert compare_ax.xaxis.label.get_text() == "Time, CET"
        assert (
            compare_ax.yaxis.label.get_text()
            == r"Sensible Heat Flux, [W$\cdot$m$^{-2}$]"
        )
        if arg_stability:
            compare_conditions = f"{arg_stability.capitalize()} Conditions"
        else:
            compare_conditions = "No Height Dependency"
        compare_title = (
            "Sensible Heat Fluxes from On-Board Software and",
            f"for Free Convection ({compare_conditions}), 03 June 2020",
        )
        assert compare_ax.get_title() == " ".join(compare_title)
        plt.close()

    @pytest.mark.dependency(
        name="TestVisualsPlotting::test_plot_comparison",
        depends=[
            "TestVisualsFormatting::test_title_plot",
            "TestVisualsFormatting::test_set_xy_labels",
            "TestVisualsPlotting::test_setup_plot_data",
            "TestVisualsPlotting::test_plot_time_series",
        ],
        scope="module",
    )
    @pytest.mark.parametrize(
        "arg_keys", [["wind_speed", "rho_air"], ["wind_speed", "wind_speed"]]
    )
    @pytest.mark.parametrize("arg_site", ["", "Test Location"])
    def test_plot_comparison(self, conftest_mock_merged_dataframe, arg_keys, arg_site):
        """Plot comparison between two dataframes."""

        test_data_01 = conftest_mock_merged_dataframe
        test_data_02 = conftest_mock_merged_dataframe
        if "rho_air" in arg_keys:
            test_labels = ["Wind Speed", "Air Density"]
        else:
            test_labels = ["Wind Speed", "Wind Speed"]
        compare_fig, compare_ax = scintillometry.visuals.plotting.plot_comparison(
            df_01=test_data_01,
            df_02=test_data_02,
            keys=arg_keys,
            labels=["Test 01", "Test 02"],
            site=arg_site,
        )
        assert isinstance(compare_fig, plt.Figure)
        assert isinstance(compare_ax, plt.Axes)

        if arg_site:
            test_site = f" at {arg_site}"
        else:
            test_site = ""
        test_title_label = f"{test_labels[0]}"
        if test_labels[1] != test_labels[0]:
            test_title_label = f"{test_title_label} and {test_labels[1]}"
        test_title = (
            f"{test_title_label} from Test 01 and Test 02{test_site}, 03 June 2020"
        )
        assert compare_ax.get_title() == test_title
        plt.close()

    @pytest.mark.dependency(
        name="TestVisualsPlotting::test_plot_iterated_fluxes",
        depends=[
            "TestVisualsFormatting::test_title_plot",
            "TestVisualsFormatting::test_set_xy_labels",
            "TestVisualsPlotting::test_setup_plot_data",
            "TestVisualsPlotting::test_plot_time_series",
            "TestVisualsPlotting::test_plot_generic",
            "TestVisualsPlotting::test_plot_comparison",
        ],
        scope="module",
    )
    @pytest.mark.parametrize("arg_location", ["", "Test Location"])
    def test_plot_iterated_fluxes(
        self,
        conftest_mock_iterated_dataframe,
        conftest_mock_save_figure,
        arg_location,
    ):
        """Plot and save iterated fluxes."""

        _ = conftest_mock_save_figure  # otherwise figure is saved to disk
        timestamp = conftest_mock_iterated_dataframe.index[0]
        (
            compare_shf,
            compare_comp,
        ) = scintillometry.visuals.plotting.plot_iterated_fluxes(
            bls_data=conftest_mock_iterated_dataframe,
            iteration_data=conftest_mock_iterated_dataframe,
            time_id=timestamp,
            location=arg_location,
        )
        assert isinstance(compare_shf, plt.Figure)
        assert isinstance(compare_comp, plt.Figure)

        if arg_location:
            test_location = f" at {arg_location}"
        else:
            test_location = ""
        compare_ax = plt.gca()
        test_title = (
            "Sensible Heat Flux from Free Convection and Iterated Flux",
            f"{test_location}, 03 June 2020",
        )
        assert compare_ax.get_title() == "".join(test_title)
        plt.close()
