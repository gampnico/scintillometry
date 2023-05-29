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

Tests plotting module.

Any test that creates a plot should be explicitly appended with
`plt.close("all")`, otherwise the plots remain open in memory.
"""

import datetime

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import pytz

import scintillometry.visuals.plotting


class TestVisualsBoilerplate:
    """Boilerplate methods for testing plots."""

    def assert_constant_lines(
        self, ax: plt.Axes, hlines: dict = None, vlines: dict = None
    ):
        """Assert horizontal or vertical lines exist in plot."""

        test_lines = [hlines, vlines]
        compare_ax_properties = ax.properties()
        ax.legend()
        for lines in test_lines:
            if lines:
                compare_labels = []
                for compare_lines in compare_ax_properties["lines"]:
                    line_label = compare_lines.properties()["label"]
                    compare_labels.append(line_label)
                    if line_label.lower() in lines.keys():
                        assert [
                            lines[line_label.lower()],
                            lines[line_label.lower()],
                        ] in compare_lines.properties()["data"]

                for key in lines.keys():
                    if not lines[key]:  # if line data is None
                        assert key.title() not in compare_labels
                    else:
                        assert key.title() in compare_labels

    @pytest.mark.dependency(name="TestVisualsBoilerplate::test_assert_constant_lines")
    @pytest.mark.parametrize("arg_vlines", [{"v_a": 1}, {"v_a": 1, "v_b": 2.1}, None])
    @pytest.mark.parametrize("arg_hlines", [{"h_a": 1}, {"h_a": 1, "h_b": 2.1}, None])
    def test_assert_constant_lines(
        self, conftest_generate_series, arg_hlines, arg_vlines
    ):
        """Validate test for constant lines existing on axis."""
        test_data, test_index = conftest_generate_series
        test_series = pd.Series(data=test_data, index=test_index)

        plt.figure(figsize=(10, 10))
        plt.plot(test_index, test_series)
        if arg_vlines:
            for vline in arg_vlines:
                plt.axvline(arg_vlines[vline], label=vline.title())
        if arg_hlines:
            for hline in arg_hlines:
                plt.axhline(arg_hlines[hline], label=hline.title())
        compare_ax = plt.gca()
        self.assert_constant_lines(ax=compare_ax, hlines=arg_hlines, vlines=arg_vlines)
        plt.close("all")


class TestVisualsFormatting(TestVisualsBoilerplate):
    """Tests figure and axis formatting.

    Attributes:
        test_format (FigureFormat): Figure formatting class.
        test_location (str): Placeholder for location.
        test_date (str): Placeholder for date of data collection.
    """

    test_format = scintillometry.visuals.plotting.FigureFormat()
    test_location = "Test Location"
    test_date = "03 June 2020"

    @pytest.mark.dependency(
        name="TestVisualsFormatting::test_initialise_formatting", scope="function"
    )
    @pytest.mark.parametrize("arg_offset", [1, 0])
    def test_initialise_formatting(self, arg_offset):
        """Initialise font sizes for matplotlib figures."""

        plt.rcdefaults()
        test_small = 14 + arg_offset
        test_medium = 16 + arg_offset
        test_large = 18 + arg_offset
        test_extra = 20 + arg_offset
        test_params = {
            "font.size": test_small,
            "xtick.labelsize": test_small,
            "ytick.labelsize": test_small,
            "legend.fontsize": test_small,
            "axes.labelsize": test_medium,
            "axes.titlesize": test_large,
            "figure.titlesize": test_extra,
        }

        self.test_format.initialise_formatting(
            small=test_small, medium=test_medium, large=test_large, extra=test_extra
        )
        for param, font_size in test_params.items():
            assert plt.rcParams[param] == font_size
        plt.rcdefaults()
        plt.close("all")

    @pytest.mark.dependency(name="TestVisualsFormatting::test_parse_formatting_kwargs")
    @pytest.mark.parametrize("arg_fig", [True, False])
    def test_parse_formatting_kwargs(self, arg_fig, conftest_generate_series):
        """Parse kwargs when formatting."""

        if arg_fig:
            test_data, test_index = conftest_generate_series
            test_series = pd.Series(data=test_data, index=test_index)
            plt.figure(figsize=(10, 10))
            plt.plot(test_index, test_series)

        test_kwargs = {
            "hlines": {"h_a": 1},
            "vlines": {"v_a": 1},
            "title": "Test Title",
            "y_label": "Test Y-label",
        }

        compare_ax = plt.gca()
        self.test_format.parse_formatting_kwargs(axis=compare_ax, **test_kwargs)

        self.assert_constant_lines(
            ax=compare_ax,
            hlines=test_kwargs["hlines"],
            vlines=test_kwargs["vlines"],
        )
        assert compare_ax.yaxis.get_label_text() == "Test Y-label"
        assert compare_ax.get_title() == "Test Title"
        plt.close("all")

    @pytest.mark.dependency(name="TestVisualsFormatting::test_get_site_name")
    @pytest.mark.parametrize("arg_string", ["", 3, "Test Location"])
    @pytest.mark.parametrize("arg_dataframe", [None, True, "name", "wrong_key"])
    def test_get_site_name(
        self, conftest_mock_bls_dataframe_tz, arg_string, arg_dataframe
    ):
        """Gets name of site from user string or metadata."""

        if arg_dataframe:
            test_frame = conftest_mock_bls_dataframe_tz.copy(deep=True)
            if isinstance(arg_dataframe, str):
                test_frame.attrs[arg_dataframe] = self.test_location
        else:
            test_frame = None

        compare_label = self.test_format.get_site_name(
            site_name=arg_string, dataframe=test_frame
        )
        assert isinstance(compare_label, str)

        if arg_string:
            assert compare_label == str(arg_string)
        elif isinstance(arg_dataframe, str):
            if arg_dataframe == "name":
                assert compare_label == self.test_location
            else:
                assert compare_label == ""
        else:
            assert compare_label == ""

    @pytest.mark.dependency(name="TestVisualsFormatting::test_label_selector")
    @pytest.mark.parametrize(
        "arg_label",
        [
            ["shf", ("Sensible Heat Flux", r"$Q_{H}$", r"W$\cdot$m$^{-2}$")],
            ["SHF", ("Sensible Heat Flux", r"$Q_{H}$", r"W$\cdot$m$^{-2}$")],
            ["missing key", ("Missing Key", r"$missing key$", "")],
        ],
    )
    def test_label_selector(self, arg_label):
        """Construct axis label and title from dependent variable name."""

        test_label = self.test_format.label_selector(dependent=arg_label[0])

        assert isinstance(test_label, tuple)
        assert all(isinstance(label, str) for label in test_label)
        assert len(test_label) == 3
        for i in range(0, 3):
            assert test_label[i] == arg_label[1][i]

    @pytest.mark.dependency(name="TestVisualsFormatting::test_get_date_and_timezone")
    def test_get_date_and_timezone(self, conftest_mock_bls_dataframe_tz):
        """Get start date and timezone from dataframe."""

        test_data = conftest_mock_bls_dataframe_tz.copy(deep=True)
        compare_times = self.test_format.get_date_and_timezone(data=test_data)
        assert isinstance(compare_times, dict)
        assert all(key in compare_times for key in ("date", "tzone"))

        assert isinstance(compare_times["date"], str)
        assert compare_times["date"] == self.test_date

        assert isinstance(compare_times["tzone"], datetime.tzinfo)
        assert compare_times["tzone"].zone == "CET"

    @pytest.mark.dependency(name="TestVisualsFormatting::test_title_plot")
    @pytest.mark.parametrize("arg_location", ["", "Test Location", None])
    @pytest.mark.parametrize("arg_string", [True, False])
    def test_title_plot(self, arg_location, arg_string):
        """Construct title and legend."""

        test_fig = plt.figure(figsize=(26, 6))
        test_title = r"Test Title $X_{sub}$"
        if not arg_string:
            test_date = pd.to_datetime(self.test_date)
        else:
            test_date = self.test_date

        compare_title = self.test_format.title_plot(
            title=test_title, timestamp=test_date, location=arg_location
        )

        assert isinstance(compare_title, str)
        assert compare_title[:20] == test_title
        assert compare_title[-12:] == self.test_date
        if arg_location:
            location_idx = -14 - len(arg_location)
            assert compare_title[location_idx:-14] == arg_location
        else:
            assert not arg_location
        assert test_fig.legend
        plt.close("all")

    @pytest.mark.dependency(name="TestVisualsFormatting::test_merge_label_with_unit")
    @pytest.mark.parametrize("arg_unit", ["m", "", r"m$\theta$", r"m$^{-1}$"])
    def test_merge_label_with_unit(self, arg_unit):
        """Merge name and unit strings."""

        test_label = ("Variable Name", r"$V$", arg_unit)
        compare_label = self.test_format.merge_label_with_unit(label=test_label)

        assert isinstance(compare_label, str)
        assert test_label[1] not in compare_label
        if not arg_unit:
            assert compare_label == "Variable Name"
        else:
            assert compare_label == f"Variable Name, [{arg_unit}]"

    @pytest.mark.dependency(
        name="TestVisualsFormatting::test_merge_merge_multiple_labels"
    )
    @pytest.mark.parametrize("arg_labels", [1, 2, 3])
    def test_merge_multiple_labels(self, arg_labels):
        """Merge and format multiple labels."""

        test_labels = []
        for i in range(arg_labels):
            test_labels.append(f"{i}")
        compare_label = self.test_format.merge_multiple_labels(labels=test_labels)

        assert isinstance(compare_label, str)
        for label in test_labels:
            assert label in compare_label
        if arg_labels == 1:
            assert compare_label == test_labels[0]
        elif arg_labels == 2:
            test_merged = "0 and 1"
            assert compare_label == test_merged
        else:
            test_merged = "0, 1, 2"
            assert compare_label == test_merged

    @pytest.mark.dependency(
        name="TestVisualsFormatting::test_set_xy_labels",
        depends=[
            "TestVisualsFormatting::test_label_selector",
            "TestVisualsFormatting::test_merge_label_with_unit",
        ],
    )
    @pytest.mark.parametrize("arg_name", ["shf", "test variable"])
    def test_set_xy_labels(self, arg_name):
        """Construct title and legend."""

        plt.figure(figsize=(26, 6))
        test_timezone = pytz.timezone(zone="CET")
        test_axis = plt.gca()
        test_formatter = matplotlib.dates.DateFormatter("%H:%M", test_timezone)
        compare_axis = self.test_format.set_xy_labels(
            ax=test_axis, timezone=test_timezone, name=arg_name
        )

        assert isinstance(compare_axis, plt.Axes)
        assert compare_axis.xaxis.get_label_text() == "Time, CET"

        compare_name = compare_axis.yaxis.get_label_text()
        if arg_name != "shf":
            assert compare_name == arg_name.title()
        else:
            assert compare_name == r"Sensible Heat Flux, [W$\cdot$m$^{-2}$]"

        compare_formatter = compare_axis.xaxis.get_major_formatter()
        assert isinstance(compare_formatter, matplotlib.dates.DateFormatter)
        assert compare_formatter.tz == test_formatter.tz
        assert compare_formatter.fmt == test_formatter.fmt
        plt.close("all")

    @pytest.mark.dependency(
        name="TestVisualsFormatting::test_plot_constant_lines",
        depends=[
            "TestVisualsBoilerplate::test_assert_constant_lines",
            "TestVisualsFormatting::test_label_selector",
        ],
    )
    @pytest.mark.parametrize("arg_vlines", [{"va": None}, {"va": 1, "vb": 2.1}, None])
    @pytest.mark.parametrize("arg_hlines", [{"ha": None}, {"ha": 1, "hb": 2.1}, None])
    def test_plot_constant_lines(
        self, conftest_generate_series, arg_hlines, arg_vlines
    ):
        """Plot horizontal and vertical lines."""

        test_data, test_index = conftest_generate_series
        test_series = pd.Series(data=test_data, index=test_index)

        plt.figure(figsize=(10, 10))
        plt.plot(test_index, test_series)
        plt.legend()
        test_ax = plt.gca()
        self.test_format.plot_constant_lines(
            axis=test_ax, hlines=arg_hlines, vlines=arg_vlines
        )
        self.assert_constant_lines(ax=test_ax, hlines=arg_hlines, vlines=arg_vlines)

        plt.close("all")


class TestVisualsPlotting(TestVisualsBoilerplate):
    """Tests time series plots.

    Attributes:
        test_plotting (FigurePlotter): Figure plotting class.
        test_location (str): Placeholder for location.
        test_date (str): Placeholder for date of data collection.
        test_timestamp (pd.Timestamp): Placeholder for index timestamp.
    """

    test_plotting = scintillometry.visuals.plotting.FigurePlotter()
    test_location = "Test Location"
    test_date = "03 June 2020"
    test_timestamp = pd.Timestamp(f"{test_date} 05:20", tz="CET")

    def test_visuals_plotting_attributes(self):
        assert isinstance(self.test_timestamp, pd.Timestamp)
        assert self.test_timestamp.strftime("%Y-%m-%d %H:%M") == "2020-06-03 05:20"
        assert self.test_timestamp.tz.zone == "CET"

    @pytest.mark.dependency(
        name="TestVisualsPlotting::test_setup_plot_data",
        depends=["TestVisualsFormatting::test_get_date_and_timezone"],
    )
    @pytest.mark.parametrize("arg_names", [["H_convection"], None])
    def test_setup_plot_data(
        self, conftest_mock_bls_dataframe_tz, conftest_boilerplate, arg_names
    ):
        """Setup data for plotting."""

        test_data = conftest_mock_bls_dataframe_tz.copy(deep=True)
        conftest_boilerplate.check_timezone(dataframe=test_data, tzone="CET")

        (
            compare_data,
            compare_mean,
            compare_time,
        ) = self.test_plotting.setup_plot_data(data=test_data, names=arg_names)

        for dataframe in [compare_data, compare_mean]:
            conftest_boilerplate.check_dataframe(dataframe=dataframe)
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
        self.test_plotting.plot_time_series(
            series_data=test_data["CT2"],
            series_mean=test_mean,
            name=arg_name,
            line_colour=arg_colour,
            grey=arg_grey,
        )
        plt.legend()
        compare_axes = test_fig.get_axes()
        for ax in compare_axes:
            assert ax.xaxis.get_label_text() == "time"
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

        plt.close("all")  # otherwise the plots are kept in memory

    @pytest.mark.dependency(
        name="TestVisualsPlotting::test_plot_generic",
        depends=[
            "TestVisualsFormatting::test_title_plot",
            "TestVisualsFormatting::test_set_xy_labels",
            "TestVisualsFormatting::test_get_site_name",
            "TestVisualsPlotting::test_setup_plot_data",
            "TestVisualsPlotting::test_plot_time_series",
        ],
    )
    def test_plot_generic(self, conftest_mock_bls_dataframe_tz, conftest_boilerplate):
        """Plot time series of variable."""

        test_data = conftest_mock_bls_dataframe_tz.copy(deep=True)
        test_title = f"{self.test_location},\n{self.test_date}"

        compare_fig, compare_ax = self.test_plotting.plot_generic(
            dataframe=test_data, name="pressure", site=self.test_location
        )
        compare_params = {
            "plot": (compare_fig, compare_ax),
            "x_label": "Time, CET",
            "y_label": "Pressure, [mbar]",
            "title": "Pressure at ",
        }
        conftest_boilerplate.check_plot(plot_params=compare_params, title=test_title)

        plt.close("all")

    @pytest.mark.dependency(
        name="TestVisualsPlotting::test_plot_convection",
        depends=[
            "TestVisualsFormatting::test_title_plot",
            "TestVisualsFormatting::test_set_xy_labels",
            "TestVisualsFormatting::test_get_site_name",
            "TestVisualsPlotting::test_setup_plot_data",
            "TestVisualsPlotting::test_plot_time_series",
        ],
    )
    @pytest.mark.parametrize("arg_stability", ["unstable", None])
    def test_plot_convection(
        self, conftest_mock_bls_dataframe_tz, conftest_boilerplate, arg_stability
    ):
        """Plot SHFs for scintillometer and free convection."""

        test_data = conftest_mock_bls_dataframe_tz.copy(deep=True)
        test_data["H_free"] = pd.Series([4.4, 5.5], index=test_data.index)
        if arg_stability:
            test_conditions = f"{arg_stability.capitalize()} Conditions"
        else:
            test_conditions = "No Height Dependency"
        test_title = (
            "Sensible Heat Fluxes from On-Board Software and for Free Convection",
            f"({test_conditions}),\n{self.test_date}",
        )

        compare_fig, compare_ax = self.test_plotting.plot_convection(
            dataframe=test_data, stability=arg_stability
        )
        compare_params = {
            "plot": (compare_fig, compare_ax),
            "x_label": "Time, CET",
            "y_label": r"Sensible Heat Flux, [W$\cdot$m$^{-2}$]",
            "title": " ".join(test_title),
        }
        conftest_boilerplate.check_plot(plot_params=compare_params)

        plt.close("all")

    @pytest.mark.dependency(
        name="TestVisualsPlotting::test_plot_comparison",
        depends=[
            "TestVisualsFormatting::test_title_plot",
            "TestVisualsFormatting::test_set_xy_labels",
            "TestVisualsFormatting::test_get_site_name",
            "TestVisualsPlotting::test_setup_plot_data",
            "TestVisualsPlotting::test_plot_time_series",
        ],
    )
    @pytest.mark.parametrize(
        "arg_keys", [["wind_speed", "rho_air"], ["wind_speed", "wind_speed"]]
    )
    @pytest.mark.parametrize("arg_site", ["", "Test Location"])
    def test_plot_comparison(
        self, conftest_mock_merged_dataframe, conftest_boilerplate, arg_keys, arg_site
    ):
        """Plot comparison between two dataframes."""

        test_data_01 = conftest_mock_merged_dataframe.copy(deep=True)
        test_data_02 = conftest_mock_merged_dataframe.copy(deep=True)
        if "rho_air" in arg_keys:
            test_labels = ["Wind Speed", "Air Density"]
        else:
            test_labels = ["Wind Speed", "Wind Speed"]
        if arg_site:
            test_site = f" at {arg_site}"
        else:
            test_site = ""
        test_title_label = f"{test_labels[0]}"
        if test_labels[1] != test_labels[0]:
            test_title_label = f"{test_title_label} and {test_labels[1]}"
        test_title = (
            f"{test_title_label} from Test 01 and Test 02{test_site},\n{self.test_date}"
        )

        compare_fig, compare_ax = self.test_plotting.plot_comparison(
            df_01=test_data_01,
            df_02=test_data_02,
            keys=arg_keys,
            labels=["Test 01", "Test 02"],
            site=arg_site,
        )
        compare_params = {
            "plot": (compare_fig, compare_ax),
            "x_label": "Time, CET",
            "title": test_title,
        }
        conftest_boilerplate.check_plot(plot_params=compare_params)

        plt.close("all")

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
    )
    @pytest.mark.parametrize("arg_location", ["", "Test Location"])
    def test_plot_iterated_fluxes(
        self,
        conftest_mock_iterated_dataframe,
        conftest_mock_save_figure,
        conftest_boilerplate,
        arg_location,
    ):
        """Plot and save iterated fluxes."""

        _ = conftest_mock_save_figure  # otherwise figure is saved to disk

        test_data = conftest_mock_iterated_dataframe.copy(deep=True)
        if arg_location:
            test_location = f" at {arg_location}"
        else:
            test_location = ""
        test_title = f"{test_location},\n{self.test_date}"
        timestamp = test_data.index[0]

        with pytest.deprecated_call():
            compare_plots = self.test_plotting.plot_iterated_fluxes(
                iteration_data=test_data,
                time_id=timestamp,
                location=arg_location,
            )
        assert isinstance(compare_plots, list)
        assert all(isinstance(compare_tuple, tuple) for compare_tuple in compare_plots)

        compare_plots = {
            "shf": {
                "title": "Sensible Heat Flux",
                "y_label": r"Sensible Heat Flux, [W$\cdot$m$^{-2}$]",
                "x_label": "Time, CET",
                "plot": (compare_plots[0]),
            },
            "comparison": {
                "title": "Sensible Heat Flux from Free Convection and Iteration",
                "y_label": r"Sensible Heat Flux, [W$\cdot$m$^{-2}$]",
                "x_label": "Time, CET",
                "plot": (compare_plots[1]),
            },
        }

        for params in compare_plots.values():
            conftest_boilerplate.check_plot(plot_params=params, title=test_title)

        plt.close("all")

    @pytest.mark.dependency(
        name="TestVisualsPlotting::test_plot_innflux",
        depends=[
            "TestVisualsFormatting::test_label_selector",
            "TestVisualsFormatting::test_title_plot",
            "TestVisualsFormatting::test_set_xy_labels",
            "TestVisualsFormatting::test_get_site_name",
            "TestVisualsPlotting::test_setup_plot_data",
        ],
    )
    @pytest.mark.parametrize("arg_site", ["", "Test Location", None])
    def test_plot_innflux(
        self,
        conftest_mock_iterated_dataframe,
        conftest_mock_innflux_dataframe_tz,
        conftest_boilerplate,
        arg_site,
    ):
        """Plots comparison with InnFLUX data."""

        test_iterated = conftest_mock_iterated_dataframe.copy(deep=True)
        test_innflux = conftest_mock_innflux_dataframe_tz.copy(deep=True)

        if arg_site:
            test_site = f" at {arg_site}"
        else:
            test_site = ""
        test_title = f"{test_site},\n{self.test_date}"

        fig_obukhov, ax_obukhov = self.test_plotting.plot_innflux(
            iter_data=test_iterated,
            innflux_data=test_innflux,
            name="obukhov",
            site=arg_site,
        )
        fig_shf, ax_shf = self.test_plotting.plot_innflux(
            iter_data=test_iterated,
            innflux_data=test_innflux,
            name="shf",
            site=arg_site,
        )

        compare_plots = {
            "obukhov": {
                "plot": (fig_obukhov, ax_obukhov),
                "x_label": "Time, CET",
                "y_label": "Obukhov Length, [m]",
                "title": "Obukhov Length from Scintillometer and innFLUX",
            },
            "shf": {
                "plot": (fig_shf, ax_shf),
                "x_label": "Time, CET",
                "y_label": r"Sensible Heat Flux, [W$\cdot$m$^{-2}$]",
                "title": "Sensible Heat Flux from Scintillometer and innFLUX",
            },
        }
        for params in compare_plots.values():
            conftest_boilerplate.check_plot(plot_params=params, title=test_title)

        plt.close("all")

    @pytest.mark.dependency(name="TestVisualsPlotting::test_plot_scatter")
    @pytest.mark.parametrize("arg_site", ["", "Test Location", None])
    @pytest.mark.parametrize("arg_score", [None, 0.561734521])
    @pytest.mark.parametrize("arg_regression", [True, False])
    def test_plot_scatter(
        self,
        conftest_boilerplate,
        conftest_generate_series,
        arg_score,
        arg_site,
        arg_regression,
    ):
        """Plot scatter between two datasets with regression line."""

        test_name = "obukhov"
        test_data, test_index = conftest_generate_series

        test_x = pd.Series(name=test_name, data=test_data, index=test_index)
        test_y = pd.Series(name=test_name, data=test_data + 0.5, index=test_index)
        for series in [test_x, test_y]:
            assert isinstance(series, pd.Series)
            assert series.shape == (100,)
            assert not (series.isnull()).any()
        if arg_site:
            test_site = f" at {arg_site},"
        else:
            test_site = ","
        test_title = (
            "Obukhov Length Regression Between",
            f"Baseline and Comparison{test_site}",
            f"{self.test_date}",
        )
        if not arg_regression:
            test_line = None
        else:
            test_line = np.arange(0, 1000, 10)

        compare_fig, compare_ax = self.test_plotting.plot_scatter(
            x_data=test_x,
            y_data=test_y,
            name=test_name,
            sources=["Baseline", "Comparison"],
            site=arg_site,
            score=arg_score,
            regression_line=test_line,
        )
        compare_params = {
            "plot": (compare_fig, compare_ax),
            "x_label": "Obukhov Length, [m] (Baseline)",
            "y_label": "Obukhov Length, [m] (Comparison)",
            "title": "\n".join(test_title),
        }
        conftest_boilerplate.check_plot(plot_params=compare_params)

        if arg_regression:
            _, labels = compare_params["plot"][1].get_legend_handles_labels()
            assert "Line of Best Fit" in labels

        if arg_score is not None:
            assert (
                compare_params["plot"][1].texts[0].get_text()
                == f"R$^{2}$= {arg_score:.5f}"
            )

        plt.close("all")

    @pytest.mark.dependency(
        name="TestVisualsPlotting::test_plot_vertical_profile",
        depends=["TestVisualsFormatting::test_plot_constant_lines"],
    )
    @pytest.mark.parametrize("arg_site", ["", "Test Location", None])
    @pytest.mark.parametrize("arg_name", ["temperature", "test_variable"])
    @pytest.mark.parametrize("arg_kwargs", [{}, {"hlines": {"test height": 10}}])
    @pytest.mark.parametrize("arg_y_lim", [None, 2000])
    def test_plot_vertical_profile(
        self,
        conftest_mock_hatpro_temperature_dataframe_tz,
        conftest_boilerplate,
        arg_site,
        arg_name,
        arg_y_lim,
        arg_kwargs,
    ):
        """Plots vertical profile at specific time."""

        test_data = {
            arg_name: conftest_mock_hatpro_temperature_dataframe_tz.copy(deep=True)
        }

        if arg_site:
            test_site = f"\nat {arg_site}, "
        else:
            test_site = ",\n"
        test_title = (
            f"Vertical Profile of {arg_name.title()}{test_site}",
            f"{self.test_date} 05:20 CET",
        )
        if arg_name != "temperature":
            test_x_label = arg_name.title()
        else:
            test_x_label = "Temperature, [K]"

        compare_fig, compare_ax = self.test_plotting.plot_vertical_profile(
            vertical_data=test_data,
            time_idx=self.test_timestamp,
            name=arg_name,
            site=arg_site,
            y_lim=arg_y_lim,
            **arg_kwargs,
        )

        compare_params = {
            "plot": (compare_fig, compare_ax),
            "title": "".join(test_title),
            "y_label": "Height [m]",
            "x_label": test_x_label,
        }
        conftest_boilerplate.check_plot(plot_params=compare_params)
        plt.close("all")

    @pytest.mark.dependency(
        name="TestVisualsPlotting::test_plot_vertical_comparison",
        depends=["TestVisualsFormatting::test_label_selector"],
    )
    @pytest.mark.parametrize("arg_site", ["", "Test Location", None])
    @pytest.mark.parametrize("arg_kwargs", [{}, {"hlines": {"test_height": 10}}])
    def test_plot_vertical_comparison(
        self, conftest_mock_hatpro_dataset, arg_site, arg_kwargs
    ):
        """Plots comparison of vertical profiles at specific time."""

        test_data = conftest_mock_hatpro_dataset.copy()

        if arg_site:
            test_site = f"\nat {arg_site}, "
        else:
            test_site = ",\n"
        test_keys = ["temperature", "humidity"]
        test_title = (
            f"Vertical Profiles of Temperature and Humidity{test_site}",
            f"{self.test_date} 05:20 CET",
        )

        compare_fig, compare_ax = self.test_plotting.plot_vertical_comparison(
            dataset=test_data,
            time_index=self.test_timestamp,
            keys=test_keys,
            site=arg_site,
            **arg_kwargs,
        )
        assert isinstance(compare_fig, plt.Figure)
        assert all(isinstance(ax, plt.Axes) for ax in compare_ax)
        assert compare_ax[0].yaxis.get_label_text() == "Height [m]"
        for i in range(len(test_keys)):
            compare_x_label = compare_ax[i].xaxis.get_label_text()
            test_label = self.test_plotting.label_selector(dependent=test_keys[i])
            test_x_label = f"{test_label[0]}, [{test_label[2]}]"
            assert test_x_label in compare_x_label
            if arg_kwargs:
                compare_ax[i].legend()
                self.assert_constant_lines(
                    ax=compare_ax[i], hlines=arg_kwargs["hlines"]
                )
        assert compare_fig.texts[0].get_text() == "".join(test_title)
        plt.close("all")

    @pytest.mark.dependency(
        name="TestVisualsPlotting::test_plot_merged_profiles",
        depends=["TestVisualsFormatting::test_plot_constant_lines"],
    )
    @pytest.mark.parametrize("arg_site", ["", "Test Location", None])
    @pytest.mark.parametrize("arg_kwargs", [{}, {"x_label": "Temperature, T [K]"}])
    @pytest.mark.parametrize("arg_y_lim", [None, 2000])
    def test_plot_merged_profiles(
        self,
        conftest_mock_hatpro_dataset,
        conftest_boilerplate,
        arg_site,
        arg_kwargs,
        arg_y_lim,
    ):
        """Plots vertical profile at specific time."""

        test_data = conftest_mock_hatpro_dataset.copy()
        test_index = self.test_timestamp
        if arg_site:
            test_site = f"\nat {arg_site}, "
        else:
            test_site = ",\n"
        test_title = (
            f"Vertical Profiles of Temperature and Humidity{test_site}",
            f"{self.test_date} 05:20 CET",
        )
        if not arg_kwargs:
            test_x_label = r"Humidity, [kg$\cdot$m$^{-3}$]"
        else:
            test_x_label = "Temperature, T [K]"

        compare_fig, compare_ax = self.test_plotting.plot_merged_profiles(
            dataset=test_data,
            time_index=test_index,
            site=arg_site,
            y_lim=arg_y_lim,
            **arg_kwargs,
        )
        compare_params = {
            "plot": (compare_fig, compare_ax),
            "title": "".join(test_title),
            "x_label": test_x_label,
            "y_label": "Height [m]",
        }
        conftest_boilerplate.check_plot(plot_params=compare_params)
        assert len(compare_ax.properties()["lines"]) == len(test_data.keys())

        plt.close("all")
