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

Automates data plotting.
"""

import math
import os
import re

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scintillometry.backend.deprecations import Decorators


class FigureFormat:
    """Regulates plot annotations and formatting."""

    def __init__(self):
        super().__init__()

    def initialise_formatting(self, small=14, medium=16, large=18, extra=20):
        """Initialises font sizes for matplotlib figures.

        Called separately from other plotting functions in this module
        to avoid overwriting on-the-fly formatting changes.

        Args:
            small (int): Size of text, ticks, legend. Default 14.
            medium (int): Size of axis labels. Default 16.
            large (int): Size of axis title. Default 18.
            extra (int): Size of figure title (suptitle). Default 20.
        """

        plt.rc("font", size=small)  # controls default text sizes
        plt.rc("axes", titlesize=large)  # fontsize of the axes title
        plt.rc("axes", labelsize=medium)  # fontsize of the x and y labels
        plt.rc("xtick", labelsize=small)  # fontsize of the tick labels
        plt.rc("ytick", labelsize=small)  # fontsize of the tick labels
        plt.rc("legend", fontsize=small)  # legend fontsize
        plt.rc("figure", titlesize=extra)  # fontsize of the figure title

    def parse_formatting_kwargs(self, axis, **kwargs):
        """Parses kwargs for generic formatting.

        Applies horizontal or vertical lines, updates axis title and
        labels.

        Args:
            axis (plt.Axes): Target axes.

        Keyword Args:
            hlines (dict): Name and y-axis value for which to plot a
                horizontal line.
            title (str): Axis title.
            vlines (dict): Name and x-axis value for which to plot a
                vertical line.
            x_label (str): X-axis label.
            y_label (str): Y-axis label.
        """

        hlines = kwargs.get("hlines")
        title = kwargs.get("title")
        vlines = kwargs.get("vlines")
        x_label = kwargs.get("x_label")
        y_label = kwargs.get("y_label")

        self.plot_constant_lines(axis=axis, hlines=hlines, vlines=vlines)
        if title:
            plt.title(title, fontweight="bold")
        if x_label:
            plt.xlabel(x_label)
        if y_label:
            plt.ylabel(y_label)

    def get_site_name(self, site_name, dataframe=None):
        """Gets name of site from user string or dataframe metadata.

        Args:
            site_name (str): Location of data collection.
            dataframe (pd.DataFrame or pd.Series): Any collected
                dataset. Default None.

        Returns:
            str: Location of data collection. Returns empty string if no
            user argument or metadata was found.
        """

        if site_name:
            label = site_name
        elif dataframe is None:
            label = ""
        elif dataframe.attrs and ("name" in dataframe.attrs):
            label = dataframe.attrs["name"]
        else:
            label = ""

        if not isinstance(label, str):
            label = str(label)

        return label

    def label_selector(self, dependent):
        """Constructs label parameters from dependent variable name.

        If the variable name is not implemented, the unconverted
        variable is passed directly, with the symbol duplicated from its
        name and with no unit provided.

        Args:
            dependent (str): Name of dependent variable.

        Returns:
            tuple[str, str, str]: Name of dependent variable, symbol in
            LaTeX format, unit in LaTeX format -> (name, symbol, unit).
        """

        implemented_labels = {
            "air_pressure": ("Water Vapour Pressure", r"$e$", "Pa"),
            "boundary_layer_height": ("Boundary Layer Height", r"$z_{BL}$", "m"),
            "cn2": ("Structure Parameter of Refractive Index", r"$C_{n}^{2}$", ""),
            "ct2": ("Structure Parameter of Temperature", r"$C_{T}^{2}$", ""),
            "dry_adiabatic_lapse_rate": (
                "Dry Adiabatic Lapse Rate",
                r"\Gamma_{d}",
                "Km$^{-1}$",
            ),
            "environmental_lapse_rate": (
                "Environmental Lapse Rate",
                r"\Gamma_{e}",
                "Km$^{-1}$",
            ),
            "grad_potential_temperature": (
                "Gradient of Potential Temperature",
                r"$\Delta \theta$",
                r"K$\cdot$m$^{-1}$",
            ),
            "h_free": ("Sensible Heat Flux", r"$Q_{H free}$", r"W$\cdot$m$^{-2}$"),
            "humidity": ("Humidity", r"$\rho_{v}$", r"kg$\cdot$m$^{-3}$"),
            "mixing_ratio": ("Mixing Ratio", r"$r$", r"kg$\cdot$kg$^{-1}"),
            "moist_adiabatic_lapse_rate": (
                "Moist Adiabatic Lapse Rate",
                r"\Gamma_{m}",
                "Km$^{-1}$",
            ),
            "msl_pressure": ("Mean Sea-Level Pressure", r"$P_{MSL}$", "Pa"),
            "obukhov": ("Obukhov Length", r"$L_{Ob}$", "m"),
            "potential_temperature": ("Potential Temperature", r"$\theta$", "K"),
            "pressure": ("Pressure", r"$P$", "mbar"),
            "rho_air": ("Air Density", r"$\rho_{air}$", r"kg$\cdot$m$^{3}$"),
            "saturated_temperature": (
                "Parcel Temperature (Saturated)",
                r"$T_{sat}$",
                "K",
            ),
            "shf": ("Sensible Heat Flux", r"$Q_{H}$", r"W$\cdot$m$^{-2}$"),
            "temperature": ("Temperature", r"$T$", "K"),
            "temperature_2m": ("2m Temperature", r"$T$", "K"),
            "theta_star": ("Temperature Scale", r"$\theta^{*}$", "K"),
            "u_star": ("Friction Velocity", r"$u^{*}$", r"m$\cdot$s$^{-2}$"),
            "unsaturated_temperature": (
                "Parcel Temperature (Unsaturated)",
                r"$T_{unsat}$",
                "K",
            ),
            "virtual_temperature": ("Virtual Temperature", r"$T_{v}$", "K"),
            "water_vapour_pressure": ("Water Vapour Pressure", r"$e$", "Pa"),
            "wind_speed": ("Wind Speed", r"$u$", r"ms$^{-2}$"),
        }

        name = dependent.lower()
        if name in implemented_labels:
            label = implemented_labels[name]
        else:
            label = (dependent.title(), f"${name}$", "")

        return label

    def get_date_and_timezone(self, data):
        """Return first time index and timezone.

        Args:
            data (pd.DataFrame or pd.Series): TZ-aware dataframe with
                DatetimeIndex.

        Returns:
            dict[str, datetime.tzinfo]: Date formatted as
            "DD Month YYYY", and timezone object.
        """

        date = data.index[0].strftime("%d %B %Y")
        timezone = data.index.tz

        return {"date": date, "tzone": timezone}

    def title_plot(self, title, timestamp, location=""):
        """Constructs title and legend.

        Args:
            title (str): Prefix to include in title.
            timestamp (Union[str, pd.Timestamp]): Date or time of data
                collection.
            location (str): Location of data collection. Default empty
                string.

        Returns:
            str: Title of plot with location and time.
        """

        if not location:
            location = ""  # Otherwise None interpreted literally in f-strings
        else:
            location = f" at {location}"

        if not isinstance(timestamp, str):
            timestamp = timestamp.strftime("%d %B %Y")

        title_string = f"{title}{location},\n{timestamp}"
        plt.title(title_string, fontweight="bold")
        plt.legend(loc="upper left")

        return title_string

    def merge_label_with_unit(self, label):
        """Merges variable name with its unit if applicable.

        Args:
            label (tuple[str, str, str]): Contains the name, symbol, and
                unit of a variable. Supports both TeX and empty strings.
                Strings containing TeX must be passed as raw strings::

                    label = ("Name", "Symbol", r"$Unit_{TeX}$")
                    label = ("Name", r"$Symbol_{TeX}$", "")

        Returns:
            str: Formatted string with the name and unit of a variable.
        """

        if not label[2]:  # if unit given
            merged_label = f"{label[0]}"
        else:
            merged_label = f"{label[0]}, [{label[2]}]"

        return merged_label

    def merge_multiple_labels(self, labels):
        """Merges multiple labels into a single formatted string.

        Args:
            labels (list[str]): Labels, which may contain duplicates.

        Returns:
            str: A formatted, punctuated string with no duplicates.
        """

        unique_text = list(dict.fromkeys(labels))
        if len(unique_text) < 2:
            merged = f"{unique_text[0]}"
        elif len(unique_text) == 2:
            merged = " and ".join(unique_text)
        else:
            merged = ", ".join(unique_text)

        return merged

    def set_xy_labels(self, ax, timezone, name):
        """Sets labels for X (time), Y (variable) axis.

        Args:
            ax (plt.Axes): Plot's axes.
            timezone (datetime.tzinfo): Local timezone of data.
            name (str): Name or abbreviation of dependent variable.

        Returns:
            plt.Axes: Plot axes with labels for local time on the x-axis
            and for the dependent variable with units on the y-axis.
            Ticks on the x-axis are formatted at hourly intervals.
        """

        x_label = f"Time, {timezone.zone}"
        label_text = self.label_selector(dependent=name)
        y_label = self.merge_label_with_unit(label=label_text)

        plt.xlabel(x_label)
        plt.ylabel(y_label)
        ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%H:%M", timezone))

        return ax

    def plot_constant_lines(self, axis, hlines=None, vlines=None):
        """Plots horizontal or vertical lines onto axis.

        Args:
            axis (plt.Axes): Target axes.
            hlines (dict[str, float]): Name and x-axis value for which
                to plot a horizontal line. Default None.
            vlines (dict[str, float]): Name and x-axis value for which
                to plot a vertical line. Default None.
        """

        if hlines:
            for key in hlines:
                if hlines[key]:
                    axis.axhline(
                        hlines[key],
                        color="grey",
                        linestyle="--",
                        label=self.label_selector(dependent=key)[0],
                    )
        if vlines:
            for key in vlines:
                if vlines[key]:
                    axis.axvline(
                        vlines[key],
                        color="red",
                        label=self.label_selector(dependent=key)[0],
                    )


class FigurePlotter(FigureFormat):
    """Handles drawing and saving figures from various data."""

    def __init__(self):
        super().__init__()

    def save_figure(
        self,
        figure,
        timestamp,
        suffix="",
        img_format="svg",
        output_dir="./reports/figures/",
    ):
        """Saves figure to disk.

        Args:
            figure (plt.Figure): Matplotlib figure object.
            timestamp (pd.Timestamp): Date or time of data collection.
            suffix (str): Additional string to add to file name.
            img_format (str): Image file format. Default SVG.
            output_dir (str): Location to save images.
                Default "./reports/figures/".
        """

        if not os.path.isdir(output_dir):
            try:
                os.mkdir(output_dir)
            except OSError as error:
                print(error)

        plot_id = timestamp.strftime("%Y%m%d")
        plot_id = re.sub(r"\W+", "", str(plot_id))

        if suffix:
            suffix = f"_{suffix}"

        figure.savefig(fname=f"{output_dir}{plot_id}{suffix}.{img_format}")
        plt.close()

    def setup_plot_data(self, data, names=None):
        """Sets up data for plotting.

        The original data is deep copied as slicing in the outer scope
        can overwrite data before it is plotted.

        Args:
            data (pd.DataFrame): Labelled data.
            names (list): Names of dependent variables. Default None.

        Returns:
            tuple[pd.DataFrame, pd.DataFrame, dict]: Deep copy of
            original data, hourly mean of data, and dictionary with date
            and timestamp metadata.
        """

        dataframe = data.copy(deep=True)  # don't overwrite outer scope
        dataframe = dataframe.astype("float64", errors="ignore")
        date_timezone = self.get_date_and_timezone(data=dataframe)
        hourly_mean = dataframe.dropna().resample("H", origin="start_day").mean()

        if names:
            for key in names:
                hourly_mean[key] = (
                    dataframe[key].resample("H", origin="start_day").mean()
                )
        hourly_mean.tz_convert(date_timezone["tzone"])

        return dataframe, hourly_mean, date_timezone

    def plot_time_series(
        self,
        series_data,
        series_mean=None,
        name="Time Series",
        line_colour="black",
        grey=False,
    ):
        """Plots time series and mean.

        Args:
            series_data (pd.Series): Series with data to plot.
            series_mean (pd.Series): Series with mean data to plot.
                Default None.
            name (str): Time series label. Default "Time Series".
            line_colour (str): Line colour. Default "black".
            grey (bool): If True, plotted line is greyed out.
                Default False.
        """

        if grey:
            series_data.plot(color="grey", label=name, x_compat=True)
        else:
            series_data.plot(color=line_colour, label=name, x_compat=True)

        if series_mean is not None:
            if name and name != "Time Series":
                mean_label = f"{name}, Hourly Mean"
            else:
                mean_label = "Hourly Mean"
            series_mean.plot(
                label=f"{mean_label}",
                color=line_colour,
                linestyle="dashed",
                x_compat=True,
            )

    def plot_generic(self, dataframe, name, site=""):
        """Plots time series of variable with hourly mean.

        Args:
            dataframe (pd.DataFrame): Contains data for times series.
            name (str): Name of dependent variable, must be key in
                dataframe.
            site (str): Location of data collection. Default empty
                string.

        Returns:
            tuple[plt.Figure, plt.Axes]: Figure and axes of time series
            plot.
        """

        plot_data, plot_mean, date_tzone = self.setup_plot_data(
            data=dataframe, names=[name]
        )

        figure = plt.figure(figsize=(26, 6))
        self.plot_time_series(
            series_data=plot_data[name],
            series_mean=plot_mean[name],
            line_colour="black",
            name="Time Series",
            grey=True,
        )

        title_name = self.label_selector(name)
        title_string = f"{title_name[0]}"
        site_label = self.get_site_name(site_name=site, dataframe=plot_data)
        self.title_plot(
            title=title_string, timestamp=date_tzone["date"], location=site_label
        )
        axes = plt.gca()
        self.set_xy_labels(ax=axes, timezone=date_tzone["tzone"], name=name)

        return figure, axes

    def plot_convection(self, dataframe, stability=None, site=""):
        """Plots scintillometer convection and free convection.

        Args:
            dataframe (pd.DataFrame): Contains sensible heat fluxes from
                on-board software and for free convection |H_free|.
            stability (str): Stability conditions. Default None.
            site (str): Location of data collection. Default empty
                string.

        Returns:
            tuple[plt.Figure, plt.Axes]: Figure, axes comparing the
            sensible heat flux from on-board software to free
            convection.
        """

        plot_data, plot_mean, date_tzone = self.setup_plot_data(
            data=dataframe, names=["H_convection", "H_free"]
        )

        figure = plt.figure(figsize=(26, 6))
        self.plot_time_series(
            series_data=plot_data["H_convection"],
            series_mean=plot_mean["H_convection"],
            line_colour="black",
            name="On-Board Software",
        )
        self.plot_time_series(
            series_data=plot_data["H_free"],
            series_mean=plot_mean["H_free"],
            line_colour="red",
            name="Free Convection",
        )

        if stability:
            stability_suffix = f"{stability.capitalize()} Conditions"
        else:
            stability_suffix = "No Height Dependency"
        title_string = (
            "Sensible Heat Fluxes from On-Board Software and",
            f"for Free Convection ({stability_suffix})",
        )
        site_label = self.get_site_name(site_name=site, dataframe=plot_data)
        self.title_plot(
            title=" ".join(title_string),
            timestamp=date_tzone["date"],
            location=site_label,
        )
        axes = plt.gca()
        self.set_xy_labels(ax=axes, timezone=date_tzone["tzone"], name="shf")

        return figure, axes

    def plot_comparison(self, df_01, df_02, keys, labels, site=""):
        """Plots two dataframes with identical indices.

        Args:
            df_01 (pd.DataFrame): First dataframe.
            df_02 (pd.DataFrame): Second dataframe.
            keys (list): Key names formatted as:
                [<df_01_key>, <df_series_key>].
            labels (list): Labels for legend formatted as:
                [<df_01_label>, <df_02_label>].
            site (str): Location of data collection. Default empty
                string.

        Returns:
            tuple[plt.Figure, plt.Axes]: Figure and axes of time series
            plot.
        """

        plot_data_01, plot_mean_01, plot_tzone = self.setup_plot_data(
            data=df_01, names=[keys[0]]
        )
        plot_data_02, plot_mean_02, _ = self.setup_plot_data(
            data=df_02, names=[keys[1]]
        )

        figure = plt.figure(figsize=(26, 6))
        self.plot_time_series(
            series_data=plot_data_01[keys[0]],
            series_mean=plot_mean_01[keys[0]],
            line_colour="black",
            name=labels[0],
        )
        self.plot_time_series(
            series_data=plot_data_02[keys[1]],
            series_mean=plot_mean_02[keys[1]],
            line_colour="red",
            name=labels[1],
        )

        key_labels = [self.label_selector(keys[0])[0], self.label_selector(keys[1])[0]]
        title_name = self.merge_multiple_labels(labels=key_labels)

        title_string = f"{title_name} from {labels[0]} and {labels[1]}"
        site_label = self.get_site_name(site_name=site, dataframe=plot_data_01)
        self.title_plot(
            title=title_string, timestamp=plot_tzone["date"], location=site_label
        )
        axes = plt.gca()
        self.set_xy_labels(ax=axes, timezone=plot_tzone["tzone"], name=keys[1])

        return figure, axes

    @Decorators.deprecated(
        stage="pending",
        reason="Superseded by MetricsFlux.plot_iterated_metrics.",
        version="1.0.5",
    )
    def plot_iterated_fluxes(self, iteration_data, time_id, location=""):
        """Plots and saves iterated SHF, comparison to free convection.

        .. note:: Pending deprecation in a future patch release. Use
            :func:`MetricsFlux.plot_iterated_metrics()
            <scintillometry.metrics.calculations.MetricsFlux.plot_iterated_metrics>`
            instead.

        .. todo::
            ST-126: Deprecate FigurePlotter.plot_iterated_fluxes in
                favour of plot_iterated_metrics.

        Args:
            iteration_data (pd.DataFrame): TZ-aware with sensible heat
                fluxes calculated for free convection |H_free|, and
                MOST |H|.
            time_id (pd.Timestamp): Local time of data collection.
            location (str): Location of data collection. Default empty
                string.

        Returns:
            list[tuple[plt.Figure, plt.Axes]]: Time series and
            comparison.
        """

        fig_shf, ax_shf = self.plot_generic(iteration_data, "shf", site=location)
        fig_shf = plt.gcf()
        self.save_figure(figure=fig_shf, timestamp=time_id, suffix="shf")

        fig_comp, ax_comp = self.plot_comparison(
            df_01=iteration_data,
            df_02=iteration_data,
            keys=["H_free", "shf"],
            labels=["Free Convection", "Iteration"],
            site=location,
        )
        fig_comp = plt.gcf()
        self.save_figure(figure=fig_comp, timestamp=time_id, suffix="shf_comp")

        return [(fig_shf, ax_shf), (fig_comp, ax_comp)]

    def plot_innflux(self, iter_data, innflux_data, name="obukhov", site=""):
        """Plots comparison between scintillometer and InnFLUX data.

        Args:
            iter_data (pd.DataFrame): Iterated data from scintillometer.
            innflux_data (pd.DataFrame): InnFLUX data.
            name (str): Name of dependent variable, must be key in
                dataframe.
            site (str): Location of data collection. Default empty
                string.

        Returns:
            tuple[plt.Figure, plt.Axes]: Figure and axes of time series
            plot.
        """

        iter_data, _, iter_tzone = self.setup_plot_data(data=iter_data, names=[name])
        iter_mean = iter_data.dropna().resample("30T", origin="start_day").mean()
        iter_mean[name] = (
            iter_data[name].dropna().resample("30T", origin="start_day").mean()
        )
        inn_data, _, _ = self.setup_plot_data(data=innflux_data, names=[name])

        figure = plt.figure(figsize=(26, 8))
        iter_data[name].plot(color="grey", label="Scintillometer")
        iter_mean[name].plot(
            color="black", linestyle="dashed", label="Scintillometer, half-hourly mean"
        )
        inn_data[name].plot(
            color="red", linestyle="dashed", label="InnFLUX, half-hourly mean"
        )

        title_name = self.label_selector(name)
        title_string = f"{title_name[0]} from Scintillometer and innFLUX"
        site_label = self.get_site_name(site_name=site, dataframe=iter_data)
        self.title_plot(
            title=title_string, timestamp=iter_tzone["date"], location=site_label
        )
        axes = plt.gca()
        self.set_xy_labels(ax=axes, timezone=iter_tzone["tzone"], name=name)

        return figure, axes

    def plot_scatter(
        self, x_data, y_data, name, sources, score=None, regression_line=None, site=""
    ):
        """Plots scatter between two datasets with a regression line.

        Args:
            x_data (pd.Series): Labelled explanatory data.
            y_data (pd.Series): Labelled response data.
            name (str): Name of variable.
            sources (list[str, str]): Names of data sources formatted
                as: [<Explanatory Source>, <Response Source>].
            score (float): Coefficient of determination |R^2|.
                Default None.
            regression_line (np.ndarray): Values for regression line.
                Default None.
            site (str): Location of data collection. Default empty
                string.

        Returns:
            tuple[plt.Figure, plt.Axes]: Regression plot of explanatory
            and response data, with fitted regression line and
            regression score.
        """

        figure = plt.figure(figsize=(8, 8))
        date = self.get_date_and_timezone(data=x_data)["date"]
        scatter_frame = pd.merge(
            x_data, y_data, left_index=True, right_index=True, sort=True
        )

        scatter_frame = scatter_frame.dropna(axis=0)  # drop mismatched index
        x_fit_data = scatter_frame.iloc[:, 0].values.reshape(-1, 1)
        y_fit_data = scatter_frame.iloc[:, 1].values.reshape(-1, 1)
        plt.scatter(x_fit_data, y_fit_data, marker=".", color="gray")

        if regression_line is not None:
            plt.plot(
                x_fit_data, regression_line, color="black", label="Line of Best Fit"
            )

        axes = plt.gca()
        if score is not None:
            plt.text(
                0.05,
                0.9,
                f"R$^{2}$= {score:.5f}",
                horizontalalignment="left",
                verticalalignment="bottom",
                transform=axes.transAxes,
            )
        variable_name = self.label_selector(name)
        sources_string = f"{sources[0]} and {sources[1]}"
        title_string = f"{variable_name[0]} Regression Between\n{sources_string}"
        site_label = self.get_site_name(site_name=site, dataframe=x_data)
        self.title_plot(title=title_string, timestamp=date, location=site_label)
        x_label = self.merge_label_with_unit(label=variable_name)
        y_label = self.merge_label_with_unit(label=variable_name)
        plt.xlabel(f"{x_label} ({sources[0]})")
        plt.ylabel(f"{y_label} ({sources[1]})")

        return figure, axes

    def plot_vertical_profile(
        self, vertical_data, time_idx, name, site="", y_lim=None, **kwargs
    ):
        """Plots vertical profile of variable.

        Args:
            vertical_data (dict[str, pd.DataFrame]): Contains time
                series of vertical measurements.
            time_idx (pd.Timestamp): The local time for which to plot a
                vertical profile.
            name (str): Name of dependent variable, must be key in
                <vertical_data>.
            site (str): Location of data collection. Default empty
                string.
            y_lim (float): Y-axis limit. Default None.

        Keyword Args:
            hlines (dict): Name and y-axis value for which to plot a
                horizontal line.
            vlines (dict): Name and x-axis value for which to plot a
                vertical line.

        Returns:
            tuple[plt.Figure, plt.Axes]: Figure and axis of vertical
            profiles.

        """

        if not y_lim:
            figure = plt.figure(figsize=(4, 8))
        else:
            figure = plt.figure()
        vertical_profile = vertical_data[name].loc[[time_idx]]
        time_data = self.get_date_and_timezone(data=vertical_data[name])
        variable_name = self.label_selector(dependent=name)

        plt.plot(
            vertical_profile.values[0],
            vertical_profile.columns,
            color="black",
            label=variable_name,
        )
        if not y_lim:
            plt.ylim(bottom=0)
        else:
            plt.ylim(0, y_lim)
            heights = vertical_profile.columns[vertical_profile.columns <= y_lim]

            xlim_max = math.ceil(max(vertical_profile[heights].values[0]))
            if xlim_max > 1:
                xlim_min = math.floor(min(vertical_profile[heights].values[0]))
                if not np.isclose(xlim_min, xlim_max):
                    plt.xlim(xlim_min, xlim_max)

        x_label = self.merge_label_with_unit(label=variable_name)
        y_label = "Height [m]"
        plt.xlabel(x_label)
        plt.ylabel(y_label)

        title = f"Vertical Profile of {variable_name[0]}"
        site_label = self.get_site_name(site_name=site, dataframe=vertical_data[name])
        if not site_label:
            location = ",\n"
        else:
            location = f"\nat {site_label}, "
        time_string = time_idx.strftime("%H:%M")
        time_label = f"{time_data['date']} {time_string} {time_data['tzone']}"
        title_string = f"{title}{location}{time_label}"
        plt.title(title_string, fontweight="bold")
        axes = plt.gca()
        if kwargs:
            self.parse_formatting_kwargs(axis=axes, **kwargs)

        return figure, axes

    def plot_vertical_comparison(self, dataset, time_index, keys, site="", **kwargs):
        """Plots comparison of two vertical profiles with the same indices.

        Args:
            dataset (dict): Dataframes of vertical profiles.
            time_index (str or pd.Timestamp): The time for which to plot a
                vertical profile.
            keys (list): Dependent variable keys.
            site (str): Location of data collection. Default empty string.

        Keyword Args:
            hlines (dict): Name and y-axis value for which to plot
                horizontal lines.
            vlines (dict): Name and x-axis value for which to plot
                vertical lines.

        Returns:
            tuple[plt.Figure, plt.Axes]: Figure and axis of vertical
            profiles.
        """

        key_length = len(keys)
        figure, axes = plt.subplots(
            nrows=1, ncols=key_length, sharey=False, figsize=(4 * key_length, 8)
        )
        subplot_labels = []
        for i in range(key_length):
            vertical_profile = dataset[keys[i]].loc[[time_index]]
            axes[i].plot(
                vertical_profile.values[0],
                vertical_profile.columns,
                color="black",
                label=keys[i],
            )
            label = self.label_selector(dependent=keys[i])
            subplot_labels.append(label[0])
            x_label = self.merge_label_with_unit(label=label)
            axes[i].set_xlabel(x_label)
            axes[i].set_ylim(bottom=0)

            if kwargs:
                self.parse_formatting_kwargs(axis=axes[i], **kwargs)
        axes[0].set_ylabel("Height [m]")

        title_name = self.merge_multiple_labels(labels=subplot_labels)
        title = f"Vertical Profiles of {title_name}"
        site_label = self.get_site_name(site_name=site, dataframe=dataset[keys[0]])
        if not site_label:
            location = ",\n"
        else:
            location = f"\nat {site_label}, "
        time_data = self.get_date_and_timezone(data=dataset[keys[-1]].loc[[time_index]])
        time_string = time_index.strftime("%H:%M")
        time_label = f"{time_data['date']} {time_string} {time_data['tzone']}"
        title_string = f"{title}{location}{time_label}"
        figure.suptitle(title_string, fontweight="bold")

        return figure, axes

    def plot_merged_profiles(self, dataset, time_index, site="", y_lim=None, **kwargs):
        """Plots vertical profiles on the same axis.

        Args:
            dataset (dict): Dataframes of vertical profiles.
            time_index (str or pd.Timestamp): The time for which to plot
                a vertical profile.
            site (str): Location of data collection. Default empty
                string.
            y_lim (float): Y-axis limit. Default None.

        Keyword Args:
            hlines (dict): Names and y-axis values for which to plot
                horizontal lines.
            title (str): Figure title.
            vlines (dict): Names and x-axis values for which to plot
                vertical lines.
            x_label (str): X-axis label. Default None.

        Returns:
            tuple[plt.Figure, plt.Axes]: Figure and axis of vertical
            profiles.
        """

        keys = list(dataset)
        key_length = len(keys)
        figure = plt.figure(figsize=(8, 8))
        subplot_labels = []
        xlims = []
        for i in range(key_length):
            vertical_profile = dataset[keys[i]].loc[[time_index]]
            line_label = self.label_selector(dependent=keys[i])
            plt.plot(
                vertical_profile.values[0],
                vertical_profile.columns,
                label=line_label[0],
            )
            subplot_labels.append(line_label[0])

            if y_lim:
                heights = vertical_profile.columns[vertical_profile.columns <= y_lim]
                xlim_max = math.ceil(max(vertical_profile[heights].values[0]))
                xlims.append(xlim_max)
                if xlim_max > 1:
                    xlim_min = math.floor(min(vertical_profile[heights].values[0]))
                    xlims.append(xlim_min)
        line_label = self.label_selector(dependent=keys[-1])
        x_label = self.merge_label_with_unit(label=line_label)
        plt.xlabel(x_label)
        plt.ylabel("Height [m]")
        if not y_lim:
            plt.ylim(bottom=0)
        else:
            plt.ylim(0, y_lim)
            if xlims:
                if not np.isclose(min(xlims), max(xlims)):
                    plt.xlim(min(xlims), max(xlims))
        plt.legend()

        site_label = self.get_site_name(site_name=site, dataframe=dataset[keys[0]])
        if not site_label:
            location = ",\n"
        else:
            location = f"\nat {site_label}, "
        time_data = self.get_date_and_timezone(data=dataset[keys[-1]].loc[[time_index]])
        time_string = time_index.strftime("%H:%M")
        time_label = f"{time_data['date']} {time_string} {time_data['tzone']}"
        title_name = self.merge_multiple_labels(labels=subplot_labels)
        title = f"Vertical Profiles of {title_name}"
        title_string = f"{title}{location}{time_label}"
        plt.title(title_string, fontweight="bold")

        axes = plt.gca()
        if kwargs:
            self.parse_formatting_kwargs(axis=axes, **kwargs)
            if "title" in kwargs:
                title_string = f"{kwargs['title']}{location}{time_label}"
                plt.title(title_string, fontweight="bold")

        return figure, axes
