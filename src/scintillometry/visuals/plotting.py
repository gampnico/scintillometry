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

Automates data plotting.
"""

import os
import re

import matplotlib
import matplotlib.pyplot as plt


def initialise_formatting(small=14, medium=16, large=18, extra=20):
    """Initialises font sizes for matplotlib figures.

    Called separately from other plotting functions in this module to
    avoid overwriting on-the-fly formatting changes.

    Args:
        small (int): Size of text, ticks, legend.
        medium (int): Size of axis labels.
        large (int): Size of axis title.
        extra (int): Size of figure title (suptitle).
    """

    plt.rc("font", size=small)  # controls default text sizes
    plt.rc("axes", titlesize=large)  # fontsize of the axes title
    plt.rc("axes", labelsize=medium)  # fontsize of the x and y labels
    plt.rc("xtick", labelsize=small)  # fontsize of the tick labels
    plt.rc("ytick", labelsize=small)  # fontsize of the tick labels
    plt.rc("legend", fontsize=small)  # legend fontsize
    plt.rc("figure", titlesize=extra)  # fontsize of the figure title


def get_site_name(site_name, dataframe=None):
    """Gets name of site from user string or from dataframe metadata.

    Args:
        site_name (str): Location of data collection.
        dataframe (pd.DataFrame): Any collected dataset.

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


def label_selector(dependent):
    """Constructs label parameters from dependent variable name.

    If the variable name is not implemented, the unconverted variable is
    passed directly, with the symbol duplicated from its name and with
    no unit provided.

    Args:
        dependent (str): Name of dependent variable.

    Returns:
        tuple[str, str, str]: Name of dependent variable, symbol in
        LaTeX format, unit in LaTeX format -> (name, symbol, unit).
    """

    implemented_labels = {
        "air_pressure": ("Water Vapour Pressure", r"$e$", "Pa"),
        "cn2": ("Structure Parameter of Refractive Index", r"$C_{n}^{2}$", ""),
        "ct2": ("Structure Parameter of Temperature", r"$C_{T}^{2}$", ""),
        "grad_potential_temperature": (
            "Gradient of Potential Temperature",
            r"$\Delta \theta$",
            r"K$\cdot$m$^{-1}$",
        ),
        "h_free": ("Sensible Heat Flux", r"$Q_{H free}$", r"W$\cdot$m$^{-2}$"),
        "humidity": ("Humidity", r"$\rho_{v}$", r"kg$\cdot$m$^{-3}$"),
        "mixing_ratio": ("Mixing Ratio", r"$r$", r"kg$\cdot$kg$^{-1}"),
        "msl_pressure": ("Mean Sea-Level Pressure", r"$P_{MSL}$", "Pa"),
        "obukhov": ("Obukhov Length", r"$L_{Ob}$", "m"),
        "potential_temperature": ("Potential Temperature", r"$\theta$", "K"),
        "pressure": ("Pressure", r"$P$", "mbar"),
        "rho_air": ("Air Density", r"$\rho_{air}$", r"kg$\cdot$m$^{3}$"),
        "shf": ("Sensible Heat Flux", r"$Q_{H}$", r"W$\cdot$m$^{-2}$"),
        "temperature": ("Temperature", r"$T$", "K"),
        "temperature_2m": ("2m Temperature", r"$T$", "K"),
        "theta_star": ("Temperature Scale", r"$\theta^{*}$", "K"),
        "u_star": ("Friction Velocity", r"$u^{*}$", r"m$\cdot$s$^{-2}$"),
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


def get_date_and_timezone(data):
    """Return first time index and timezone.

    Args:
        data (pd.DataFrame): TZ-aware dataframe with DateTimeIndex.

    Returns:
        dict[str, pd.DatetimeIndex.tz]: Date formatted as
        "DD Month YYYY", and timezone object.
    """

    date = data.index[0].strftime("%d %B %Y")
    timezone = data.index.tz

    return {"date": date, "tzone": timezone}


def title_plot(title, timestamp, location=""):
    """Constructs title and legend.

    Args:
        title (str): Prefix to include in title.
        timestamp (str): Date or time of data collection.
        location (str): Location of data collection. Default is empty
            string.

    Returns:
        str: Title of plot with location and time.
    """

    if not location:
        location = ""  # Otherwise None is interpreted literally in f-strings
    else:
        location = f" at {location}"

    title_string = "".join((f"{title}{location},\n{timestamp}"))
    plt.title(title_string, fontweight="bold")
    plt.legend(loc="upper left")

    return title_string


def merge_label_with_unit(label):
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


def merge_multiple_labels(labels):
    """Merges multiple labels into a single formatted string.

    Args:
        labels (list[str]): Labels, which may contain duplicates.

    Returns:
        str: A formatted, puncutated string with no duplicates.
    """

    unique_text = list(dict.fromkeys(labels))
    if len(unique_text) < 2:
        merged = f"{unique_text[0]}"
    elif len(unique_text) == 2:
        merged = " and ".join(unique_text)
    else:
        merged = ", ".join(unique_text)

    return merged


def set_xy_labels(ax, timezone, name):
    """Sets labels for X (time), Y (variable) axis.

    Args:
        ax (plt.Axes): Plot's axes.
        timezone (pd.DatetimeIndex.tz): Local timezone of data.
        name (str): Name or abbreviation of dependent variable.

    Returns:
        plt.Axes: Plot axes with labels for local time on the x axis and
        for the dependent variable with units on the y axis. Ticks on
        the x axis are formatted at hourly intervals.
    """

    x_label = f"Time, {timezone.zone}"
    label_text = label_selector(dependent=name)
    y_label = merge_label_with_unit(label=label_text)

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%H:%M", timezone))

    return ax


def save_figure(
    figure, timestamp, suffix="", img_format="svg", output_dir="./reports/figures/"
):
    """Saves figure to disk.

    Args:
        figure (plt.Figure): Matplotlib figure object.
        timestamp (pd.Timestamp): Time or date of data collection.
        suffix (str): Additional string to add to file name.
        img_format (str): Image file format. Default SVG.
        output_dir (str): Location to save images.
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


def setup_plot_data(df, names=None):
    """Sets up data for plotting.

    Args:
        df (pd.DataFrame): Dataframe to plot.
        names (list): Names of dependent variables.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame, dict]: Deep copy of original
        data, hourly mean of data, and dictionary with date and
        timestamp metadata.
    """

    dataframe = df.copy(deep=True)  # don't overwrite outer scope
    dataframe = dataframe.astype("float64", errors="ignore")
    date_timezone = get_date_and_timezone(data=dataframe)
    hourly_mean = dataframe.dropna().resample("H", origin="start_day").mean()

    if names:
        for key in names:
            hourly_mean[key] = dataframe[key].resample("H", origin="start_day").mean()
    hourly_mean.tz_convert(date_timezone["tzone"])

    return dataframe, hourly_mean, date_timezone


def plot_time_series(
    series_data, series_mean=None, name="Time Series", line_colour="black", grey=False
):
    """Plots time series and mean.

    Args:
        series_data (pd.Series): Series with data to plot.
        series_mean (pd.Series): Series with mean data to plot.
            Default None.
        name (str): Time series label. Default empty string.
        line_colour (str): Line colour. Default "black".
        grey (bool): If True, time series is greyed out. Default False.
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


def plot_generic(dataframe, name, site=""):
    """Plots time series of variable with hourly mean.

    Args:
        dataframe (pd.DataFrame): Contains data for times series.
        name (str): Name of dependent variable, must be key in
            dataframe.
        site (str): Location of data collection. Default empty string.

    Returns:
        tuple[plt.Figure, plt.Axes]: Figure and axes of time series
        plot.
    """

    plot_data, plot_mean, date_tzone = setup_plot_data(df=dataframe, names=[name])

    fig = plt.figure(figsize=(26, 6))
    plot_time_series(
        series_data=plot_data[name],
        series_mean=plot_mean[name],
        line_colour="black",
        name="Time Series",
        grey=True,
    )

    title_name = label_selector(name)
    title_string = f"{title_name[0]}"
    site_label = get_site_name(site_name=site, dataframe=plot_data)
    title_plot(title=title_string, timestamp=date_tzone["date"], location=site_label)
    axes = plt.gca()
    set_xy_labels(ax=axes, timezone=date_tzone["tzone"], name=name)

    return fig, axes


def plot_convection(dataframe, stability, site=""):
    """Plots scintillometer convection and free convection.

    Args:
        dataframe (pd.DataFrame): Contains sensible heat fluxes from
            on-board software and for free convection |H_free|.
        stability (str): Stability conditions.
        site (str): Location of data collection. Default empty
            string.

    Returns:
        tuple[plt.Figure, plt.Axes]: Figure and axes of plotted data.
    """

    plot_data, plot_mean, date_tzone = setup_plot_data(
        df=dataframe, names=["H_convection", "H_free"]
    )

    figure = plt.figure(figsize=(26, 6))
    plot_time_series(
        series_data=plot_data["H_convection"],
        series_mean=plot_mean["H_convection"],
        line_colour="black",
        name="On-Board Software",
    )
    plot_time_series(
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
    site_label = get_site_name(site_name=site, dataframe=plot_data)
    title_plot(
        title=" ".join(title_string), timestamp=date_tzone["date"], location=site_label
    )
    axes = plt.gca()
    set_xy_labels(ax=axes, timezone=date_tzone["tzone"], name="shf")

    return figure, axes


def plot_comparison(df_01, df_02, keys, labels, site=""):
    """Plots comparison between two dataframes with the same indices.

    Args:
        df_01 (pd.DataFrame): First dataframe.
        df_02 (pd.DataFrame): Second dataframe.
        keys (list): Key names formatted as:
            [<df_01_key>, <df_series_key>].
        labels (list): Labels for legend formatted as:
            [<df_01_label>, <df_02_label>].
        site (str): Location of data collection. Default empty string.

    Returns:
        tuple[plt.Figure, plt.Axes]: Figure and axes of time series
        plot.
    """

    plot_data_01, plot_mean_01, plot_tzone = setup_plot_data(df=df_01, names=[keys[0]])
    plot_data_02, plot_mean_02, _ = setup_plot_data(df=df_02, names=[keys[1]])

    fig = plt.figure(figsize=(26, 6))
    plot_time_series(
        series_data=plot_data_01[keys[0]],
        series_mean=plot_mean_01[keys[0]],
        line_colour="black",
        name=labels[0],
    )
    plot_time_series(
        series_data=plot_data_02[keys[1]],
        series_mean=plot_mean_02[keys[1]],
        line_colour="red",
        name=labels[1],
    )

    key_labels = [label_selector(keys[0])[0], label_selector(keys[1])[0]]
    title_name = merge_multiple_labels(labels=key_labels)

    title_string = f"{title_name} from {labels[0]} and {labels[1]}"
    site_label = get_site_name(site_name=site, dataframe=plot_data_01)
    title_plot(title=title_string, timestamp=plot_tzone["date"], location=site_label)
    axes = plt.gca()
    set_xy_labels(ax=axes, timezone=plot_tzone["tzone"], name=keys[1])

    return fig, axes


def plot_iterated_fluxes(iteration_data, time_id, location=""):
    """Plots and saves iterated SHF and a comparison to free convection.

    Args:
        iteration_data (pd.DataFrame): TZ-aware with sensible heat
            fluxes calculated for free convection |H_free|, and MOST |H|.
        time_id (pd.Timestamp): Local time of data collection.
        location (str): Location of data collection. Default empty
            string.

    Returns:
        tuple[plt.Figure, plt.Figure]: Time series and comparison.
    """

    fig_shf, _ = plot_generic(iteration_data, "shf", site=location)
    fig_shf = plt.gcf()
    save_figure(figure=fig_shf, timestamp=time_id, suffix="shf")

    fig_comp, _ = plot_comparison(
        df_01=iteration_data,
        df_02=iteration_data,
        keys=["H_free", "shf"],
        labels=["Free Convection", "Iteration"],
        site=location,
    )
    fig_comp = plt.gcf()
    save_figure(figure=fig_comp, timestamp=time_id, suffix="shf_comp")

    return fig_shf, fig_comp


def plot_innflux(iter_data, innflux_data, name="obukhov", site=""):
    """Plots comparison with InnFLUX data.

    Args:
        iter_data (pd.DataFrame): Iterated data from scintillometer.
        innflux_data (pd.DataFrame): InnFLUX data.
        name (str): Name of dependent variable, must be key in
            dataframe.
        site (str): Location of data collection. Default empty string.

    Returns:
        tuple[plt.Figure, plt.Axes]: Figure and axes of time series
        plot.
    """

    iter_data, _, iter_tzone = setup_plot_data(df=iter_data, names=[name])
    iter_mean = iter_data.dropna().resample("30T", origin="start_day").mean()
    iter_mean[name] = (
        iter_data[name].dropna().resample("30T", origin="start_day").mean()
    )
    inn_data, _, _ = setup_plot_data(df=innflux_data, names=[name])

    fig = plt.figure(figsize=(26, 8))
    iter_data[name].plot(color="grey", label="Scintillometer")
    iter_mean[name].plot(
        color="black", linestyle="dashed", label="Scintillometer, half-hourly mean"
    )
    inn_data[name].plot(
        color="red", linestyle="dashed", label="InnFLUX, half-hourly mean"
    )

    title_name = label_selector(name)
    title_string = f"{title_name[0]} from Scintillometer and InnFLUX"
    site_label = get_site_name(site_name=site, dataframe=iter_data)
    title_plot(title=title_string, timestamp=iter_tzone["date"], location=site_label)
    axes = plt.gca()
    set_xy_labels(ax=axes, timezone=iter_tzone["tzone"], name=name)

    return fig, axes


def plot_vertical_profile(vertical_data, time_idx, name, site=""):
    """Plots vertical profile of variable.

    Args:
        vertical_data (dict[pd.DataFrame]): Contains time series of
            vertical profiles.
        time_idx (str or pd.Timestamp): The local time for which to plot
            a vertical profile.
        name (str): Name of dependent variable, must be key in
            <vertical_data>.
        site (str): Location of data collection. Default empty string.
    """

    fig = plt.figure(figsize=(4, 8))
    vertical_profile = vertical_data[name].iloc[
        vertical_data[name].index.indexer_at_time(time_idx)
    ]
    time_data = get_date_and_timezone(data=vertical_data[name])
    title_name = label_selector(name)

    plt.plot(
        vertical_profile.values[0],
        vertical_profile.columns,
        color="black",
        label=title_name,
    )

    plt.ylim(bottom=0)
    label = label_selector(dependent=name)
    x_label = merge_label_with_unit(label=label)
    y_label = "Height [m]"
    plt.xlabel(x_label)
    plt.ylabel(y_label)

    title = f"Vertical Profile of {title_name[0]}"
    site_label = get_site_name(site_name=site, dataframe=vertical_data[name])
    if not site_label:
        location = ",\n"
    else:
        location = f"\nat {site_label}, "
    if not isinstance(time_idx, str):
        time_idx = time_idx.strftime("%H:%M")
    time_label = f"{time_data['date']} {time_idx} {time_data['tzone']}"
    title_string = f"{title}{location}{time_label}"
    plt.title(title_string, fontweight="bold")
    axes = plt.gca()

    return fig, axes
