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

Automates data plotting.
"""

import os
import re

import matplotlib
import matplotlib.pyplot as plt


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
        "shf": ("Sensible Heat Flux", r"$Q_{H}$", r"[W$\cdot$m$^{-2}$]"),
        "wind_speed": ("Wind Speed", r"$u$", r"[ms$^{-2}$]"),
        "obukhov": ("Obukhov Length", r"$L_{Ob}$", "[m]"),
        "theta_star": ("Temperature Scale", r"$\theta^{*}$", "[K]"),
        "u_star": ("Friction Velocity", r"$u^{*}$", r"[m$\cdot$s$^{-2}$]"),
        "temperature_2m": ("Temperature", r"$T$", "[K]"),
        "pressure": ("Pressure", r"$P$", "[mbar]"),
        "rho_air": ("Air Density", r"$\rho_{air}$", r"[kg$\cdotm^{3}$]"),
        "cn2": ("Structure Parameter of Refractive Index", "", r"$C_{n}^{2}$"),
        "ct2": ("Structure Parameter of Temperature", "", r"$C_{T}^{2}$"),
        "h_free": (
            "Sensible Heat Flux (Free Convection)",
            r"$Q_{H free}$",
            r"[W$\cdot$m$^{-2}$]",
        ),
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

    title_string = "".join((f"{title}{location}, {timestamp}"))
    plt.title(title_string, fontsize=24, fontweight="bold")
    plt.legend(loc="upper left")

    return title_string


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
    label = label_selector(dependent=name)
    y_label = f"{label[0]}"
    if label[2]:  # if unit given
        y_label = f"{y_label}, {label[2]}"

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
    title_plot(title=title_string, timestamp=date_tzone["date"], location=site)
    axes = plt.gca()
    set_xy_labels(ax=axes, timezone=date_tzone["tzone"], name=name)

    return fig, axes
