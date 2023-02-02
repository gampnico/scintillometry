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

Parses raw data, creates datasets.
"""

import io
import os
import re

import numpy as np
import pandas as pd


def check_file_exists(fname):
    """Check file exists.

    Args:
        fname: Path to a file.

    Raises:
        FileNotFoundError: No file found with path: <fname>.
    """

    if not os.path.exists(fname):
        raise FileNotFoundError(f"No file found with path: {fname}")


def file_handler(filename):
    """Opens file as read-only and appends each line up to EOF to a list.

    Args:
        filename (str): Path to file.

    Returns:
        list: List of lines read from file.

    Raises:
        FileNotFoundError: No file found with path: <filename>.
    """

    check_file_exists(fname=filename)

    with open(filename, mode="r", encoding="utf-8") as file:
        file_list = file.readlines()

    return file_list


def parse_mnd_lines(line_list):
    """Parses data and variable names from a list of .mnd lines.

    Args:
        line_list (list): Lines read from .mnd file in FORMAT-1.

    Returns:
        dict[list, list, str, list]: Contains a list of lines of parsed
        BLS data, an ordered list of variable names, the file timestamp,
        and any additional header parameters in the file header.

    Raises:
        Warning: The input file does not follow FORMAT-1.
    """

    if line_list[0][:8] != "FORMAT-1":
        raise Warning("The input file does not follow FORMAT-1.")

    timestamp = line_list[1]
    header_number = int(line_list[3].partition(" ")[0])
    variable_number = int(line_list[3].partition(" ")[-1])
    variables_start_line = 6 + header_number
    data_start_line = variables_start_line + variable_number + 1

    # fmt: off
    header_parameters = [
        line.strip() for line in line_list[5: (variables_start_line - 1)]
    ]
    # fmt: on

    reg_match = r"\#(.+?)\#"  # find variable abbreviations in second column
    variable_names = []
    for line in line_list[variables_start_line:data_start_line]:
        if line != "\n":  # avoid greedy capture
            match_list = re.findall(reg_match, line)
            variable_names.append(match_list[0].strip())

    parsed_mnd_lines = {
        "data": line_list[data_start_line:],
        "names": variable_names,
        "timestamp": timestamp,
        "parameters": header_parameters,
    }

    return parsed_mnd_lines


def parse_iso_date(x, date=True):
    """Parses timestamp with mixed ISO-8601 duration and date.

    Uses integer properties of bool to act as index for partition.

    Args:
        x (str): Timestamp containing ISO-8601 duration and date,
            i.e. "<ISO-8601 duration>/<ISO-8601 date>".
        date (bool): If True, returns date. Otherwise returns duration.

    Returns:
        str: ISO-8601 string representing either a duration or a date.
    """

    return x.partition("/")[-date]


def calibrate_data(data, path_lengths):
    """Calibrates data if the wrong path length was set.

    Recalibrate data if the wrong path length was set in SRun or on the
    dip switches in the scintillometer. Use the argument::

        --c, --calibrate <wrong_path_length> <correct_path_length>

    Args:
        data (pd.DataFrame): Parsed and localised scintillometry
            dataframe.
        path_lengths (list): Contains the incorrect and correct path
            lengths. Format as [incorrect, correct].

    Returns:
        pd.DataFrame: Recalibrated dataframe.

    Raises:
        ValueError: Calibration path lengths must be formatted as:
            <wrong_path_length> <correct_path_length>.
    """

    if len(path_lengths) == 2:
        calibration_constant = (float(path_lengths[1]) ** (-3)) / (
            float(path_lengths[0]) ** (-3)  # correct / incorrect
        )
        for key in ["Cn2", "H_convection"]:
            data[key] = data[key] * calibration_constant  # (path / incorrect) * correct
    else:
        error_msg = (
            "Calibration path lengths must be formatted as: ",
            "<wrong_path_length> <correct_path_length>.",
        )
        raise ValueError(error_msg)

    return data


def convert_time_index(data, tzone=None):
    """Make tz-naive dataframe tz-aware.

    Args:
        data (pd.DataFrame): Tz-naive dataframe.
        tzone (str): Local timezone. Default "UTC".

    Returns:
        pd.DataFrame: Tz-aware dataframe in local timezone or UTC.
    """

    data["time"] = pd.to_datetime(data["time"])
    data = data.set_index("time")
    if tzone:
        data = data.tz_convert(tzone)
    else:
        data = data.tz_convert("UTC")

    return data


def parse_scintillometer(file_path, timezone=None, calibration=None):
    """Parses .mnd files into dataframes.

    Args:
        filename (str): Path to a raw .mnd data file using FORMAT-1.
        timezone (str): Local timezone during the scintillometer's
            operation. Default None.
        calibration (list): Contains the incorrect and correct path
            lengths. Format as [incorrect, correct].

    Returns:
        pd.DataFrame: Parsed and localised scintillometry data.
    """

    mnd_lines = file_handler(filename=file_path)

    mnd_data = parse_mnd_lines(line_list=mnd_lines)

    dataframe = pd.read_table(
        io.StringIO("".join(mnd_data["data"])),
        delim_whitespace=True,
        names=mnd_data["names"],
        header=None,
    )

    # Parse mixed-format timestamp
    if "PT" in dataframe["time"][0]:
        dataframe["iso_duration"] = dataframe["time"].apply(parse_iso_date, date=False)
        dataframe["iso_duration"] = pd.to_timedelta(dataframe["iso_duration"])
        dataframe["time"] = dataframe["time"].apply(parse_iso_date, date=True)

    dataframe = convert_time_index(data=dataframe, tzone=timezone)

    if calibration:
        dataframe = calibrate_data(data=dataframe, path_lengths=calibration)

    return dataframe


def parse_transect(file_path):
    """Parses scintillometer path transect.

    Args:
        file_path (str): Path to processed transect, formatted as
            <path_height>, <normalised_path_position>. The normalised
            path position maps to:
            [0: receiver location, 1: transmitter location].

    Returns:
        pd.DataFrame: Parsed path transect data.

    Raises:
        FileNotFoundError: No file found with path: <file_path>.
        ValueError: Normalised position is not between 0 and 1.
    """

    check_file_exists(fname=file_path)

    path_height_dataframe = pd.read_csv(
        file_path, header=None, names=["path_height", "norm_position"]
    )

    if not all(path_height_dataframe["norm_position"].between(0, 1, "both")):
        raise ValueError("Normalised position is not between 0 and 1.")

    return path_height_dataframe


def parse_zamg_data(
    timestamp, station_id, data_dir="./ext/data/raw/ZAMG/", timezone=None
):
    """Parses ZAMG climate records.

    Args:
        timestamp (pd.Timestamp): Start time of climate record.
        station_id (int): ZAMG weather station ID (Klima-ID).
        data_dir (str): Location of ZAMG data files.
        timezone (str): Local timezone during the scintillometer's
            operation. Default None.

    Returns:
        pd.DataFrame: Parsed ZAMG records.
    """

    date = timestamp.strftime("%Y%m%d")
    file_name = (
        f"{data_dir}{str(station_id)}_ZEHNMIN Datensatz_{date}T0000_{date}T2350.csv"
    )
    check_file_exists(file_name)

    zamg_data = pd.read_csv(file_name, sep=",")
    zamg_data = convert_time_index(data=zamg_data, tzone=timezone)

    # resample to 60s intervals
    oidx = zamg_data.index
    nidx = pd.date_range(oidx.min(), oidx.max(), freq="60s")
    zamg_data = zamg_data.reindex(oidx.union(nidx)).interpolate("index").reindex(nidx)

    zamg_names = {
        "DD": "wind_direction",
        "DDX": "gust_direction",
        "FF": "vector_wind_speed",
        "FFAM": "wind_speed",
        "FFX": "gust_speed",
        "GSX": "global_irradiance",
        "HSR": "diffuse_sky_radiation_mv",
        "HSX": "diffuse_sky_radiation_wm",
        "P": "pressure",
        "P0": "mean_sea_level_pressure",
        "RF": "relative_humidity",
        "RR": "precipitation",
        "SH": "snow_depth",
        "SO": "sunshine_duration",
        "TB1": "soil_temperature_10cm",
        "TB2": "soil_temperature_20cm",
        "TB3": "soil_temperature_50cm",
        "TL": "temperature_2m",
        "TLMAX": "temperature_2m_max",
        "TLMIN": "temperature_2m_min",
        "TP": "dew_point",
        "TS": "temperature_5cm",
        "TSMAX": "temperature_5cm_max",
        "TSMIN": "temperature_5cm_min",
        "ZEITX": "gust_time",
    }

    rename_columns = {}
    for key, name in zamg_names.items():
        if key in zamg_data.columns:
            rename_columns[key] = name

    zamg_data = zamg_data.rename(columns=rename_columns)

    return zamg_data


def merge_scintillometry_weather(scint_dataframe, weather_dataframe):
    """Merges parsed scintillometry and weather dataframes.

    This replaces any weather data collected by the scintillometer with
    external weather data. Only the collected |Cn2| data is preserved
    from the scintillometer.

    Args:
        scint_dataframe (pd.DataFrame): Parsed and localised
            scintillometry data.
        weather_dataframe (pd.DataFrame): Parsed and localised weather
            data.

    Returns:
        pd.DataFrame: Merged dataframe containing both scintillometry
            data, and interpolated weather conditions.

    .. |Cn2| replace:: Cn :sup:`2`
    """

    kelvin = 273.15
    merged = scint_dataframe.filter(["Cn2", "H_convection"], axis=1)
    merged = merged.join(weather_dataframe)

    # adjust units
    merged["temperature_2m"] = merged["temperature_2m"] + kelvin  # C -> K

    return merged
