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

Parses raw data, creates datasets.
"""

import io
import os
import re

import numpy as np
import pandas as pd
import scipy

from scintillometry.backend.constants import AtmosConstants


def check_file_exists(fname):
    """Check file exists.

    Args:
        fname (str): Path to a file.

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
    raw_headers = []
    for line in line_list[5: (variables_start_line - 1)]:
        raw_headers.append(" ".join(line.strip().split()))
    # fmt: on

    header_parameters = {}
    for param in raw_headers:
        key, _, value = param.partition(": ")
        header_parameters[key] = value

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
            lengths, [m]. Format as [incorrect, correct].

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


def change_index_frequency(data, frequency="60S"):
    """Change frequency of time index.

    Args:
        data (pd.DataFrame or pd.Series): An object with a time or
            datetime index.
        frequency (str): Reindexing frequency. Default "60s".

    Returns:
        pd.DataFrame or pd.Series: Object with new index frequency.
    """

    old_idx = data.index
    new_idx = pd.date_range(old_idx.min(), old_idx.max(), freq=frequency)
    data = data.reindex(old_idx.union(new_idx)).interpolate("index").reindex(new_idx)

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
            lengths, [m]. Format as [incorrect, correct].

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
    if "PT" in str(dataframe["time"][0]):
        dataframe["iso_duration"] = dataframe["time"].apply(parse_iso_date, date=False)
        dataframe["iso_duration"] = pd.to_timedelta(dataframe["iso_duration"])
        dataframe["time"] = dataframe["time"].apply(parse_iso_date, date=True)

    dataframe = convert_time_index(data=dataframe, tzone=timezone)

    if calibration:
        dataframe = calibrate_data(data=dataframe, path_lengths=calibration)

    if "Station Code" in mnd_data["parameters"]:
        caption = mnd_data["parameters"]["Station Code"]
        dataframe.attrs["name"] = caption  # attrs is experimental

    return dataframe


def parse_transect(file_path):
    """Parses scintillometer path transect.

    Args:
        file_path (str): Path to processed transect. The data must be
            formatted as <path_height>, <normalised_path_position>. The
            normalised path position maps to:
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
    timestamp, klima_id, data_dir="./ext/data/raw/ZAMG/", timezone=None
):
    """Parses ZAMG climate records.

    Args:
        timestamp (pd.Timestamp): Start time of climate record.
        klima_id (str): ZAMG weather station ID (Klima-ID).
        data_dir (str): Location of ZAMG data files.
        timezone (str): Local timezone during the scintillometer's
            operation. Default None.

    Returns:
        pd.DataFrame: Parsed ZAMG records.
    """

    date = timestamp.strftime("%Y%m%d")
    file_name = f"{data_dir}{klima_id}_ZEHNMIN Datensatz_{date}T0000_{date}T2350.csv"
    check_file_exists(file_name)

    zamg_data = pd.read_csv(file_name, sep=",", dtype={"station": str})
    zamg_data = convert_time_index(data=zamg_data, tzone=timezone)
    station_id = zamg_data["station"][0]

    # resample to 60s intervals
    zamg_data = change_index_frequency(data=zamg_data, frequency="60S")
    zamg_data["station"] = station_id  # str objects were converted to NaN

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
    external weather data. It only preserves |Cn2| and SHF data from the
    scintillometer.

    If temperature or pressure data is in Celsius or Pa, they are
    automatically converted to Kelvin and hPa, respectively - any
    subsequent maths assumes these units.

    Args:
        scint_dataframe (pd.DataFrame): Parsed and localised
            scintillometry data.
        weather_dataframe (pd.DataFrame): Parsed and localised weather
            data.

    Returns:
        pd.DataFrame: Merged dataframe containing both scintillometry
        data, and interpolated weather conditions.
    """

    merged = scint_dataframe.filter(["Cn2", "H_convection"], axis=1)
    merged = merged.join(weather_dataframe)

    # adjust units
    if (weather_dataframe["temperature_2m"].lt(100)).any():  # if True data in Celsius
        merged["temperature_2m"] = merged["temperature_2m"] + AtmosConstants().kelvin
    if (weather_dataframe["pressure"].gt(2000)).any():  # if True data in Pa
        merged["pressure"] = merged["pressure"] / 100  # Pa -> hPa

    return merged


def wrangle_data(
    bls_path,
    transect_path,
    calibrate,
    weather_dir="./ext/data/raw/ZAMG/",
    station_id="11803",
    tzone="CET",
):
    """Wrangle BLS, ZAMG, and transect datasets.

    Args:
        bls_path (str): Path to a raw .mnd data file using FORMAT-1.
        transect_path (str): Path to processed transect. The data must
            be formatted as <path_height>, <normalised_path_position>.
            The normalised path position maps to:
            [0: receiver location, 1: transmitter location].
        calibrate (list): Contains the incorrect and correct path
            lengths. Format as [incorrect, correct].
        weather_dir (str): Path to directory with local weather data.
            Default "./ext/data/raw/ZAMG/".
        station_id (str): ZAMG weather station ID (Klima-ID).
            Default 11803.
        tzone (str): Local timezone during the scintillometer's
            operation. Default "CET".


    Returns:
        dict: BLS, ZAMG, and transect dataframes, an interpolated
        dataframe at 60s resolution containing BLS and ZAMG data, and a
        pd.TimeStamp object of the scintillometer's recorded start time
        of data collection. All returned objects are localised to the
        timezone selected by the user.
    """

    bls_data = parse_scintillometer(
        file_path=bls_path,
        timezone=tzone,
        calibration=calibrate,
    )
    bls_time = bls_data.index[0]

    transect_data = parse_transect(file_path=transect_path)

    zamg_data = parse_zamg_data(
        timestamp=bls_time,
        klima_id=station_id,
        data_dir=weather_dir,
        timezone=tzone,
    )
    interpolated_data = merge_scintillometry_weather(
        scint_dataframe=bls_data, weather_dataframe=zamg_data
    )

    data_dict = {
        "bls": bls_data,
        "weather": zamg_data,
        "transect": transect_data,
        "interpolated": interpolated_data,
        "timestamp": bls_time,
    }

    return data_dict


def parse_innflux_mat(file_path):
    """Parse MATLAB® data structures generated by innFLUX.

    Supports MATLAB® array version 7 with MATLAB® serial dates.
    Systematic errors in time conversion are in O(20 ms) so timestamps
    are rounded to the nearest second.

    Args:
        file_path (str): Path to .mat file.

    Returns:
        pd.DataFrame: Contains innFLUX measurements.

    Raises:
        ValueError: File does not have a .mat extension.
        KeyError: InnFLUX data does not contain any values for <key>.
    """

    if file_path[-4:] != ".mat":
        raise ValueError("File does not have a .mat extension.")

    mat_structure = scipy.io.loadmat(
        file_path, verify_compressed_data_integrity=True, simplify_cells=True
    )
    for key in ["time", "MET"]:
        if key not in mat_structure.keys():
            error_msg = f"InnFLUX data does not contain any values for {key}."
            raise KeyError(error_msg)
    time_data = mat_structure["time"]

    # Conversion error is O(20 ms)
    innflux_index = pd.to_datetime(time_data - 719529, unit="D").round(freq="s")
    innflux_names = {
        "hws": "wind_speed",
        "wdir": "wind_direction",
        "ust": "friction_velocity",
        "T": "temperature",
        "wT": "shf",
        "L": "obukhov",
        "zoL": "stability_parameter",
        "p": "pressure",
        "theta": "potential_temperature",
        "theta_v": "virtual_potential_temperature",
    }

    met_data = {}
    rename_columns = {}
    for key, name in innflux_names.items():
        if key in mat_structure["MET"].keys():
            met_data[key] = mat_structure["MET"][key]
            rename_columns[key] = name

    eddy_data = pd.DataFrame.from_dict(data=met_data)
    eddy_data.index = innflux_index
    eddy_data = eddy_data.rename(columns=rename_columns)

    return eddy_data


def parse_innflux_csv(file_path, header_list=None):
    """Parse pre-processed innFLUX data from .csv files.

    If innFLUX data was provided as a pre-processed .csv file (i.e. you
    are not licensed to use raw data), it may only contain data for a
    limited number of variables with no headers present.

    Optionally pass a list of column headers using the <header_list>
    argument. If no list is passed, a default list of headers is used.

    Args:
        file_path (str): Path to .csv file.
        header_list (list): List of column headers for data. Default
            None.

    Returns:
        pd.DataFrame: Contains innFLUX measurements.
    """

    if not header_list:
        header_list = [
            "year",
            "month",
            "day",
            "hour",
            "minute",
            "second",
            "shf",
            "wind_speed",
            "obukhov",
        ]
    dataframe = pd.read_csv(
        file_path, header=None, index_col=None, names=header_list, sep=","
    )

    t_cols = ["year", "month", "day", "hour", "minute", "second"]
    dataframe.index = pd.to_datetime(dataframe[t_cols])
    dataframe = dataframe.drop(t_cols, axis=1)

    return dataframe


def parse_innflux(file_name, tzone=None, headers=None):
    """Parses InnFLUX eddy covariance data.

    The input data should be pre-processed from raw eddy covariance
    measurements using the innFLUX Eddy Covariance
    code. [#striednig2020]_

    The input file should contain data for the sensible heat flux, wind
    speed, and Obukhov length. It may optionally contain data for the
    stability parameter.

    If parsing a .csv file, optionally pass a list of column headers
    using the <headers> argument. If no list is passed, a default list
    of headers is used. This argument is ignored for non-csv files.

    Args:
        file_name (str): Path to innFLUX data.
        tzone (str): Local timezone during the scintillometer's
            operation. Default None.
        headers (list): List of column headers for data. Default None.

    Returns:
        pd.DataFrame: Parsed and localised InnFLUX data.
    """

    check_file_exists(file_name)

    if file_name[-4:] == ".mat":
        dataframe = parse_innflux_mat(file_path=file_name)
    else:
        dataframe = parse_innflux_csv(file_path=file_name, header_list=headers)

    dataframe.index = dataframe.index + pd.DateOffset(hours=3)
    dataframe = dataframe.replace(-999, np.nan)
    dataframe = dataframe.fillna(method="ffill")

    dataframe = change_index_frequency(data=dataframe, frequency="60S")

    if tzone:
        dataframe = dataframe.tz_localize(tzone)
    else:
        dataframe = dataframe.tz_localize("UTC")

    return dataframe


def construct_hatpro_levels(levels=None):
    """Construct HATPRO scanning levels.

    Hardcoded scan levels specifically for HATPRO Retrieval data from
    HATPRO UIBK Met (612m). Scan levels are integer measurement heights
    relative to the station's elevation.

    Args:
        levels (list[int]): HATPRO measurement heights, |z_scan| [m].

    Returns:
        list: HATPRO measurement heights, |z_scan| [m].

    Raises:
        TypeError: Input levels must be a list or tuple of integers.
    """

    if not levels:  # structured for readability
        levels = [
            (0, 10, 30),
            (50, 75, 100, 125),
            (150, 200, 250),
            (325, 400, 475, 550, 625),
            (700, 800, 900),
            (1000, 1150, 1300, 1450),
            (1600, 1800, 2000),
            (2200, 2500, 2800),
            (3100, 3500, 3900),
            (4400, 5000, 5600, 6200),
            (7000, 8000, 9000, 10000),
        ]
        scan = [i for heights in levels for i in heights]
    elif all(isinstance(height, int) for height in levels):
        scan = [heights for heights in levels]
    else:
        raise TypeError("Input levels must be a list or tuple of integers.")

    return scan


def load_hatpro(file_name, levels, tzone, station_elevation=612.0):
    """Load raw HATPRO data into dataframe.

    Args:
        file_name (str): Path to raw HATPRO data.
        levels (list[int]): Height of HATPRO scan level, |z_scan| [m].
        tzone (str): Local timezone during the scintillometer's
            operation. Default None.
        station_elevation (float): Station elevation, |z_stn| [m].
            Default 612.0 m.
    Returns:
        pd.DataFrame: Contains tz-aware and pre-processed HATPRO data.

    Raises:
        FileNotFoundError: No file found with path: <fname>.
    """

    check_file_exists(file_name)
    data = pd.read_csv(
        file_name,
        sep=";",
        comment="#",
        header=0,
        names=levels,
        index_col=0,
        parse_dates=True,
    )

    data.index = data.index + pd.DateOffset(hours=2)
    if tzone:
        data = data.tz_localize(tzone)
    else:
        data = data.tz_localize("UTC")

    data.attrs["elevation"] = station_elevation

    return data


def parse_hatpro(file_prefix, scan_heights=None, timezone=None, elevation=612.0):
    """Parses HATPRO Retrieval data.

    Args:
        file_prefix (str): Path prefix for HATPRO Retrieval data. There
            should be two HATPRO files ending with "humidity" and
            "temp". The path prefix should be identical for both files,
            e.g.::

                ./path/to/file_humidity.csv
                ./path/to/file_temp.csv

            would require `file_prefix = "./path/to/file_"`.
        scan_heights (list[int]): Heights of HATPRO measurement levels,
            |z_scan| [m].
        timezone (str): Local timezone during HATPRO operation.
            Default None.
        elevation (float): Station elevation, |z_stn| [m].
            Default 612.0 m.

    Returns:
        dict[pd.DataFrame, pd.DataFrame]: Vertical measurements from
        HATPRO for temperature |T| [K], and absolute humidity
        |rho_v| [|gm^-3|].
    """

    humidity_path = f"{file_prefix}humidity.csv"
    temperature_path = f"{file_prefix}temp.csv"

    scan_levels = construct_hatpro_levels(levels=scan_heights)
    humidity_data = (10 ** (-3)) * load_hatpro(
        file_name=humidity_path,
        levels=scan_levels,
        tzone=timezone,
        station_elevation=elevation,
    )
    temperature_data = load_hatpro(
        file_name=temperature_path,
        levels=scan_levels,
        tzone=timezone,
        station_elevation=elevation,
    )

    data = {}
    data["humidity"] = humidity_data
    data["temperature"] = temperature_data

    return data


def parse_vertical(
    file_path, device="hatpro", levels=None, tzone=None, station_elevation=612.0
):
    """Parses vertical measurements.

    Currently only supports HATPRO.

    Args:
        file_path (str): Path to vertical measurements. For HATPRO
            Retrieval data there should be two HATPRO files ending with
            "humidity" and "temp". The path should be identical for both
            files, e.g.::

                ./path/to/file_humidity.csv
                ./path/to/file_temp.csv

            would require `file_path = "./path/to/file_"`.
        device (str): Instrument used for vertical measurements. Only
            supports HATPRO. Default "hatpro".
        levels (list[int]): Heights of HATPRO measurements,
            |z_scan| [m].
        tzone (str): Local timezone during the radiometer's operation.
            Default None.
        station_elevation (float): Station elevation, |z_stn| [m].
            Default 612.0 m.

    Returns:
        dict[pd.DataFrame, pd.DataFrame]: Vertical measurements for
        temperature |T| [K], and absolute humidity |rho_v| [|gm^-3|].

    Raises:
        NotImplementedError: <device> measurements are not supported.
            Use "hatpro".
    """

    if not device.lower() == "hatpro":
        error_msg = f"{device.title()} measurements are not supported. Use 'hatpro'."
        raise NotImplementedError(error_msg)
    else:
        vert_data = parse_hatpro(
            file_prefix=file_path,
            scan_heights=levels,
            timezone=tzone,
            elevation=station_elevation,
        )

    return vert_data
