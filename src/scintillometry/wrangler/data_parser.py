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

import pandas as pd


def file_handler(filename):
    """Opens file as read-only and appends each line up to EOF to a list.

    Args:
        filename (str): Path to file.

    Returns:
        list: List of lines read from file.

    Raises:
        FileNotFoundError: No file found with path: <filename>.
    """

    if not os.path.exists(filename):
        raise FileNotFoundError(f"No file found with path: {filename}")

    with open(filename, "r") as file:
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
    header_number = line_list[3].partition(" ")[0]
    variables_start_line = 6 + int(header_number)
    # fmt: off
    header_parameters = [
        line.strip() for line in line_list[5: (variables_start_line - 1)]
    ]
    # fmt: on

    reg_match = r"\#(.+?)\#"  # find variable abbreviations in second column
    variable_names = []
    for line in line_list[variables_start_line:]:
        if line != "\n":  # ensure stop before reaching data
            match_list = re.findall(reg_match, line)
            variable_names.append(match_list[0].strip())
        else:
            break
    data_start_line = variables_start_line + len(variable_names) + 1
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


def parse_scintillometer(file_path, timezone=None):
    """Parses .mnd files into dataframes.

    Args:
        filename (str): Path to a raw .mnd data file using FORMAT-1.
        timezone (str): Local timezone during the scintillometer's
            operation. Default None.

    Returns:
        pd.DataFrame: Correctly indexed scintillometry data.
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
        dataframe["time"] = pd.to_datetime(
            dataframe["time"].apply(parse_iso_date, date=True)
        )

    dataframe = dataframe.set_index("time")
    if timezone:
        dataframe = dataframe.tz_convert(timezone)

    return dataframe
