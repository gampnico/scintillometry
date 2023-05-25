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

Parses raw data, creates datasets.

Writing functions for parsing yet another variation of yet another data
format is often excruciatingly finicky. To add support for new data
sources:

    - Create a new parsing function in its most relevant class. Possible
      classes include `WranglerScintillometer`, `WranglerWeather`,
      `WranglerTransect`,`WranglerEddy`,`WranglerVertical`.
    - Add an `elif` statement to the class' `parse_<ClassName>` function
      (the only function in the class that includes a `source`
      argument).
    - Call the new parsing function e.g.::

        parse_vertical(file_path, source="new_format")
"""

import io
import os
import re

import numpy as np
import pandas as pd
import scipy

from scintillometry.backend.constants import AtmosConstants


class WranglerIO:
    """Performs file operations on data."""

    def __init__(self):
        super().__init__()

    def check_file_exists(self, fname):
        """Check file exists.

        Args:
            fname (str): Path to a file.

        Raises:
            FileNotFoundError: No file found with path: <fname>.
        """

        if not os.path.exists(fname):
            raise FileNotFoundError(f"No file found with path: {fname}")

    def file_handler(self, filename):
        """Opens file as read-only and appends each line to a list.

        Args:
            filename (str): Path to file.

        Returns:
            list: List of lines read from file up to EOF.
        """

        self.check_file_exists(fname=filename)

        with open(filename, mode="r", encoding="utf-8") as file:
            file_list = file.readlines()

        return file_list


class WranglerTransform:
    """Transforms data labelling and indexing."""

    def __init__(self):
        super().__init__()

    def parse_iso_date(self, x, date=True):
        """Parses timestamp with mixed ISO-8601 duration and date.

        Uses integer properties of bool to act as index for partition.

        Args:
            x (str): Timestamp containing ISO-8601 duration and date,
                i.e. "<ISO-8601 duration>/<ISO-8601 date>".
            date (bool): If True, returns date. Otherwise, returns
                duration. Default True.

        Returns:
            str: ISO-8601 string representing either a duration or a
            date.
        """

        return x.partition("/")[-date]

    def change_index_frequency(self, data, frequency="60S"):
        """Change frequency of time index.

        Args:
            data (pd.DataFrame or pd.Series): An object with a time or
                datetime index.
            frequency (str): Reindexing frequency. Default "60S".

        Returns:
            pd.DataFrame or pd.Series: Object with new index frequency.
        """

        old_idx = data.index
        new_idx = pd.date_range(old_idx.min(), old_idx.max(), freq=frequency)
        data = (
            data.reindex(old_idx.union(new_idx)).interpolate("index").reindex(new_idx)
        )

        return data

    def convert_time_index(self, data, tzone=None):
        """Make tz-naive dataframe tz-aware.

        Args:
            data (pd.DataFrame): Tz-naive dataframe.
            tzone (str): Local timezone. Default None.

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


class WranglerScintillometer(WranglerIO):
    """Parses scintillometry data.

    Attributes:
        transform (WranglerTransform): Inherited methods for
            transforming dataframe parameters.
    """

    def __init__(self):
        super().__init__()
        self.transform = WranglerTransform()

    def parse_mnd_lines(self, line_list):
        """Parses data and variable names from a list of .mnd lines.

        Args:
            line_list (list): Lines read from .mnd file in FORMAT-1.

        Returns:
            dict: Contains a list of lines of parsed BLS data, an
            ordered list of variable names, the file timestamp, and any
            additional header parameters in the file header.

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

    def calibrate_data(self, data, path_lengths):
        """Calibrates data if the wrong path length was set.

        Recalibrate data if the wrong path length was set in SRun or on
        the dip switches in the scintillometer. Use the argument::

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
                data[key] = (
                    data[key] * calibration_constant
                )  # (path / incorrect) * correct
        else:
            error_msg = (
                "Calibration path lengths must be formatted as: ",
                "<wrong_path_length> <correct_path_length>.",
            )
            raise ValueError(error_msg)

        return data

    def parse_scintillometer(self, file_path, timezone="CET", calibration=None):
        """Parses .mnd files into dataframes.

        Args:
            file_path (str): Path to a raw .mnd data file using
                FORMAT-1.
            timezone (str): Local timezone during the scintillometer's
                operation. Default "CET".
            calibration (list): Contains the incorrect and correct path
                lengths, [m]. Format as [incorrect, correct].
                Default None.

        Returns:
            pd.DataFrame: Parsed and localised scintillometry data.
        """

        mnd_lines = self.file_handler(filename=file_path)

        mnd_data = self.parse_mnd_lines(line_list=mnd_lines)

        dataframe = pd.read_table(
            io.StringIO("".join(mnd_data["data"])),
            delim_whitespace=True,
            names=mnd_data["names"],
            header=None,
        )

        # Parse mixed-format timestamp
        if "PT" in str(dataframe["time"][0]):
            dataframe["iso_duration"] = dataframe["time"].apply(
                self.transform.parse_iso_date, date=False
            )
            dataframe["iso_duration"] = pd.to_timedelta(dataframe["iso_duration"])
            dataframe["time"] = dataframe["time"].apply(
                self.transform.parse_iso_date, date=True
            )

        dataframe = self.transform.convert_time_index(data=dataframe, tzone=timezone)

        if calibration:
            dataframe = self.calibrate_data(data=dataframe, path_lengths=calibration)

        if "Station Code" in mnd_data["parameters"]:
            caption = mnd_data["parameters"]["Station Code"]
            dataframe.attrs["name"] = caption  # attrs is experimental

        return dataframe


class WranglerTransect(WranglerIO):
    """Parses topographical data.

    Attributes:
        transform (WranglerTransform): Inherited methods for
            transforming dataframe parameters.
    """

    def __init__(self):
        super().__init__()
        self.transform = WranglerTransform()

    def parse_dgm_processed(self, file_path):
        """Parses path transect from pre-processed DGM data.

        The pre-processed data is a .csv file formatted as:
        <path_height>, <normalised_path_position>. The normalised path
        position maps to:
        [0: receiver location, 1: transmitter location].

        Args:
            file_path (str): Path to processed transect. The data must
                be formatted as:
                <path_height>, <normalised_path_position>. The
                normalised path position maps to:
                [0: receiver location, 1: transmitter location].

        Returns:
            pd.DataFrame: Parsed path transect data.

        Raises:
            FileNotFoundError: No file found with path: <file_path>.
            ValueError: Normalised position is not between 0 and 1.
        """

        path_height_dataframe = pd.read_csv(
            file_path, header=None, names=["path_height", "norm_position"]
        )

        if not all(path_height_dataframe["norm_position"].between(0, 1, "both")):
            raise ValueError("Normalised position is not between 0 and 1.")

        return path_height_dataframe

    def parse_transect(self, file_path, source="dgm_processed"):
        """Parses scintillometer path transect.

        Args:
            file_path (str): Path to topographical data.
            source (str): Data source of topographical data. Currently
                supports pre-processed DGM 5m. Default "dgm_processed".

        Returns:
            pd.DataFrame: Parsed path transect data.

        Raises:
            NotImplementedError: <source> measurements are not
                supported. Use "dgm_processed".
        """

        if source.lower() == "dgm_processed":
            transect = self.parse_dgm_processed(file_path=file_path)
        else:
            error_msg = (
                f"{source.title()} measurements are not supported. Use 'dgm_processed'."
            )
            raise NotImplementedError(error_msg)

        return transect


class WranglerWeather(WranglerIO):
    """Parses meteorological data.

    Attributes:
        transform (WranglerTransform): Inherited methods for
            transforming dataframe parameters.
    """

    def __init__(self):
        super().__init__()
        self.transform = WranglerTransform()

    def parse_zamg_data(
        self,
        timestamp,
        klima_id="11803",
        data_dir="./ext/data/raw/ZAMG/",
        timezone="CET",
    ):
        """Parses ZAMG climate records.

        Args:
            timestamp (pd.Timestamp): Start time of climate record.
            klima_id (str): ZAMG weather station ID (Klima-ID).
                Default "11803".
            data_dir (str): Location of ZAMG data files.
                Default "./ext/data/raw/ZAMG/".
            timezone (str): Local timezone during the scintillometer's
                operation. Default "CET".

        Returns:
            pd.DataFrame: Parsed ZAMG records.
        """

        date = timestamp.strftime("%Y%m%d")
        file_name = (
            f"{data_dir}{klima_id}_ZEHNMIN Datensatz_{date}T0000_{date}T2350.csv"
        )
        self.check_file_exists(file_name)

        zamg_data = pd.read_csv(file_name, sep=",", dtype={"station": str})
        zamg_data = self.transform.convert_time_index(data=zamg_data, tzone=timezone)
        station_id = zamg_data["station"][0]

        # resample to 60s intervals
        zamg_data = self.transform.change_index_frequency(
            data=zamg_data, frequency="60S"
        )
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

    def parse_weather(
        self,
        timestamp,
        source="zamg",
        data_dir="./ext/data/raw/ZAMG/",
        station_id="11803",
        timezone="CET",
    ):
        """Parses weather data. Only supports ZAMG files.

        Args:
            timestamp (pd.Timestamp): Start time of climate record.
            source (str): Data source of weather data. Currently
                supports ZAMG. Default "zamg".
            data_dir (str): Location of weather data files.
                Default "./ext/data/raw/ZAMG/".
            station_id (str): Weather station ID (e.g. ZAMG Klima-ID).
                Default 11803.
            timezone (str): Local timezone of the scintillometer's
                measurement period. Default "CET".

        Returns:
            pd.DataFrame: Parsed weather data.

        Raises:
            NotImplementedError: <source> measurements are not
                supported. Use "zamg".
        """

        if source.lower() == "zamg":
            weather = self.parse_zamg_data(
                timestamp=timestamp,
                klima_id=station_id,
                data_dir=data_dir,
                timezone=timezone,
            )
        else:
            error_msg = f"{source.title()} measurements are not supported. Use 'zamg'."
            raise NotImplementedError(error_msg)

        return weather


class WranglerEddy(WranglerIO):
    """Parses eddy covariance data.

    Attributes:
        transform (WranglerTransform): Inherited methods for
            transforming dataframe parameters.
    """

    def __init__(self):
        super().__init__()
        self.transform = WranglerTransform()

    def parse_innflux_mat(self, file_path):
        """Parse MATLAB® data structures generated by innFLUX.

        Supports MATLAB® array version 7 with MATLAB® serial dates.
        Systematic errors in time conversion are in O(20 ms) so
        timestamps are rounded to the nearest second.

        Args:
            file_path (str): Path to .mat file.

        Returns:
            pd.DataFrame: Contains innFLUX measurements.

        Raises:
            ValueError: File does not have a .mat extension.
            KeyError: InnFLUX data does not contain any values for
                <key>.
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

    def parse_innflux_csv(self, file_path, header_list=None):
        """Parse pre-processed innFLUX data from .csv files.

        If innFLUX data was provided as a pre-processed .csv file (i.e.
        you are not licensed to use raw data), it may only contain data
        for a limited number of variables with no headers present.

        Optionally pass a list of column headers using the <header_list>
        argument. If no list is passed, a default list of headers is
        used.

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

    def parse_innflux(self, file_name, timezone=None, headers=None):
        """Parses InnFLUX eddy covariance data.

        The input data should be pre-processed from raw eddy covariance
        measurements using the innFLUX Eddy Covariance
        code. [#striednig2020]_

        The input file should contain data for the sensible heat flux,
        wind speed, and Obukhov length. It may optionally contain data
        for the stability parameter.

        If parsing a .csv file, optionally pass a list of column headers
        using the <headers> argument. If no list is passed, a default
        list of headers is used. This argument is ignored for non-csv
        files.

        Args:
            file_name (str): Path to innFLUX data.
            timezone (str): Local timezone during the scintillometer's
                operation. Default None.
            headers (list): List of column headers for data.
                Default None.

        Returns:
            pd.DataFrame: Parsed and localised innFLUX data.
        """

        self.check_file_exists(file_name)

        if file_name[-4:] == ".mat":
            dataframe = self.parse_innflux_mat(file_path=file_name)
        else:
            dataframe = self.parse_innflux_csv(file_path=file_name, header_list=headers)

        dataframe.index = dataframe.index + pd.DateOffset(hours=3)
        dataframe = dataframe.replace(-999, np.nan)
        dataframe = dataframe.fillna(method="ffill")

        dataframe = self.transform.change_index_frequency(
            data=dataframe, frequency="60S"
        )

        if timezone:
            dataframe = dataframe.tz_localize(timezone)
        else:
            dataframe = dataframe.tz_localize("UTC")

        return dataframe

    def parse_eddy_covariance(self, file_path, source="innflux", tzone=None):
        """Parses eddy covariance measurements.

        Currently only supports innFLUX.

        Args:
            file_path (str): Path to eddy covariance measurements.
            source (str): Data source of eddy covariance measurements.
                Only supports innFLUX. Default "innflux".
            tzone (str): Local timezone of the measurement period.
                Default None.

        Returns:
            dict[pd.DataFrame, pd.DataFrame]: Parsed eddy covariance
                measurements.

        Raises:
            NotImplementedError: <source> measurements are not
                supported. Use "innflux".
        """

        if source.lower() == "innflux":
            eddy_data = self.parse_innflux(file_name=file_path, timezone=tzone)
        else:
            error_msg = (
                f"{source.title()} measurements are not supported. Use 'innflux'."
            )
            raise NotImplementedError(error_msg)

        return eddy_data


class WranglerVertical(WranglerIO):
    """Parses vertical measurements.

    Attributes:
        transform (WranglerTransform): Inherited methods for
            transforming dataframe parameters.
    """

    def __init__(self):
        super().__init__()
        self.transform = WranglerTransform()

    def construct_hatpro_levels(self, levels=None):
        """Construct HATPRO scanning levels.

        Hardcoded scan levels specifically for HATPRO Retrieval data
        from HATPRO UIBK Met (612m). Scan levels are integer measurement
        heights relative to the station's elevation.

        Args:
            levels (list[int]): HATPRO measurement heights,
                |z_scan| [m]. Default None.

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

    def load_hatpro(self, file_name, levels, tzone="CET", station_elevation=612.0):
        """Load raw HATPRO data into dataframe.

        Args:
            file_name (str): Path to raw HATPRO data.
            levels (list[int]): Height of HATPRO scan level,
                |z_scan| [m].
            tzone (str): Local timezone of the scintillometer's
                measurement period. Default "CET".
            station_elevation (float): Station elevation, |z_stn| [m].
                Default 612.0.
        Returns:
            pd.DataFrame: Contains tz-aware and pre-processed HATPRO
            data.
        """

        self.check_file_exists(file_name)
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

    def parse_hatpro(
        self, file_prefix, timezone="CET", scan_heights=None, elevation=612.0
    ):
        """Parses HATPRO Retrieval data.

        Args:
            file_prefix (str): Path prefix for HATPRO Retrieval data.
                There should be two HATPRO files ending with "humidity"
                and "temp". The path prefix should be identical for both
                files, e.g.::

                    ./path/to/file_humidity.csv
                    ./path/to/file_temp.csv

                would require `file_prefix = "./path/to/file_"`.
            timezone (str): Local timezone of the scintillometer's
                measurement period. Default "CET".
            scan_heights (list[int]): Heights of HATPRO measurement
                levels, |z_scan| [m]. Default None.
            elevation (float): Station elevation, |z_stn| [m].
                Default 612.0.

        Returns:
            dict[str, pd.DataFrame]: Vertical measurements from HATPRO
            for temperature |T| [K], and absolute humidity
            |rho_v| [|gm^-3|].
        """

        humidity_path = f"{file_prefix}humidity.csv"
        temperature_path = f"{file_prefix}temp.csv"

        scan_levels = self.construct_hatpro_levels(levels=scan_heights)
        humidity_data = (10 ** (-3)) * self.load_hatpro(
            file_name=humidity_path,
            levels=scan_levels,
            tzone=timezone,
            station_elevation=elevation,
        )
        temperature_data = self.load_hatpro(
            file_name=temperature_path,
            levels=scan_levels,
            tzone=timezone,
            station_elevation=elevation,
        )

        data = {"humidity": humidity_data, "temperature": temperature_data}

        return data

    def parse_vertical(
        self,
        file_path,
        source="hatpro",
        tzone="CET",
        levels=None,
        station_elevation=612.0,
    ):
        """Parses vertical measurements.

        Currently only supports HATPRO.

        Args:
            file_path (str): Path to vertical measurements. For HATPRO
                Retrieval data there should be two HATPRO files ending
                with "humidity" and "temp". The path should be identical
                for both files, e.g.::

                    ./path/to/file_humidity.csv
                    ./path/to/file_temp.csv

                would require `file_path = "./path/to/file_"`.
            source (str): Instrument used for vertical measurements.
                Only supports HATPRO. Default "hatpro".
            levels (list[int]): Heights of HATPRO measurements,
                |z_scan| [m]. Default None.
            tzone (str): Local timezone of the scintillometer's
                measurement period. Default "CET".
            station_elevation (float): Station elevation, |z_stn| [m].
                Default 612.0.

        Returns:
            dict[pd.DataFrame, pd.DataFrame]: Vertical measurements for
            temperature |T| [K], and absolute humidity
            |rho_v| [|gm^-3|].

        Raises:
            NotImplementedError: <source> measurements are not
                supported. Use "hatpro".
        """

        if source.lower() == "hatpro":
            vert_data = self.parse_hatpro(
                file_prefix=file_path,
                scan_heights=levels,
                timezone=tzone,
                elevation=station_elevation,
            )
        else:
            error_msg = (
                f"{source.title()} measurements are not supported. Use 'hatpro'."
            )
            raise NotImplementedError(error_msg)

        return vert_data


class WranglerStitch:
    """Merges parsed data into unified datasets."""

    def __init__(self):
        super().__init__()
        self.constants = AtmosConstants()

    def merge_scintillometry_weather(self, scintillometry, weather):
        """Merges parsed scintillometry and weather dataframes.

        This replaces any weather data collected by the scintillometer
        with external weather data. It only preserves |Cn2| and SHF data
        from the scintillometer.

        If temperature or pressure data is in Celsius or Pa, they are
        automatically converted to Kelvin and hPa, respectively - any
        subsequent maths assumes these units.

        Args:
            scintillometry (pd.DataFrame): Parsed and localised
                scintillometry data.
            weather (pd.DataFrame): Parsed and localised weather data.

        Returns:
            pd.DataFrame: Merged dataframe containing both
            scintillometry data, and interpolated weather data.
        """

        merged = scintillometry.filter(["Cn2", "H_convection"], axis=1)
        merged = merged.join(weather)

        # adjust units
        if (weather["temperature_2m"].lt(100)).any():  # if True data in Celsius
            merged["temperature_2m"] = merged["temperature_2m"] + self.constants.kelvin
        if (weather["pressure"].gt(2000)).any():  # if True data in Pa
            merged["pressure"] = merged["pressure"] / 100  # Pa -> hPa

        return merged


class WranglerParsing(WranglerIO):
    """Wrapper class for parsing data.

    Attributes:
        scintillometer (WranglerScintillometer): Inherited methods for
            parsing scintillometry data.
        weather (WranglerWeather): Inherited methods for parsing weather
            data.
        transect (WranglerTransect): Inherited methods for parsing
            topographical data.
        eddy (WranglerEddy): Inherited methods for parsing eddy
            covariance data.
        vertical (WranglerVertical): Inherited methods for parsing
            vertical measurements.
        stitch (WranglerStitch): Inherited methods for merging parsed
            data into unified datasets.
    """

    def __init__(self):
        super().__init__()

        self.scintillometer = WranglerScintillometer()
        self.weather = WranglerWeather()
        self.transect = WranglerTransect()
        self.eddy = WranglerEddy()
        self.vertical = WranglerVertical()
        self.stitch = WranglerStitch()

    def wrangle_data(
        self,
        bls_path,
        transect_path,
        calibrate,
        weather_dir="./ext/data/raw/ZAMG/",
        station_id="11803",
        tzone="CET",
        weather_source="zamg",
    ):
        """Wrangle BLS, ZAMG, and transect datasets.

        Args:
            bls_path (str): Path to a raw .mnd data file using FORMAT-1.
            transect_path (str): Path to processed transect. The data
                must be formatted as:
                <path_height>, <normalised_path_position>. The
                normalised path position maps to:
                [0: receiver location, 1: transmitter location].
            calibrate (list): Contains the incorrect and correct path
                lengths. Format as [incorrect, correct].
            weather_dir (str): Path to directory with local weather
                data. Default "./ext/data/raw/ZAMG/".
            station_id (str): ZAMG weather station ID (Klima-ID).
                Default 11803.
            tzone (str): Local timezone during the scintillometer's
                operation. Default "CET".
            weather_source (str): Data source of weather data. Currently
                supports ZAMG. Default "zamg".

        Returns:
            dict: BLS, ZAMG, and transect dataframes, an interpolated
            dataframe at 60s resolution containing BLS and ZAMG data,
            and a pd.TimeStamp object of the scintillometer's recorded
            start time of data collection. All returned objects are
            localised to the timezone selected by the user.
        """

        bls_data = self.scintillometer.parse_scintillometer(
            file_path=bls_path,
            timezone=tzone,
            calibration=calibrate,
        )
        bls_time = bls_data.index[0]

        transect_data = self.transect.parse_transect(file_path=transect_path)

        weather_data = self.weather.parse_weather(
            timestamp=bls_time,
            source=weather_source,
            station_id=station_id,
            data_dir=weather_dir,
            timezone=tzone,
        )
        interpolated_data = self.stitch.merge_scintillometry_weather(
            scintillometry=bls_data, weather=weather_data
        )

        data_dict = {
            "bls": bls_data,
            "weather": weather_data,
            "transect": transect_data,
            "interpolated": interpolated_data,
            "timestamp": bls_time,
        }

        return data_dict
