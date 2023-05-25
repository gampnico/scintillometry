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

Tests data parsing module.

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

MATLAB® arrays generated from the innFLUX Eddy Covariance code are in a
proprietary format that cannot be mocked. Sample files containing
randomised data are available in `tests/test_data/`.

Use the `conftest_boilerplate` fixture to avoid duplicating tests.
"""

import datetime
import io
import os
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pandas.api.types as ptypes
import pytest

import scintillometry.backend.constants
import scintillometry.wrangler.data_parser


class TestWranglerIO:
    """Test class for file operations."""

    test_wrangler_io = scintillometry.wrangler.data_parser.WranglerIO()

    @pytest.mark.dependency(name="TestWranglerIO::test_check_file_exists")
    def test_check_file_exists(self):
        """Raise error if file not found."""

        wrong_path = "non_existent_file"
        error_message = f"No file found with path: {wrong_path}"
        with pytest.raises(FileNotFoundError, match=error_message):
            self.test_wrangler_io.check_file_exists(fname=wrong_path)

    @pytest.mark.dependency(
        name="TestWranglerIO::test_file_handler_not_found",
        depends=["TestWranglerIO::test_check_file_exists"],
        scope="class",
    )
    def test_file_handler_not_found(self):
        """Raise error if mnd file not found."""

        wrong_path = "non_existent_file"
        error_message = f"No file found with path: {wrong_path}"
        with pytest.raises(FileNotFoundError, match=error_message):
            self.test_wrangler_io.file_handler(filename=wrong_path)

    @pytest.mark.dependency(
        name="TestWranglerIO::test_file_handler_read",
        depends=["TestWranglerIO::test_file_handler_not_found"],
        scope="class",
    )
    @patch("builtins.open")
    def test_file_handler_read(
        self,
        open_mock: Mock,
        conftest_mock_mnd_raw,
        conftest_mock_check_file_exists,
    ):
        """Convert file to list."""

        _ = conftest_mock_check_file_exists
        open_mock.return_value = io.StringIO(conftest_mock_mnd_raw)

        compare_lines = self.test_wrangler_io.file_handler(filename="path/to/file")
        open_mock.assert_called_once()

        assert isinstance(compare_lines, list)
        assert compare_lines[0] == "FORMAT-1.1\n"


class TestWranglerTransform:
    """Tests generic dataframe operations."""

    test_wrangler_frame = scintillometry.wrangler.data_parser.WranglerTransform()

    @pytest.mark.dependency(name="TestWranglerTransform::test_parse_iso_date")
    @pytest.mark.parametrize("arg_date", [True, False])
    def test_parse_iso_date(self, arg_date):
        """Parse timestamp with mixed ISO-8601 duration and date."""

        test_string = "PT00H00M30S/2020-06-03T03:23:00Z"
        compare_string = self.test_wrangler_frame.parse_iso_date(
            x=test_string, date=arg_date
        )
        assert isinstance(compare_string, str)
        assert compare_string != "/"
        if not arg_date:
            assert compare_string == "PT00H00M30S"
        else:
            assert compare_string == "2020-06-03T03:23:00Z"

    @pytest.mark.dependency(name="TestWranglerTransform::test_change_index_frequency")
    @pytest.mark.parametrize("arg_frequency", ["60S", "60s", "30T", "1H"])
    def test_change_index_frequency(self, conftest_boilerplate, arg_frequency):
        """Change index frequency."""

        date_today = datetime.datetime.now()
        hours = pd.date_range(date_today, date_today + datetime.timedelta(24), freq="H")

        random_data = np.random.randint(1, high=100, size=len(hours))
        test_data = pd.DataFrame({"time": hours, "x1": random_data})
        test_data = test_data.set_index("time")
        test_data = test_data.tz_localize("CET")

        compare_data = self.test_wrangler_frame.change_index_frequency(
            data=test_data, frequency=arg_frequency
        )

        conftest_boilerplate.check_dataframe(dataframe=compare_data)
        conftest_boilerplate.check_timezone(dataframe=compare_data, tzone="CET")
        assert compare_data.index.freq == arg_frequency

    @pytest.mark.dependency(name="TestWranglerTransform::test_convert_time_index")
    @pytest.mark.parametrize("arg_timezone", ["CET", "Europe/Berlin", "UTC", None])
    def test_convert_time_index(
        self, conftest_mock_weather_dataframe, conftest_boilerplate, arg_timezone
    ):
        """Convert timezone of index."""

        test_data = conftest_mock_weather_dataframe.copy(deep=True)
        assert not ptypes.is_datetime64_any_dtype(test_data.index)

        compare_data = self.test_wrangler_frame.convert_time_index(
            data=test_data, tzone=arg_timezone
        )
        assert compare_data.index.name == "time"
        assert "time" not in compare_data.columns
        conftest_boilerplate.check_timezone(dataframe=compare_data, tzone=arg_timezone)

    @pytest.mark.dependency(name="TestWranglerTransform::test_pandas_attrs")
    def test_pandas_attrs(self):
        """Ensure experimental pd.DataFrame.attrs is safe."""

        test_data = pd.DataFrame()
        assert "name" not in test_data.attrs
        assert not test_data.attrs

        test_data.attrs["name"] = "Test Name"
        assert isinstance(test_data, pd.DataFrame)
        assert test_data.attrs["name"] == "Test Name"


class TestWranglerScintillometer:
    """Test class for parsing raw BLS data."""

    test_wrangler_scintillometer = (
        scintillometry.wrangler.data_parser.WranglerScintillometer()
    )

    @pytest.mark.dependency(name="TestWranglerScintillometer::test_scintillometer_init")
    def test_scintillometer_init(self):
        """Inherit methods from parent."""

        test_class = scintillometry.wrangler.data_parser.WranglerScintillometer()
        assert test_class
        assert test_class.transform
        assert isinstance(
            test_class.transform, scintillometry.wrangler.data_parser.WranglerTransform
        )
        assert test_class.check_file_exists

    @pytest.mark.dependency(
        name="TestWranglerScintillometer::test_parse_mnd_lines_format",
        depends=["TestWranglerScintillometer::test_scintillometer_init"],
    )
    def test_parse_mnd_lines_format(self, conftest_mock_mnd_raw):
        """Raise error if .mnd file is not FORMAT-1"""

        test_lines = io.StringIO(conftest_mock_mnd_raw).readlines()
        test_lines[0] = "FORMAT-2.1"

        with pytest.raises(Warning, match="The input file does not follow FORMAT-1."):
            self.test_wrangler_scintillometer.parse_mnd_lines(line_list=test_lines)

    @pytest.mark.dependency(
        name="TestWranglerScintillometer::test_parse_mnd_lines",
        depends=["TestWranglerScintillometer::test_parse_mnd_lines_format"],
    )
    def test_parse_mnd_lines(self, conftest_mock_mnd_raw):
        """Parse .mnd file lines."""

        test_lines = io.StringIO(conftest_mock_mnd_raw).readlines()
        compare_data = self.test_wrangler_scintillometer.parse_mnd_lines(
            line_list=test_lines
        )
        assert isinstance(compare_data, dict)  # correct return format
        assert all(
            key in compare_data for key in ("data", "names", "timestamp", "parameters")
        )

        variable_number = int(test_lines[3].partition(" ")[-1].strip())
        assert len(compare_data["names"]) == variable_number  # correct variables
        compare_names = ["time", "Cn2", "CT2", "H_convection", "pressure"]
        assert all(x in compare_names for x in compare_data["names"])

        assert len(compare_data["data"]) == 2  # correct number of rows

        assert isinstance(compare_data["parameters"], dict)  # correct headers
        assert all(
            key in compare_data["parameters"] for key in ("Station Code", "Software")
        )
        assert compare_data["parameters"]["Station Code"] == "Test"
        assert compare_data["parameters"]["Software"] == "SRun 1.49"

    @pytest.mark.dependency(name="TestWranglerScintillometer::test_calibrate_data")
    def test_calibrate_data(self, conftest_mock_bls_dataframe):
        """Recalibrate data from path lengths."""

        test_data = conftest_mock_bls_dataframe
        compare_data = self.test_wrangler_scintillometer.calibrate_data(
            data=test_data.copy(deep=True), path_lengths=[2, 3]  # [incorrect, correct]
        )  # without copy list is modified in place, so test_data == compare_data
        for key in ["Cn2", "H_convection"]:
            test_calib = (
                test_data[key] * (3 ** (-3)) / (2 ** (-3))
            )  # correct / incorrect
            assert ptypes.is_numeric_dtype(compare_data[key])
            assert np.allclose(compare_data[key], test_calib)

    @pytest.mark.dependency(
        name="TestWranglerScintillometer::test_calibrate_data_error",
        depends=["TestWranglerScintillometer::test_calibrate_data"],
    )
    @pytest.mark.parametrize("arg_calibration", [["2", "3", "4"], ["2"]])
    def test_calibrate_data_error(self, conftest_mock_bls_dataframe, arg_calibration):
        """Raise error if calibration is incorrectly formatted."""

        test_data = conftest_mock_bls_dataframe
        error_message = "Calibration path lengths must be formatted as: "
        with pytest.raises(  # incorrect path or missing file raises error
            ValueError, match=error_message
        ):
            self.test_wrangler_scintillometer.calibrate_data(
                data=test_data, path_lengths=arg_calibration
            )

    @pytest.mark.dependency(
        name="TestWranglerScintillometer::test_parse_scintillometer",
        depends=[
            "TestWranglerIO::test_file_handler_read",
            "TestWranglerTransform::test_parse_iso_date",
            "TestWranglerTransform::test_pandas_attrs",
            "TestWranglerTransform::test_convert_time_index",
            "TestWranglerScintillometer::test_scintillometer_init",
            "TestWranglerScintillometer::test_parse_mnd_lines",
            "TestWranglerScintillometer::test_calibrate_data_error",
        ],
        scope="module",
    )
    @patch("builtins.open")
    def test_parse_scintillometer(
        self,
        open_mock: Mock,
        conftest_mock_mnd_raw,
        conftest_mock_check_file_exists,
        conftest_boilerplate,
    ):
        """Parse raw data from BLS450."""

        _ = conftest_mock_check_file_exists
        open_mock.return_value = io.StringIO(conftest_mock_mnd_raw)
        compare_data = self.test_wrangler_scintillometer.parse_scintillometer(
            file_path="path/folder/file", timezone=None, calibration=None
        )
        open_mock.assert_called_once()
        open_mock.reset_mock(return_value=True)

        assert isinstance(compare_data, pd.DataFrame)
        assert "name" in compare_data.attrs
        assert compare_data.attrs["name"] == "Test"

        data_keys = ["Cn2", "CT2", "H_convection", "pressure"]
        conftest_boilerplate.check_dataframe(compare_data[data_keys])
        for key in data_keys:
            assert key in compare_data.columns
        assert "iso_duration" in compare_data.columns
        assert ptypes.is_timedelta64_dtype(compare_data["iso_duration"])

    @pytest.mark.dependency(
        name="TestWranglerScintillometer::test_parse_scintillometer_args",
        depends=["TestWranglerScintillometer::test_parse_scintillometer"],
    )
    @pytest.mark.parametrize("arg_timezone", ["CET", "UTC", None, "Europe/Berlin"])
    @pytest.mark.parametrize("arg_calibration", [[2, 3], None])
    @pytest.mark.parametrize("arg_station", [True, False])
    @patch("pandas.read_table")
    @patch("builtins.open")
    def test_parse_scintillometer_args(
        self,
        open_mock: Mock,
        read_table_mock: Mock,
        conftest_mock_mnd_raw,
        conftest_mock_bls_dataframe,
        conftest_mock_check_file_exists,
        conftest_boilerplate,
        arg_timezone,
        arg_calibration,
        arg_station,
    ):
        """Parse raw data from BLS450."""

        _ = conftest_mock_check_file_exists
        if arg_station:
            test_mnd_raw = conftest_mock_mnd_raw
        else:
            test_mnd_raw = conftest_mock_mnd_raw.replace("Station Code:     Test", "")
        open_mock.return_value = io.StringIO(test_mnd_raw)
        read_table_mock.return_value = conftest_mock_bls_dataframe.copy(deep=True)

        test_data = conftest_mock_bls_dataframe.copy(deep=True)
        compare_data = self.test_wrangler_scintillometer.parse_scintillometer(
            file_path="path/folder/file",
            timezone=arg_timezone,
            calibration=arg_calibration,
        )
        read_table_mock.assert_called_once()
        open_mock.reset_mock(return_value=True)
        read_table_mock.reset_mock(return_value=True)

        conftest_boilerplate.check_dataframe(dataframe=compare_data)
        assert compare_data.index[0].strftime("%Y-%m-%d") == "2020-06-03"
        conftest_boilerplate.check_timezone(dataframe=compare_data, tzone=arg_timezone)
        if compare_data.index.tzinfo == datetime.timezone.utc:
            assert compare_data.index[0].strftime("%H:%M") == "03:23"

        if arg_calibration:
            for key in ["Cn2", "H_convection"]:
                test_calib = (
                    test_data[key]
                    * (arg_calibration[1] ** (-3))
                    / (arg_calibration[0] ** (-3))
                )
                assert np.allclose(compare_data[key], test_calib)

        if not arg_station:
            assert "name" not in compare_data.attrs
            assert not compare_data.attrs
        else:
            compare_data.attrs["name"] = "Test Name"
            conftest_boilerplate.check_dataframe(dataframe=compare_data)
            assert compare_data.attrs["name"] == "Test Name"


class TestWranglerTransect:
    """Test class for parsing path transects."""

    test_wrangler_transect = scintillometry.wrangler.data_parser.WranglerTransect()

    @pytest.mark.dependency(name="TestWranglerTransect::test_transect_init")
    def test_transect_init(self):
        """Inherit methods from parent."""

        test_class = scintillometry.wrangler.data_parser.WranglerTransect()
        assert test_class
        assert test_class.transform
        assert isinstance(
            test_class.transform, scintillometry.wrangler.data_parser.WranglerTransform
        )
        assert test_class.check_file_exists

    @pytest.mark.dependency(
        name="TestWranglerTransect::test_parse_dgm_processed_file_not_found",
        depends=["TestWranglerIO::test_check_file_exists"],
        scope="module",
    )
    def test_parse_dgm_processed_file_not_found(self):
        """Raise error if DGM file not found."""

        with pytest.raises(FileNotFoundError):
            self.test_wrangler_transect.parse_dgm_processed(file_path="wrong/file")

    @pytest.mark.dependency(
        name="TestWranglerTransect::test_parse_dgm_processed_out_of_range"
    )
    @pytest.mark.parametrize("arg_position", [-0.9, 1.01, np.nan])
    @patch("pandas.read_csv")
    def test_parse_dgm_processed_out_of_range(
        self,
        read_csv_mock: Mock,
        conftest_mock_transect_dataframe,
        conftest_mock_check_file_exists,
        arg_position,
    ):
        """Raise error if normalised position is out of range."""

        _ = conftest_mock_check_file_exists
        test_transect = conftest_mock_transect_dataframe
        test_transect["norm_position"][0] = arg_position

        error_msg = "Normalised position is not between 0 and 1."
        with pytest.raises(ValueError, match=error_msg):
            read_csv_mock.return_value = test_transect
            self.test_wrangler_transect.parse_dgm_processed(file_path="wrong/file")
        read_csv_mock.assert_called_once()

    @pytest.mark.dependency(
        name="TestWranglerTransect::test_parse_dgm_processed",
        depends=[
            "TestWranglerTransect::test_parse_dgm_processed_file_not_found",
            "TestWranglerTransect::test_parse_dgm_processed_out_of_range",
        ],
    )
    @patch("pandas.read_csv")
    def test_parse_dgm_processed(
        self,
        read_csv_mock: Mock,
        conftest_mock_transect_dataframe,
        conftest_mock_check_file_exists,
    ):
        """Parse pre-processed DGM file into dataframe."""

        _ = conftest_mock_check_file_exists
        read_csv_mock.return_value = conftest_mock_transect_dataframe
        compare_dataframe = self.test_wrangler_transect.parse_dgm_processed(
            file_path="/path/to/file"
        )
        read_csv_mock.assert_called_once()
        assert isinstance(compare_dataframe, pd.DataFrame)
        for key in compare_dataframe.keys():
            assert key in ["path_height", "norm_position"]
            assert ptypes.is_numeric_dtype(compare_dataframe[key])

        assert all(compare_dataframe["norm_position"].between(0, 1, "both"))

    @pytest.mark.dependency(name="TestWranglerTransect::test_parse_transect_error")
    @pytest.mark.parametrize("arg_source", ["wrong source", "wrong_SOURCE"])
    def test_parse_transect_error(self, arg_source):
        """Raise error if transect data source is unsupported."""

        test_source = arg_source.title()
        error_msg = (
            f"{test_source}",
            "measurements are not supported. Use 'dgm_processed'.",
        )

        with pytest.raises(NotImplementedError, match=" ".join(error_msg)):
            self.test_wrangler_transect.parse_transect(
                file_path="/path/to/file", source=test_source
            )

    @pytest.mark.dependency(
        name="TestWranglerTransect::test_parse_transect",
        depends=[
            "TestWranglerTransect::test_parse_dgm_processed",
            "TestWranglerTransect::test_parse_transect_error",
        ],
    )
    @pytest.mark.parametrize("arg_source", ["dgm_processed"])
    @patch("pandas.read_csv")
    def test_parse_transect(
        self,
        read_csv_mock: Mock,
        conftest_mock_transect_dataframe,
        conftest_mock_check_file_exists,
        arg_source,
    ):
        """Parse transect data file into dataframe."""

        _ = conftest_mock_check_file_exists
        read_csv_mock.return_value = conftest_mock_transect_dataframe
        compare_dataframe = self.test_wrangler_transect.parse_transect(
            file_path="/path/to/file", source=arg_source
        )
        read_csv_mock.assert_called_once()
        assert isinstance(compare_dataframe, pd.DataFrame)
        for key in compare_dataframe.keys():
            assert key in ["path_height", "norm_position"]
            assert ptypes.is_numeric_dtype(compare_dataframe[key])

        assert all(compare_dataframe["norm_position"].between(0, 1, "both"))


class TestWranglerWeather:
    """Tests the parsing of weather observations."""

    test_wrangler_weather = scintillometry.wrangler.data_parser.WranglerWeather()

    @pytest.mark.dependency(name="TestWranglerWeather::test_weather_init")
    def test_weather_init(self):
        """Inherit methods from parent."""

        test_class = scintillometry.wrangler.data_parser.WranglerWeather()
        assert test_class
        assert test_class.transform
        assert isinstance(
            test_class.transform, scintillometry.wrangler.data_parser.WranglerTransform
        )
        assert test_class.check_file_exists

    @pytest.mark.dependency(name="TestWranglerWeather::test_parse_zamg_data")
    @pytest.mark.parametrize(
        "arg_timestamp", ["2020-06-03T00:00:00Z", "2020-06-03T03:23:00Z"]
    )
    @pytest.mark.parametrize("arg_name", [None, "rand_var"])
    @patch("pandas.read_csv")
    def test_parse_zamg_data(
        self,
        read_csv_mock: Mock,
        conftest_mock_weather_raw,
        conftest_mock_check_file_exists,
        conftest_boilerplate,
        arg_timestamp,
        arg_name,
    ):
        """Parse ZAMG data to dataframe."""

        _ = conftest_mock_check_file_exists
        test_timestamp = pd.to_datetime(arg_timestamp)
        test_station_id = "0000"
        assert isinstance(test_timestamp, pd.Timestamp)

        if not arg_name:
            test_weather = conftest_mock_weather_raw
        else:
            test_weather = conftest_mock_weather_raw.rename(columns={"RR": arg_name})
            assert arg_name in test_weather.columns
            assert "RR" not in test_weather.columns
        read_csv_mock.return_value = test_weather

        compare_data = self.test_wrangler_weather.parse_zamg_data(
            timestamp=test_timestamp,
            data_dir="path/directory/",  # mock prefix
            klima_id=test_station_id,
            timezone=None,
        )
        read_csv_mock.assert_called_once()
        read_csv_mock.reset_mock(return_value=True)

        assert "station" in compare_data
        conftest_boilerplate.check_dataframe(
            dataframe=compare_data.drop("station", axis=1)
        )
        assert isinstance(compare_data.index, pd.DatetimeIndex)
        assert all(  # renamed columns
            x not in compare_data.columns
            for x in ["DD", "FF", "FAM", "GSX", "P", "RF", "RR", "TL"]
        )
        if not arg_name:
            assert "precipitation" in compare_data.columns
        else:
            assert "RR" not in compare_data.columns
            assert arg_name in compare_data.columns
        assert ptypes.is_string_dtype(compare_data["station"])
        assert all(station == "0000" for station in compare_data["station"])
        assert compare_data.index.resolution == "minute"

    @pytest.mark.dependency(name="TestWranglerWeather::test_parse_weather_error")
    @pytest.mark.parametrize("arg_source", ["wrong source", "wrong_SOURCE"])
    def test_parse_weather_error(self, arg_source):
        """Raise error if weather data source is unsupported."""

        test_source = arg_source.title()
        error_msg = f"{test_source} measurements are not supported. Use 'zamg'."
        test_timestamp = pd.to_datetime("2020-06-03T00:00:00Z")
        assert isinstance(test_timestamp, pd.Timestamp)

        with pytest.raises(NotImplementedError, match=error_msg):
            self.test_wrangler_weather.parse_weather(
                timestamp=test_timestamp,
                data_dir="/path/directory/",
                source=test_source,
            )

    @pytest.mark.dependency(name="TestWranglerWeather::test_parse_zamg_data")
    @pytest.mark.parametrize("arg_source", ["zamg"])
    @patch("pandas.read_csv")
    def test_parse_weather(
        self,
        read_csv_mock: Mock,
        conftest_mock_weather_raw,
        conftest_mock_check_file_exists,
        conftest_boilerplate,
        arg_source,
    ):
        """Parse ZAMG data to dataframe."""

        _ = conftest_mock_check_file_exists
        test_timestamp = pd.to_datetime("2020-06-03T00:00:00Z")
        assert isinstance(test_timestamp, pd.Timestamp)
        test_station_id = "0000"

        test_weather = conftest_mock_weather_raw.copy(deep=True)
        read_csv_mock.return_value = test_weather

        compare_data = self.test_wrangler_weather.parse_weather(
            timestamp=test_timestamp,
            source=arg_source,
            data_dir="path/directory/",  # mock prefix
            station_id=test_station_id,
            timezone=None,
        )
        read_csv_mock.assert_called_once()
        read_csv_mock.reset_mock(return_value=True)

        assert "station" in compare_data
        conftest_boilerplate.check_dataframe(
            dataframe=compare_data.drop("station", axis=1)
        )
        assert isinstance(compare_data.index, pd.DatetimeIndex)
        assert all(  # renamed columns
            x not in compare_data.columns
            for x in ["DD", "FF", "FAM", "GSX", "P", "RF", "RR", "TL"]
        )
        assert "precipitation" in compare_data.columns
        assert ptypes.is_string_dtype(compare_data["station"])
        assert all(station == "0000" for station in compare_data["station"])


class TestWranglerEddy:
    """Test class for parsing eddy covariance data.

    Saved MATLAB® arrays are in a proprietary format, which cannot be
    mocked. Test files are placed in `tests/test_data/`.

    Attributes:
        test_headers (list): Column headers for simulated innFLUX data.
    """

    test_wrangler_eddy = scintillometry.wrangler.data_parser.WranglerEddy()
    test_headers = [
        "year",
        "month",
        "day",
        "hour",
        "minutes",
        "seconds",
        "shf",
        "wind_speed",
        "obukhov",
    ]

    @pytest.mark.dependency(name="TestWranglerEddy::test_eddy_init")
    def test_eddy_init(self):
        """Inherit methods from parent."""

        test_class = scintillometry.wrangler.data_parser.WranglerEddy()
        assert test_class
        assert test_class.transform
        assert isinstance(
            test_class.transform, scintillometry.wrangler.data_parser.WranglerTransform
        )
        assert test_class.check_file_exists  # inherit WranglerIO

    @pytest.mark.dependency(name="TestWranglerEddy::test_validate_data")
    @pytest.mark.parametrize("arg_format", ["v7"])
    def test_validate_data(self, arg_format):
        """Check sample files exist in test path."""

        test_path = f"./tests/test_data/test_data_{arg_format}_empty.mat"
        assert os.path.exists(test_path)

    @pytest.mark.dependency(
        name="TestWranglerEddy::test_parse_innflux_mat_missing",
        depends=["TestWranglerEddy::test_validate_data"],
    )
    @pytest.mark.parametrize("arg_format", ["v7"])
    def test_parse_innflux_mat_missing(self, arg_format):
        """Raise error for missing fields."""

        test_path = f"./tests/test_data/test_data_{arg_format}_empty.mat"

        error_message = "InnFLUX data does not contain any values for MET."
        with pytest.raises(KeyError, match=error_message):
            self.test_wrangler_eddy.parse_innflux_mat(file_path=test_path)

    @pytest.mark.dependency(
        name="TestWranglerEddy::test_parse_innflux_mat_format",
        depends=["TestWranglerEddy::test_validate_data"],
    )
    def test_parse_innflux_mat_format(self):
        """Raise error for file with no .mat extension."""

        test_path = "/path/incorrect/extension.foo"
        error_message = "File does not have a .mat extension."
        with pytest.raises(ValueError, match=error_message):
            self.test_wrangler_eddy.parse_innflux_mat(file_path=test_path)

    @pytest.mark.dependency(
        name="TestWranglerEddy::test_parse_innflux_mat",
        depends=["TestWranglerEddy::test_validate_data"],
    )
    @pytest.mark.parametrize("arg_format", ["v7"])
    def test_parse_innflux_mat(self, conftest_boilerplate, arg_format):
        """Parse .mat file generated by innFLUX."""

        test_path = f"./tests/test_data/test_data_{arg_format}_results.mat"
        compare_mat = self.test_wrangler_eddy.parse_innflux_mat(file_path=test_path)
        test_names = {
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
        test_timestamps = pd.to_datetime(
            ["2020-06-03T00:00:00", "2020-06-03T23:00:00"], utc=False
        )

        conftest_boilerplate.check_dataframe(dataframe=compare_mat)
        assert all(name in test_names.values() for name in compare_mat.columns)
        assert "invalid_key" not in compare_mat.columns
        assert compare_mat.index.resolution == "minute"
        assert compare_mat.index[0] == test_timestamps[0]
        assert compare_mat.index[-1] == test_timestamps[-1]

    @pytest.mark.dependency(name="TestWranglerEddy::test_parse_innflux_csv")
    @pytest.mark.parametrize("arg_header", [True, False])
    @patch("pandas.read_csv")
    def test_parse_innflux_csv(
        self,
        read_csv_mock: Mock,
        conftest_mock_innflux_dataframe,
        conftest_mock_check_file_exists,
        conftest_boilerplate,
        arg_header,
    ):
        """Parse pre-processed innFLUX csv data."""

        _ = conftest_mock_check_file_exists
        read_csv_mock.return_value = conftest_mock_innflux_dataframe

        if arg_header:
            headers = self.test_headers
        else:
            headers = None
        compare_dataframe = self.test_wrangler_eddy.parse_innflux_csv(
            file_path="/path/innflux/file.csv", header_list=headers
        )
        read_csv_mock.assert_called_once()
        conftest_boilerplate.check_dataframe(dataframe=compare_dataframe)

        for key in ["year", "month", "day", "hour", "minutes", "seconds"]:
            assert key not in compare_dataframe.columns
        data_keys = ["shf", "wind_speed", "obukhov"]
        assert all(key in compare_dataframe for key in data_keys)

    @pytest.mark.dependency(
        name="TestWranglerEddy::test_parse_innflux",
        depends=[
            "TestWranglerEddy::test_validate_data",
            "TestWranglerEddy::test_parse_innflux_mat",
            "TestWranglerEddy::test_parse_innflux_csv",
        ],
    )
    @pytest.mark.parametrize("arg_timezone", ["CET", "Europe/Berlin", None])
    @pytest.mark.parametrize("arg_file", [".csv", ".mat"])
    @patch("pandas.read_csv")
    def test_parse_innflux(
        self,
        read_csv_mock: Mock,
        conftest_mock_innflux_dataframe,
        conftest_mock_check_file_exists,
        conftest_boilerplate,
        arg_timezone,
        arg_file,
    ):
        """Parse innFLUX data."""

        if arg_file == ".csv":
            _ = conftest_mock_check_file_exists
            read_csv_mock.return_value = conftest_mock_innflux_dataframe
            compare_dataframe = self.test_wrangler_eddy.parse_innflux(
                file_name="/path/innflux/file.csv",
                timezone=arg_timezone,
            )
            read_csv_mock.assert_called_once()
        else:
            compare_dataframe = self.test_wrangler_eddy.parse_innflux(
                file_name="./tests/test_data/test_data_v7_results.mat",
                timezone=arg_timezone,
            )

        conftest_boilerplate.check_dataframe(dataframe=compare_dataframe)
        conftest_boilerplate.check_timezone(
            dataframe=compare_dataframe, tzone=arg_timezone
        )
        assert compare_dataframe.index.resolution == "minute"

    @pytest.mark.dependency(name="TestWranglerEddy::test_parse_eddy_covariance_error")
    @pytest.mark.parametrize("arg_source", ["wrong source", "wrong_SOURCE"])
    def test_parse_eddy_covariance_error(self, arg_source):
        """Raise error if eddy covariance source is unsupported."""

        test_source = arg_source.title()
        error_msg = f"{test_source} measurements are not supported. Use 'innflux'."

        with pytest.raises(NotImplementedError, match=error_msg):
            self.test_wrangler_eddy.parse_eddy_covariance(
                file_path="/path/to/file", source=test_source, tzone=None
            )

    @pytest.mark.dependency(
        name="TestWranglerEddy::test_parse_eddy_covariance",
        depends=[
            "TestWranglerEddy::test_parse_innflux",
            "TestWranglerEddy::test_parse_eddy_covariance_error",
        ],
    )
    @pytest.mark.parametrize("arg_timezone", ["CET", None])
    @patch("pandas.read_csv")
    def test_parse_eddy_covariance(
        self,
        read_csv_mock: Mock,
        conftest_mock_innflux_dataframe,
        conftest_mock_check_file_exists,
        conftest_boilerplate,
        arg_timezone,
    ):
        """Parse eddy covariance data."""

        _ = conftest_mock_check_file_exists
        read_csv_mock.return_value = conftest_mock_innflux_dataframe

        compare_dataframe = self.test_wrangler_eddy.parse_eddy_covariance(
            file_path="/path/to/file", source="innflux", tzone=arg_timezone
        )
        read_csv_mock.assert_called_once()

        conftest_boilerplate.check_dataframe(dataframe=compare_dataframe)
        conftest_boilerplate.check_timezone(
            dataframe=compare_dataframe, tzone=arg_timezone
        )


class TestWranglerVertical:
    """Test class for parsing vertical measurements."""

    test_wrangler_vertical = scintillometry.wrangler.data_parser.WranglerVertical()

    @pytest.mark.dependency(name="TestWranglerVertical::test_vertical_init")
    def test_vertical_init(self):
        """Inherit methods from parent."""

        test_class = scintillometry.wrangler.data_parser.WranglerVertical()
        assert test_class
        assert test_class.transform
        assert isinstance(
            test_class.transform, scintillometry.wrangler.data_parser.WranglerTransform
        )
        assert test_class.check_file_exists

    @pytest.mark.dependency(
        name="TestWranglerVertical::test_construct_hatpro_levels_error"
    )
    @pytest.mark.parametrize("arg_levels", [[(0, 1), 0], [1.0, 30]])
    def test_construct_hatpro_levels_error(self, arg_levels):
        """Raise error for incorrectly formatted scanning levels."""

        error_message = "Input levels must be a list or tuple of integers."
        with pytest.raises(TypeError, match=error_message):
            self.test_wrangler_vertical.construct_hatpro_levels(levels=arg_levels)

    @pytest.mark.dependency(
        name="TestWranglerVertical::test_construct_hatpro_levels",
        depends=["TestWranglerVertical::test_construct_hatpro_levels_error"],
    )
    @pytest.mark.parametrize("arg_levels", [None, (0, 10, 20), [0, 10, 20]])
    def test_construct_hatpro_levels(self, arg_levels):
        """Construct HATPRO scanning levels."""

        compare_scan = self.test_wrangler_vertical.construct_hatpro_levels(
            levels=arg_levels
        )
        assert isinstance(compare_scan, list)
        assert all(isinstance(x, int) for x in compare_scan)

    @pytest.mark.dependency(
        name="TestWranglerVertical::test_load_hatpro",
        depends=[
            "TestWranglerTransform::test_pandas_attrs",
            "TestWranglerVertical::test_vertical_init",
            "TestWranglerVertical::test_construct_hatpro_levels",
        ],
        scope="module",
    )
    @pytest.mark.parametrize("arg_timezone", ["CET", "UTC", None])
    @patch("builtins.open")
    def test_load_hatpro(
        self,
        open_mock: Mock,
        conftest_mock_hatpro_humidity_raw,
        conftest_mock_check_file_exists,
        conftest_mock_hatpro_scan_levels,
        conftest_boilerplate,
        arg_timezone,
    ):
        """Load raw HATPRO data into dataframe."""

        _ = conftest_mock_check_file_exists
        test_levels = conftest_mock_hatpro_scan_levels
        open_mock.return_value = io.StringIO(conftest_mock_hatpro_humidity_raw)
        test_elevation = 612

        compare_data = self.test_wrangler_vertical.load_hatpro(
            file_name="/path/to/file",
            levels=test_levels,
            tzone=arg_timezone,
            station_elevation=test_elevation,
        )
        open_mock.assert_called_once()

        conftest_boilerplate.check_dataframe(dataframe=compare_data)
        conftest_boilerplate.check_timezone(dataframe=compare_data, tzone=arg_timezone)
        assert compare_data.index[0].year == 2020
        assert compare_data.index.resolution == "minute"

        for key in test_levels:
            assert key in compare_data.columns
        assert "elevation" in compare_data.attrs
        assert isinstance(compare_data.attrs["elevation"], int)
        assert compare_data.attrs["elevation"] == test_elevation

    @pytest.mark.dependency(
        name="TestWranglerVertical::test_parse_hatpro",
        depends=["TestWranglerVertical::test_load_hatpro"],
    )
    @pytest.mark.parametrize("arg_timezone", ["CET", "Europe/Berlin", "UTC", None])
    @patch("pandas.read_csv")
    def test_parse_hatpro(
        self,
        read_csv_mock: Mock,
        conftest_mock_hatpro_humidity_dataframe,
        conftest_mock_hatpro_temperature_dataframe,
        conftest_mock_hatpro_scan_levels,
        conftest_mock_check_file_exists,
        conftest_boilerplate,
        arg_timezone,
    ):
        """Parse unformatted HATPRO data."""

        _ = conftest_mock_check_file_exists
        test_levels = conftest_mock_hatpro_scan_levels
        read_csv_mock.side_effect = [
            conftest_mock_hatpro_humidity_dataframe,
            conftest_mock_hatpro_temperature_dataframe,
        ]

        compare_data = self.test_wrangler_vertical.parse_hatpro(
            file_prefix="/path/to/file",
            scan_heights=test_levels,
            timezone=arg_timezone,
        )

        assert isinstance(compare_data, dict)
        for frame_key, compare_frame in compare_data.items():
            assert frame_key in ["humidity", "temperature"]
            conftest_boilerplate.check_dataframe(dataframe=compare_frame)
            conftest_boilerplate.check_timezone(
                dataframe=compare_frame, tzone=arg_timezone
            )
            for key in test_levels:
                assert key in compare_frame.columns
            assert compare_frame.index.name == "rawdate"
        assert all(compare_data["humidity"]) < 0.007  # should be in |kgm^-3|

    @pytest.mark.dependency(name="TestWranglerVertical::test_parse_vertical_error")
    @pytest.mark.parametrize("arg_source", ["wrong source", "wrong_SOURCE"])
    def test_parse_vertical_error(self, conftest_mock_hatpro_scan_levels, arg_source):
        """Raise error if vertical measurement source is unsupported."""

        test_source = arg_source.title()
        test_levels = conftest_mock_hatpro_scan_levels
        error_msg = f"{test_source} measurements are not supported. Use 'hatpro'."

        with pytest.raises(NotImplementedError, match=error_msg):
            self.test_wrangler_vertical.parse_vertical(
                file_path="/path/to/file",
                source=test_source,
                levels=test_levels,
                tzone=None,
            )

    @pytest.mark.dependency(
        name="TestWranglerVertical::test_parse_vertical",
        depends=[
            "TestWranglerVertical::test_parse_hatpro",
            "TestWranglerVertical::test_parse_vertical_error",
        ],
    )
    @pytest.mark.parametrize("arg_timezone", ["CET", None])
    @patch("pandas.read_csv")
    def test_parse_vertical(
        self,
        read_csv_mock: Mock,
        conftest_mock_hatpro_humidity_dataframe,
        conftest_mock_hatpro_temperature_dataframe,
        conftest_mock_hatpro_scan_levels,
        conftest_mock_check_file_exists,
        conftest_boilerplate,
        arg_timezone,
    ):
        """Parse vertical measurements."""

        _ = conftest_mock_check_file_exists
        test_levels = conftest_mock_hatpro_scan_levels
        read_csv_mock.side_effect = [
            conftest_mock_hatpro_humidity_dataframe,
            conftest_mock_hatpro_temperature_dataframe,
        ]

        compare_data = self.test_wrangler_vertical.parse_vertical(
            file_path="/path/to/file",
            source="hatpro",
            levels=test_levels,
            tzone=arg_timezone,
        )

        assert isinstance(compare_data, dict)
        for frame_key, compare_frame in compare_data.items():
            assert frame_key in ["humidity", "temperature"]
            conftest_boilerplate.check_dataframe(dataframe=compare_frame)
            conftest_boilerplate.check_timezone(
                dataframe=compare_frame, tzone=arg_timezone
            )
            for key in test_levels:
                assert key in compare_frame.columns


class TestWranglerStitch:
    """Tests class for merging data."""

    test_wrangler_stitch = scintillometry.wrangler.data_parser.WranglerStitch()

    @pytest.mark.dependency(name="TestWranglerStitch::test_stitch_init")
    def test_stitch_init(self):
        """Inherit methods from parent."""

        test_class = scintillometry.wrangler.data_parser.WranglerStitch()
        assert test_class
        assert test_class.constants
        assert isinstance(
            test_class.constants, scintillometry.backend.constants.AtmosConstants
        )

    @pytest.mark.dependency(
        name="TestWranglerStitch::test_merge_scintillometry_weather",
        depends=[
            "TestWranglerScintillometer::test_parse_scintillometer",
            "TestWranglerWeather::test_parse_zamg_data",
        ],
        scope="module",
    )
    def test_merge_scintillometry_weather(
        self,
        conftest_mock_bls_dataframe_tz,
        conftest_mock_weather_dataframe_tz,
        conftest_boilerplate,
    ):
        """Merge scintillometry and weather data."""

        test_bls = conftest_mock_bls_dataframe_tz
        test_weather = conftest_mock_weather_dataframe_tz

        assert test_bls.index.resolution == test_weather.index.resolution
        compare_merged = self.test_wrangler_stitch.merge_scintillometry_weather(
            scintillometry=test_bls, weather=test_weather
        )
        conftest_boilerplate.check_timezone(dataframe=compare_merged, tzone="CET")
        assert "station" in compare_merged
        conftest_boilerplate.check_dataframe(
            dataframe=compare_merged.drop("station", axis=1)
        )
        for key in test_weather.columns:
            assert key in compare_merged.columns
        for key in ["Cn2", "H_convection"]:
            assert key in compare_merged.columns

        assert not (compare_merged["temperature_2m"].lt(100)).any()
        assert not (compare_merged["pressure"].gt(2000)).any()

    @pytest.mark.dependency(
        name="TestWranglerStitch::test_merge_scintillometry_weather_convert",
        depends=["TestWranglerStitch::test_merge_scintillometry_weather"],
    )
    @pytest.mark.parametrize("arg_temp", [273.15, 0])
    @pytest.mark.parametrize("arg_pressure", [100, 1])
    def test_merge_scintillometry_weather_convert(
        self,
        conftest_mock_bls_dataframe_tz,
        conftest_mock_weather_dataframe_tz,
        conftest_boilerplate,
        arg_temp,
        arg_pressure,
    ):
        """Merge scintillometry and weather data and convert units."""

        test_bls = conftest_mock_bls_dataframe_tz
        test_weather = conftest_mock_weather_dataframe_tz
        test_weather["temperature_2m"] = test_weather["temperature_2m"] + arg_temp
        test_weather["pressure"] = test_weather["pressure"] * arg_pressure

        compare_merged = self.test_wrangler_stitch.merge_scintillometry_weather(
            scintillometry=test_bls, weather=test_weather
        )

        assert "station" in compare_merged
        conftest_boilerplate.check_dataframe(
            dataframe=compare_merged.drop("station", axis=1)
        )
        for key in test_weather.columns:
            assert key in compare_merged.columns
        assert not (compare_merged["temperature_2m"].lt(100)).any()
        assert not (compare_merged["pressure"].gt(2000)).any()


class TestWranglerParsing:
    """Tests wrapper for parsing methods."""

    test_wrangler_parsing = scintillometry.wrangler.data_parser.WranglerParsing()

    @pytest.mark.dependency(name="TestWranglerParsing::test_parsing_init")
    def test_parsing_init(self):
        """Inherit methods from parent."""

        test_class = scintillometry.wrangler.data_parser.WranglerParsing()
        assert test_class
        assert test_class.scintillometer
        assert isinstance(
            test_class.scintillometer,
            scintillometry.wrangler.data_parser.WranglerScintillometer,
        )
        assert test_class.weather
        assert isinstance(
            test_class.weather, scintillometry.wrangler.data_parser.WranglerWeather
        )
        assert test_class.transect
        assert isinstance(
            test_class.transect, scintillometry.wrangler.data_parser.WranglerTransect
        )
        assert test_class.eddy
        assert isinstance(
            test_class.eddy, scintillometry.wrangler.data_parser.WranglerEddy
        )
        assert test_class.vertical
        assert isinstance(
            test_class.vertical, scintillometry.wrangler.data_parser.WranglerVertical
        )
        assert test_class.stitch
        assert isinstance(
            test_class.stitch, scintillometry.wrangler.data_parser.WranglerStitch
        )
        assert test_class.check_file_exists

    @pytest.mark.dependency(
        name="TestWranglerParsing::test_wrangle_data",
        depends=[
            "TestWranglerTransect::test_parse_transect",
            "TestWranglerScintillometer::test_parse_scintillometer",
            "TestWranglerWeather::test_parse_zamg_data",
            "TestWranglerStitch::test_merge_scintillometry_weather_convert",
            "TestWranglerParsing::test_parsing_init",
        ],
        session="module",
    )
    def test_wrangle_data(
        self,
        conftest_mock_bls_dataframe_tz,
        conftest_mock_transect_dataframe,
        conftest_mock_weather_dataframe_tz,
        conftest_boilerplate,
    ):
        """Parse BLS, weather, and transect data."""

        parse_scintillometer_mock = patch.object(
            scintillometry.wrangler.data_parser.WranglerScintillometer,
            "parse_scintillometer",
        ).start()
        parse_scintillometer_mock.return_value = conftest_mock_bls_dataframe_tz.copy(
            deep=True
        )
        parse_transect_mock = patch.object(
            scintillometry.wrangler.data_parser.WranglerTransect, "parse_transect"
        ).start()
        parse_transect_mock.return_value = conftest_mock_transect_dataframe.copy(
            deep=True
        )
        parse_weather_mock = patch.object(
            scintillometry.wrangler.data_parser.WranglerWeather, "parse_weather"
        ).start()
        parse_weather_mock.return_value = conftest_mock_weather_dataframe_tz.copy(
            deep=True
        )

        compare_dict = self.test_wrangler_parsing.wrangle_data(
            bls_path="/path/to/bls/file",
            transect_path="/path/to/transect/file",
            calibrate=None,
            weather_dir="/path/to/zamg/directory/",
            station_id="0000",
            tzone="CET",
            weather_source="zamg",
        )

        assert isinstance(compare_dict, dict)
        assert "timestamp" in compare_dict
        assert isinstance(compare_dict["timestamp"], pd.Timestamp)
        assert compare_dict["timestamp"].tz.zone == "CET"

        test_keys = ["bls", "weather", "interpolated", "transect"]
        for key in test_keys:
            assert key in compare_dict
            assert isinstance(compare_dict[key], pd.DataFrame)
        for key in test_keys[:-1]:  # time-indexed dataframes only
            assert ptypes.is_datetime64_any_dtype(compare_dict[key].index)
            conftest_boilerplate.check_timezone(
                dataframe=compare_dict[key], tzone="CET"
            )
