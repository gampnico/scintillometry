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

Tests data parsing module.
"""

import numpy as np
import pandas as pd
import pandas.api.types as ptypes
import pytest

import scintillometry.wrangler.data_parser


class TestFileHandling:
    """Test class for file handling functions."""

    @pytest.mark.dependency(
        name="TestFileHandling::test_file_handler_file_not_found",
        scope="class",
    )
    def test_file_handler_file_not_found(self):
        """Raise error if file not found."""

        wrong_path = "non_existent_file"
        error_message = f"No file found with path: {wrong_path}"
        with pytest.raises(  # incorrect path or missing file raises error
            FileNotFoundError, match=error_message
        ):
            scintillometry.wrangler.data_parser.file_handler(filename=wrong_path)

    @pytest.mark.dependency(
        name="TestFileHandling::test_file_handler_read",
        depends=["TestFileHandling::test_file_handler_file_not_found"],
        scope="module",
    )
    def test_file_handler_read(self, conftest_mnd_path):
        """Convert file to list."""

        compare_lines = scintillometry.wrangler.data_parser.file_handler(
            filename=conftest_mnd_path
        )
        assert isinstance(compare_lines, list)
        assert compare_lines[0] == "FORMAT-1.1\n"


class TestDataParsingBLS:
    """Test class for parsing raw BLS data."""

    @pytest.mark.dependency(
        name="TestDataParsingBLS::test_parse_mnd_lines_format",
        scope="class",
    )
    def test_parse_mnd_lines_format(self, conftest_create_test_data):
        """Raise error if .mnd file is not FORMAT-1"""

        test_data = conftest_create_test_data.setup_mnd_file()
        mnd_lines = test_data.data.readlines()
        mnd_lines[0] = "FORMAT-2.1"

        with pytest.raises(Warning, match="The input file does not follow FORMAT-1."):
            scintillometry.wrangler.data_parser.parse_mnd_lines(line_list=mnd_lines)

    @pytest.mark.dependency(
        name="TestDataParsingBLS::test_parse_mnd_lines",
        depends=["TestDataParsingBLS::test_parse_mnd_lines_format"],
        scope="class",
    )
    def test_parse_mnd_lines(self, conftest_create_test_data):
        """Parse .mnd file lines."""

        test_data = conftest_create_test_data.setup_mnd_file()
        test_lines = test_data.data.readlines()
        compare_data = scintillometry.wrangler.data_parser.parse_mnd_lines(
            line_list=test_lines
        )
        assert isinstance(compare_data, dict)  # correct return format
        assert all(
            key in compare_data for key in ("data", "names", "timestamp", "parameters")
        )

        variable_number = int(test_lines[3].partition(" ")[-1].strip())
        assert len(compare_data["names"]) == variable_number  # correct variables
        assert compare_data["names"] == test_data.variables

        assert len(compare_data["data"]) == 2  # correct number of rows

    @pytest.mark.dependency(
        name="TestDataParsingBLS::test_parse_iso_date",
        scope="class",
    )
    @pytest.mark.parametrize("arg_date", [True, False])
    def test_parse_iso_date(self, arg_date):
        """Parse timestamp with mixed ISO-8601 duration and date."""

        test_string = "PT00H00M30S/2020-06-03T03:23:00Z"
        compare_string = scintillometry.wrangler.data_parser.parse_iso_date(
            x=test_string, date=arg_date
        )
        assert isinstance(compare_string, str)
        assert compare_string != "/"
        if arg_date is True:
            assert compare_string == "2020-06-03T03:23:00Z"
        else:
            assert compare_string == "PT00H00M30S"

    @pytest.mark.dependency(
        name="TestDataParsingBLS::test_calibrate_data",
        scope="class",
    )
    def test_calibrate_data(self):
        """Recalibrate data from path lengths."""

        test_data = pd.DataFrame(
            data=[[1.03, 2.04, 3], [1.4, 3.1, 4]],
            columns=["Cn2", "H_convection", "Var03"],
        )
        compare_data = scintillometry.wrangler.data_parser.calibrate_data(
            data=test_data.copy(deep=True), path_lengths=[2, 3]  # [incorrect, correct]
        )
        test_calib = (3 ** (-3)) / (2 ** (-3))  # correct / incorrect
        for key in ["Cn2", "H_convection"]:
            assert ptypes.is_numeric_dtype(compare_data[key])
            assert np.allclose(compare_data[key], test_data[key] * test_calib)

    @pytest.mark.dependency(
        name="TestDataParsingBLS::test_calibrate_data_error",
        depends=["TestDataParsingBLS::test_calibrate_data"],
        scope="class",
    )
    @pytest.mark.parametrize("arg_calibration", [["2", "3", "4"], ["2"]])
    def test_calibrate_data_error(self, arg_calibration):
        """Raise error if calibration is incorrectly formatted."""

        test_data = pd.DataFrame(
            data=[[1.03, 2.04, 3], [1.4, 3.1, 4]],
            columns=["Cn2", "H_convection", "Var03"],
        )
        error_message = "Calibration path lengths must be formatted as: "
        with pytest.raises(  # incorrect path or missing file raises error
            ValueError, match=error_message
        ):
            scintillometry.wrangler.data_parser.calibrate_data(
                data=test_data, path_lengths=arg_calibration
            )

    @pytest.mark.dependency(
        name="TestDataParsingBLS::test_parse_scintillometer",
        depends=[
            "TestFileHandling::test_file_handler_read",
            "TestDataParsingBLS::test_parse_iso_date",
            "TestDataParsingBLS::test_parse_mnd_lines",
            "TestDataParsingBLS::test_calibrate_data_error",
        ],
        scope="module",
    )
    @pytest.mark.parametrize("arg_timezone", ["CET", "Europe/Berlin", None])
    def test_parse_scintillometer(self, conftest_mnd_path, arg_timezone):
        """Parse raw data from BLS450."""

        dataframe = scintillometry.wrangler.data_parser.parse_scintillometer(
            file_path=conftest_mnd_path, timezone=arg_timezone, calibration=None
        )

        assert isinstance(dataframe, pd.DataFrame)

        assert dataframe.index.name == "time"
        assert "time" not in dataframe.columns
        assert ptypes.is_datetime64_any_dtype(dataframe.index)

        data_keys = ["Cn2", "CT2", "H_convection", "pressure"]
        for key in data_keys:
            assert key in dataframe.columns
            assert ptypes.is_numeric_dtype(dataframe[key])
        assert "iso_duration" in dataframe.columns
        assert ptypes.is_timedelta64_dtype(dataframe["iso_duration"])

        if arg_timezone:
            assert dataframe.index.tz.zone == arg_timezone
        else:
            assert dataframe.index.tz.zone == "UTC"

    @pytest.mark.dependency(
        name="TestDataParsingBLS::test_parse_scintillometer_calibration",
        depends=[
            "TestDataParsingBLS::test_parse_scintillometer",
            "TestDataParsingBLS::test_calibrate_data_error",
        ],
        scope="class",
    )
    @pytest.mark.parametrize("arg_calibration", [[2, 3], None])
    def test_parse_scintillometer_calibration(self, conftest_mnd_path, arg_calibration):
        """Parse raw data from BLS450 with calibration."""

        test_data = scintillometry.wrangler.data_parser.parse_scintillometer(
            file_path=conftest_mnd_path, calibration=None
        )
        compare_data = scintillometry.wrangler.data_parser.parse_scintillometer(
            file_path=conftest_mnd_path, calibration=arg_calibration
        )

        assert isinstance(compare_data, pd.DataFrame)
        if arg_calibration:  # [<incorrect>, <correct>]
            test_calib = (arg_calibration[1] ** (-3)) / (arg_calibration[0] ** (-3))
        else:
            test_calib = 1

        data_keys = ["Cn2", "H_convection"]
        for key in data_keys:
            assert key in compare_data.columns
            assert ptypes.is_numeric_dtype(compare_data[key])
            # pd.equals() doesn't handle FPP
            assert np.allclose(compare_data[key], test_data[key] * test_calib)


class TestDataParsingTransect:
    """Test class for parsing path transects."""

    @pytest.mark.dependency(
        name="TestDataParsingTransect::test_parse_transect_file_not_found",
        scope="class",
    )
    def test_parse_transect_file_not_found(self):
        """Raise error if transect file not found."""

        wrong_path = "non_existent_file"
        error_message = f"No file found with path: {wrong_path}"
        with pytest.raises(  # incorrect path or missing file raises error
            FileNotFoundError, match=error_message
        ):
            scintillometry.wrangler.data_parser.parse_transect(file_path=wrong_path)

    @pytest.mark.dependency(
        name="TestDataParsingTransect::test_parse_transect",
        depends=["TestDataParsingTransect::test_parse_transect_file_not_found"],
        scope="class",
    )
    def test_parse_transect(self, conftest_transect_path):
        """Parse pre-processed transect file into dataframe."""

        test_dataframe = scintillometry.wrangler.data_parser.parse_transect(
            file_path=conftest_transect_path
        )

        assert isinstance(test_dataframe, pd.DataFrame)
        for key in test_dataframe.keys():
            assert key in ["path_height", "norm_position"]
            assert ptypes.is_numeric_dtype(test_dataframe[key])

        assert all(test_dataframe["norm_position"].between(0, 1, "both"))
