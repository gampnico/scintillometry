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

import pandas as pd
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
        """Raise error if .mnd file is not FORMAT-1"""

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
        elif arg_date is False:
            assert compare_string == "PT00H00M30S"

    @pytest.mark.dependency(
        name="TestDataParsingBLS::test_parse_scintillometer",
        depends=[
            "TestFileHandling::test_file_handler_read",
            "TestDataParsingBLS::test_parse_iso_date",
            "TestDataParsingBLS::test_parse_mnd_lines",
        ],
        scope="module",
    )
    @pytest.mark.parametrize("arg_timezone", ["CET", "Europe/Berlin", None])
    def test_parse_scintillometer(self, conftest_mnd_path, arg_timezone):
        """Parse raw data from BLS450."""

        dataframe = scintillometry.wrangler.data_parser.parse_scintillometer(
            file_path=conftest_mnd_path, timezone=arg_timezone
        )

        assert isinstance(dataframe, pd.DataFrame)
        data_keys = ["Cn2", "CT2", "H_convection", "pressure", "iso_duration"]
        for key in data_keys:
            assert key in dataframe.columns
        assert dataframe.index.name == "time"
        assert "time" not in dataframe.columns
        if arg_timezone:
            assert dataframe.index.tz.zone == arg_timezone
        else:
            assert dataframe.index.tz.zone == "UTC"
