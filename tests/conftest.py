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

Provides shared classes for tests.
"""

import io
import os

import pandas as pd
import pytest


class TestData:
    """Test class generating mock data. Child classes must be modified
    to pass using data generated from this class, not the other way
    around.

    Attributes:
        data (str): String mocking data in file. Can overwrite with
            `setattrs()` in child classes.
        charset (str): File encoding to emulate. Can overwrite with
            `setattrs()` in child classes.
    """

    def __init__(self):
        self.data = None
        self.charset = "utf-8"
        self.variables = None

    @classmethod
    def setup_mnd_file(cls):
        """Constructs mock output from reading a .mnd file."""

        file_header = ["FORMAT-1.1", "2020-06-03T03:23:00Z", "BLS450", "2 5\n"]
        ascii_header = ["Station Code:     Test", "Software:         SRun 1.49"]
        variables_lines = [
            "Main Data",
            "Time # time # # T3 # 1",
            "Constant of Refractive Index # Cn2 # m^(-2/3) # S # 1 # N/A",
            "Constant of Temperature Fluctuations # CT2 # K^2 m^(-2/3) # S # 1 # N/A",
            "Heat Flux (Free Convection) # H_convection # W/m^2 # S # 1 # N/A",
            "Pressure # pressure # hPa # S # 0 # N/A\n",
        ]
        mock_data = [
            "\t".join(
                [
                    "PT00H00M30S/2020-06-03T03:23:00Z",  # 03 June 2020 05:23 CET
                    "1.9115E-16",
                    "1.9343E-04",
                    "4.6",
                    "1010.0",
                ]
            ),
            "\t".join(
                [
                    "PT00H00M59S/2020-06-03T03:24:00Z",
                    "2.4472E-16",
                    "2.4764E-04",
                    "5.5",
                    "1010.0",
                ]
            ),
        ]
        full_file = []
        for data_part in [file_header, ascii_header, variables_lines, mock_data]:
            full_file.append("\n".join(data_part))

        data = io.StringIO("\n".join(full_file))
        assert isinstance(data, io.StringIO)
        cls.data = data
        cls.variables = ["time", "Cn2", "CT2", "H_convection", "pressure"]

        return cls

    @classmethod
    def setup_bls_dataframe(cls):
        """Constructs mock dataframe with BLS data."""

        column_names = ["time", "Cn2", "CT2", "H_convection", "pressure"]
        data = {
            "time": ["2020-06-03T03:23:00Z", "2020-06-03T03:24:00Z"],
            "Cn2": [1.9115e-16, 2.4472e-16],
            "CT2": [1.9343e-04, 2.4764e-04],
            "H_convection": [4.6, 5.5],
            "pressure": [1010.0, 1010.0],
        }
        dataframe = pd.DataFrame(data=data, columns=column_names)
        dataframe["time"] = pd.to_datetime(dataframe["time"])
        dataframe = dataframe.set_index("time")
        dataframe = dataframe.tz_convert("CET")

        cls.bls_dataframe = dataframe

        return cls

    @classmethod
    def setup_weather_dataframe(cls):
        """Constructs mock dataframe with weather data."""

        data = {
            "time": [
                "2020-06-03T00:10:00Z",
                "2020-06-03T00:20:00Z",
                "2020-06-03T00:30:00Z",
            ],
            "station": [0000, 0000, 0000],
            "wind_direction": [31.0, 235.0, 214.0],
            "vector_wind_speed": [1.2, 0.9, 1.0],
            "wind_speed": [1.2, 0.9, 1.1],
            "global_radiation": [0.0, 0.0, 0.0],
            "pressure": [950.5, 950.2, 950.4],
            "relative_humidity": [89.0, 70.0, 79.0],
            "precipitation": [0.0, 0.0, 0.0],
            "temperature_2m": [10.8, 10.7, 10.7],
        }
        dataframe = pd.DataFrame.from_dict(data)
        dataframe["time"] = pd.to_datetime(dataframe["time"])
        dataframe = dataframe.set_index("time")
        dataframe = dataframe.tz_convert("CET")
        assert dataframe.index.name == "time"
        cls.weather_dataframe = dataframe

        return cls


@pytest.fixture(scope="module", autouse=True)  # teardown after each module test
def conftest_mnd_path():
    """Yields path to a test mnd file."""

    test_file_path = "./tests/test_data/test_fake_bls_data.mnd"
    assert os.path.exists(test_file_path)

    yield test_file_path


@pytest.fixture(scope="module", autouse=True)  # teardown after each module test
def conftest_transect_path():
    """Yields path to a test path transect file."""

    test_file_path = "./tests/test_data/test_fake_path_transect.csv"
    assert os.path.exists(test_file_path)

    yield test_file_path


@pytest.fixture(scope="module", autouse=True)  # teardown after each module test
def conftest_create_test_data():
    """Creates TestData object."""

    yield TestData()


@pytest.fixture(scope="module", autouse=True)  # teardown after each module test
def conftest_create_test_mnd():
    """Creates list mocking .mnd file lines."""

    data_obj = TestData()

    yield data_obj.setup_mnd_file()


@pytest.fixture(scope="module", autouse=True)  # teardown after each module test
def conftest_create_test_bls():
    """Creates dataframe mocking parsed BLS data."""

    data_obj = TestData()

    yield data_obj.setup_bls_dataframe()


@pytest.fixture(scope="module", autouse=True)  # teardown after each module test
def conftest_create_test_weather():
    """Creates dataframe mocking parsed ZAMG data."""

    data_obj = TestData()

    yield data_obj.setup_weather_dataframe()
