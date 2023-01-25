"""Copyright (C) 2022 Nicolas Gampierakis. All rights reserved.

Provides shared classes for tests.
"""

import io
import os

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


@pytest.fixture(scope="module", autouse=True)  # teardown after each module test
def conftest_mnd_path():
    """Yields path to a test mnd file."""

    test_file_path = "./tests/test_data/test_fake_bls_data.mnd"
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
