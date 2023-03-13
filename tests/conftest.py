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

Provides shared fixtures for tests.

Data used in test fixtures was randomly generated - they are not real
observational data.

Metadata::
    - Date: 03 June 2023
    - Scintillometer data period: 03:23:00Z to 03:24:00Z
    - Meteorological data period: 03:10:00Z to 03:30:00Z
    - Location Name: "Test"
    - Scintillometer model: BLS450
    - ZAMG Klima-ID: "0000"
"""

from unittest.mock import patch

import numpy as np
import pandas as pd
import pandas.api.types as ptypes
import pytest


# Function patches
@pytest.fixture(scope="function", autouse=False)
def conftest_mock_check_file_exists():
    """Override checks when mocking files."""

    patcher = patch("os.path.exists")
    mock_exists = patcher.start()
    mock_exists.return_value = True


@pytest.fixture(scope="function", autouse=False)
def conftest_mock_save_figure():
    """Stops figure being saved to disk."""

    patcher = patch("scintillometry.visuals.plotting.save_figure")
    mock_exists = patcher.start()
    mock_exists.return_value = None


# Mock scintillometer data
@pytest.fixture(name="conftest_mnd_lines", scope="function", autouse=False)
def fixture_conftest_mock_mnd_lines():
    """Constructs mock list output from reading .mnd string object."""

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

    yield full_file


@pytest.fixture(name="conftest_mock_mnd_raw", scope="function", autouse=False)
def fixture_conftest_mock_mnd_raw(conftest_mnd_lines):
    """Constructs mock string output from reading a .mnd file."""

    data = "\n".join(map(str, conftest_mnd_lines))
    assert isinstance(data, str)

    yield data


@pytest.fixture(name="conftest_mock_bls_dataframe", scope="function", autouse=False)
def fixture_conftest_mock_bls_dataframe():
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

    yield dataframe


@pytest.fixture(name="conftest_mock_bls_dataframe_tz", scope="function", autouse=False)
def fixture_conftest_mock_bls_dataframe_tz(conftest_mock_bls_dataframe):
    """Constructs mock tz-aware dataframe with BLS data."""

    dataframe = conftest_mock_bls_dataframe.copy(deep=True)
    dataframe = dataframe.set_index("time")
    dataframe = dataframe.tz_convert("CET")

    yield dataframe


# Mock topographical data
@pytest.fixture(scope="function", autouse=False)  # otherwise gets overwritten
def conftest_mock_transect_dataframe():
    """Constructs mock dataframe with transect data."""

    data = {
        "path_height": [0.12335, 65.902, 168.61, 82.378, 0.12329],
        "norm_position": [0.00037546, 0.24874, 0.50942, 0.74958, 1],
    }
    dataframe = pd.DataFrame.from_dict(data)

    yield dataframe


# Mock meteorological data
@pytest.fixture(name="conftest_mock_weather_raw", scope="function", autouse=False)
def fixture_conftest_mock_weather_raw():
    """Constructs mock dataframe with unparsed weather data."""

    data = {
        "time": [
            "2020-06-03T03:10:00Z",
            "2020-06-03T03:20:00Z",
            "2020-06-03T03:30:00Z",
        ],
        "station": ["0000", "0000", "0000"],
        "DD": [31.0, 235.0, 214.0],
        "FF": [1.2, 0.9, 1.0],
        "FFAM": [1.2, 0.9, 1.1],
        "GSX": [0.0, 0.0, 0.0],
        "P": [950.5, 950.2, 950.4],
        "RF": [89.0, 70.0, 79.0],
        "RR": [0.0, 0.0, 0.0],
        "TL": [10.8, 10.7, 10.7],
    }
    dataframe = pd.DataFrame.from_dict(data)

    yield dataframe


@pytest.fixture(name="conftest_mock_weather_dataframe", scope="function", autouse=False)
def fixture_conftest_mock_weather_dataframe():
    """Constructs mock dataframe with weather data."""

    data = {
        "time": [
            "2020-06-03T03:10:00Z",
            "2020-06-03T03:20:00Z",
            "2020-06-03T03:30:00Z",
        ],
        "station": ["0000", "0000", "0000"],
        "wind_direction": [31.0, 235.0, 214.0],
        "vector_wind_speed": [1.2, 0.9, 1.0],
        "wind_speed": [1.2, 0.9, 1.1],
        "global_irradiance": [0.1, 23, 56],
        "pressure": [950.5, 950.2, 950.4],
        "relative_humidity": [89.0, 70.0, 79.0],
        "precipitation": [0.0, 0.0, 0.0],
        "temperature_2m": [10.8, 10.7, 10.7],
    }
    dataframe = pd.DataFrame.from_dict(data)

    yield dataframe


@pytest.fixture(
    name="conftest_mock_weather_dataframe_tz", scope="function", autouse=False
)
def fixture_conftest_mock_weather_dataframe_tz(conftest_mock_weather_dataframe):
    """Constructs mock dataframe with weather data."""

    dataframe = conftest_mock_weather_dataframe.copy(deep=True)
    dataframe["time"] = pd.to_datetime(dataframe["time"])
    dataframe = dataframe.set_index("time")
    dataframe = dataframe.tz_convert("CET")
    assert dataframe.index.name == "time"

    yield dataframe


# Mock UIBK-specific data
@pytest.fixture(name="conftest_mock_innflux_dataframe", scope="function", autouse=False)
def fixture_conftest_mock_innflux_dataframe():
    """Constructs mock dataframe with raw InnFLUX data."""

    data = {
        "year": [2020, 2020],
        "month": [6, 6],
        "day": [3, 3],
        "hour": [3, 3],
        "minute": [0, 30],
        "second": [0, 0],
        "shf": [5.0291, 16.320],
        "wind_speed": [0.7721, 1.0485],
        "obukhov": [-18.23, -16.122],
    }

    dataframe = pd.DataFrame.from_dict(data)

    yield dataframe


@pytest.fixture(
    name="conftest_mock_innflux_dataframe_tz", scope="function", autouse=False
)
def fixture_conftest_mock_innflux_dataframe_tz(conftest_mock_innflux_dataframe):
    """Constructs mock dataframe with parsed InnFLUX data."""

    dataframe = conftest_mock_innflux_dataframe.copy(deep=True)

    t_cols = ["year", "month", "day", "hour", "minute", "second"]
    dataframe.index = pd.to_datetime(dataframe[t_cols])
    dataframe.index = dataframe.index + pd.DateOffset(hours=3)

    dataframe = dataframe.drop(t_cols, axis=1)
    dataframe = dataframe.replace(-999, np.nan)
    dataframe = dataframe.fillna(method="ffill")
    dataframe = dataframe.tz_localize("CET")

    yield dataframe


@pytest.fixture(
    name="conftest_mock_hatpro_scan_levels", scope="function", autouse=False
)
def fixture_conftest_mock_hatpro_scan_levels():
    """Constructs mock list of HATPRO measurement heights."""

    levels = [0, 10, 30, 50, 75, 100]

    yield levels


@pytest.fixture(
    name="conftest_mock_hatpro_humidity_data", scope="function", autouse=False
)
def fixture_conftest_mock_hatpro_humidity_data():
    """Constructs mock list output for HATPRO humidity data."""

    file_header = ["#Dataset: HATPRO UIBK Humidity - RAW"]
    column_header = ["rawdate;v01;v02;v03;v04;v05;v06"]
    mock_data = [
        "2020-06-03 03:10:00;7.45;7.37;7.28;7.23;7.16;7.10",
        "2020-06-03 03:20:00;7.44;7.36;7.26;7.21;7.14;7.08",
        "2020-06-03 03:30:00;7.47;7.39;7.30;7.25;7.19;7.12",
    ]
    data = []
    for data_part in [file_header, column_header, mock_data]:
        data.append("\n".join(data_part))
    assert isinstance(data, list)

    yield data


@pytest.fixture(
    name="conftest_mock_hatpro_humidity_raw", scope="function", autouse=False
)
def fixture_conftest_mock_hatpro_humidity_raw(conftest_mock_hatpro_humidity_data):
    """Constructs mock string output from raw HATPRO humidity file."""

    full_file = "\n".join(map(str, conftest_mock_hatpro_humidity_data))
    assert isinstance(full_file, str)

    yield full_file


@pytest.fixture(
    name="conftest_mock_hatpro_humidity_dataframe", scope="function", autouse=False
)
def fixture_conftest_mock_hatpro_humidity_dataframe():
    """Constructs mock dataframe from HATPRO humidity data."""

    data_index = pd.to_datetime(
        ["2020-06-03 03:10:00", "2020-06-03 03:20:00", "2020-06-03 03:30:00"], utc=False
    )
    heights = [0, 10, 30, 50, 75, 100]
    data = {
        0: [7.45, 7.44, 7.47],
        10: [7.37, 7.36, 7.39],
        30: [7.28, 7.26, 7.30],
        50: [7.23, 7.21, 7.25],
        75: [7.16, 7.14, 7.19],
        100: [7.10, 7.08, 7.12],
    }
    dataframe = pd.DataFrame(data=data, columns=heights, index=data_index).copy(
        deep=True
    )
    assert isinstance(dataframe, pd.DataFrame)
    dataframe.index.name = "rawdate"
    for key in heights:
        assert key in dataframe.columns
        assert ptypes.is_numeric_dtype(dataframe[key])
    assert ptypes.is_datetime64_any_dtype(dataframe.index)
    assert dataframe.index.name == "rawdate"

    yield dataframe


@pytest.fixture(
    name="conftest_mock_hatpro_humidity_dataframe_tz", scope="function", autouse=False
)
def fixture_conftest_mock_hatpro_humidity_dataframe_tz(
    conftest_mock_hatpro_humidity_dataframe,
):
    """Localises TZ-aware mock dataframe from HATPRO humidity data."""

    dataframe = conftest_mock_hatpro_humidity_dataframe
    dataframe = dataframe.tz_localize("UTC")
    dataframe = dataframe.tz_convert("CET")
    assert dataframe.index.tz.zone == "CET"

    yield dataframe


@pytest.fixture(
    name="conftest_mock_hatpro_temperature_data", scope="function", autouse=False
)
def fixture_conftest_mock_hatpro_temperature_data():
    """Constructs mock list output for HATPRO temperature data."""

    file_header = ["#Dataset: HATPRO UIBK Temperature - RAW"]
    column_header = ["rawdate;v01;v02;v03;v04;v05;v06"]
    mock_data = [
        "2020-06-03 03:10:00;283.66;283.60;283.40;283.20;282.97;282.73",
        "2020-06-03 03:20:00;283.44;283.41;283.29;283.14;282.94;282.71",
        "2020-06-03 03:30:00;283.36;283.32;283.18;283.03;282.83;282.63",
    ]
    data = []
    for data_part in [file_header, column_header, mock_data]:
        data.append("\n".join(data_part))
    assert isinstance(data, list)

    yield data


@pytest.fixture(
    name="conftest_mock_hatpro_temperature_raw", scope="function", autouse=False
)
def fixture_conftest_mock_hatpro_temperature_raw(conftest_mock_hatpro_temperature_data):
    """Constructs mock string output from raw HATPRO temperature file."""

    full_file = "\n".join(map(str, conftest_mock_hatpro_temperature_data))
    assert isinstance(full_file, str)

    yield full_file


@pytest.fixture(
    name="conftest_mock_hatpro_temperature_dataframe", scope="function", autouse=False
)
def fixture_conftest_mock_hatpro_temperature_dataframe():
    """Constructs mock dataframe from HATPRO temperature data."""

    data_index = pd.to_datetime(
        ["2020-06-03 03:10:00", "2020-06-03 03:20:00", "2020-06-03 03:30:00"], utc=False
    )
    heights = [0, 10, 30, 50, 75, 100]
    data = {
        0: [283.66, 283.44, 283.36],
        10: [283.60, 283.41, 283.32],
        30: [283.40, 283.29, 283.18],
        50: [283.20, 283.14, 283.03],
        75: [282.97, 282.94, 282.83],
        100: [282.73, 282.71, 282.63],
    }
    dataframe = pd.DataFrame(data=data, columns=heights, index=data_index).copy(
        deep=True
    )
    assert isinstance(dataframe, pd.DataFrame)
    dataframe.index.name = "rawdate"
    for key in heights:
        assert key in dataframe.columns
        assert ptypes.is_numeric_dtype(dataframe[key])
    assert ptypes.is_datetime64_any_dtype(dataframe.index)
    assert dataframe.index.name == "rawdate"

    yield dataframe


@pytest.fixture(
    name="conftest_mock_hatpro_temperature_dataframe_tz",
    scope="function",
    autouse=False,
)
def fixture_conftest_mock_hatpro_temperature_dataframe_tz(
    conftest_mock_hatpro_temperature_dataframe,
):
    """Localises TZ-aware mock dataframe from HATPRO humidity data."""

    dataframe = conftest_mock_hatpro_temperature_dataframe
    dataframe = dataframe.tz_localize("UTC")
    dataframe = dataframe.tz_convert("CET")
    assert dataframe.index.tz.zone == "CET"

    yield dataframe


# Mock processed data
@pytest.fixture(name="conftest_mock_merged_dataframe", scope="function", autouse=False)
def fixture_conftest_mock_merged_dataframe():
    """Constructs mock dataframe with BLS and weather data."""

    data = {
        "time": [
            "2020-06-03T03:10:00Z",
            "2020-06-03T03:20:00Z",
            "2020-06-03T03:30:00Z",
        ],
        "Cn2": [1.9115e-16, 2.4472e-16, 2.6163e-16],
        "CT2": [1.9343e-04, 2.4764e-04, 2.6475e-04],
        "H_convection": [4.6, 5.5, 5.5],
        "wind_speed": [1.2, 0.9, 1.1],
        "global_irradiance": [0.1, 23, 56],
        "pressure": [950.5, 950.2, 950.4],
        "rho_air": [1.166186, 1.166229, 1.166474],
        "temperature_2m": [10.8, 10.7, 10.7],
    }
    dataframe = pd.DataFrame.from_dict(data)
    dataframe["time"] = pd.to_datetime(dataframe["time"])
    dataframe = dataframe.set_index("time")
    dataframe = dataframe.tz_convert("CET")
    assert dataframe.index.name == "time"

    yield dataframe


@pytest.fixture(name="conftest_mock_derived_dataframe", scope="function", autouse=False)
def fixture_conftest_mock_derived_dataframe(conftest_mock_merged_dataframe):
    """Constructs mock dataframe with derived data."""

    dataframe = conftest_mock_merged_dataframe.copy(deep=True)
    dataframe["H_free"] = [28.519535, 118.844056, 155.033711]
    dataframe["global_irradiance"] = [0.1, 23, 56]

    yield dataframe


@pytest.fixture(scope="function", autouse=False)  # otherwise gets overwritten
def conftest_mock_iterated_dataframe():
    """Constructs mock dataframe with BLS and weather data."""

    data = {
        "time": [
            "2020-06-03T03:23:00Z",
            "2020-06-03T03:24:00Z",
            "2020-06-03T03:25:00Z",
        ],
        "CT2": [1.9343e-04, 2.4764e-04, 2.6475e-04],
        "wind_speed": [1.2, 0.9, 1.1],
        "rho_air": [1.166186, 1.166229, 1.166474],
        "temperature_2m": [10.8, 10.7, 10.7],
        "obukhov": [-56.236228, -20.290260, -12.160410],
        "shf": [33.852306, 128.717874, 164.173310],
        "u_star": [0.282501, 0.314166, 0.287277],
        "theta_star": [-0.102295, -0.363034, -0.508046],
        "H_free": [28.519535, 118.844056, 155.033711],
    }
    dataframe = pd.DataFrame.from_dict(data)
    dataframe["time"] = pd.to_datetime(dataframe["time"])
    dataframe = dataframe.set_index("time")
    dataframe = dataframe.tz_convert("CET")
    assert dataframe.index.name == "time"

    yield dataframe
