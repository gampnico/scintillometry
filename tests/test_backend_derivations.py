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

Tests path weighting module.
"""

import numpy as np
import pandas as pd
import pandas.api.types as ptypes
import pytest

import scintillometry.backend.derivations


class TestBackendDerivations:
    """Test class for path weighting functions."""

    test_data = {
        "Cn2": [1.9115e-16],
        "CT2": [4.513882047592982e-10],
        "temperature_2m": [10.7],
        "pressure": [950.2],
        "Q_0": [0.020937713873256197],
    }
    test_frame = pd.DataFrame.from_dict(test_data)
    # all dataframes are in the same TZ in main.
    test_index = pd.DatetimeIndex(data=["2020-06-03T00:10:00Z"], tz="CET")
    test_frame.index = test_index

    test_z_eff = np.float64(25.628)

    @pytest.mark.dependency(
        name="TestBackendDerivations::test_get_switch_time_error",
    )
    def test_get_switch_time_error(self):
        """Raise error if no data available to calculate switch time."""

        test_switch = self.test_frame[["temperature_2m", "pressure"]].copy()
        with pytest.raises(
            KeyError, match="No data to calculate switch time. Set manually."
        ):
            scintillometry.backend.derivations.get_switch_time(
                dataframe=test_switch, local_time=None
            )

    @pytest.mark.dependency(
        name="TestBackendDerivations::test_get_switch_time",
        depends=["TestBackendDerivations::test_get_switch_time_error"],
    )
    @pytest.mark.parametrize("arg_local_time", ["02:10", None])
    def test_get_switch_time(self, arg_local_time):
        """Get time where stability conditions change."""

        test_switch = self.test_frame[["temperature_2m", "pressure"]].copy()
        test_switch["global_irradiance"] = 23.0
        compare_switch = scintillometry.backend.derivations.get_switch_time(
            dataframe=test_switch, local_time=arg_local_time
        )
        assert isinstance(compare_switch, str)
        assert compare_switch == "02:10"

    @pytest.mark.dependency(
        name="TestBackendDerivations::test_derive_ct2",
        scope="class",
    )
    def test_derive_ct2(self):
        """Derive CT2 from Cn2, temperature, pressure."""

        test_ct2 = self.test_frame[["Cn2", "temperature_2m", "pressure"]].copy()
        compare_ct2 = scintillometry.backend.derivations.derive_ct2(dataframe=test_ct2)
        assert isinstance(compare_ct2, pd.DataFrame)
        assert "CT2" in compare_ct2.columns
        for key in test_ct2.columns:
            assert key in compare_ct2.columns

        assert all(
            ptypes.is_numeric_dtype(compare_ct2[key]) for key in compare_ct2.columns
        )
        assert not compare_ct2["CT2"].isnull().values.any()
        assert compare_ct2["CT2"].gt(compare_ct2["Cn2"]).values.any()
