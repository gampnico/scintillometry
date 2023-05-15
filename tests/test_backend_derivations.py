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

Tests mathematical derivations of input data.
"""

import numpy as np
import pandas as pd
import pandas.api.types as ptypes
import pytest

import scintillometry.backend.constants
import scintillometry.backend.derivations


class TestBackendDerivations:
    """Test class for derivations of scintillometer data.


    Attributes:
        test_data (dict): Placeholder data for single timestamp.
        test_frame (pd.DataFrame): TZ-aware labelled data for single
            timestamp.
        test_index (pd.DatetimeIndex): Single placeholder timestamp for
            data collection.
        test_z_eff (np.float64): Placeholder for effective path height.
    """

    test_derive_scintillometer = (
        scintillometry.backend.derivations.DeriveScintillometer()
    )
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

    @pytest.mark.dependency(name="TestDeriveScintillometer::test_scintillometer_init")
    def test_scintillometer_init(self):
        """Inherit methods from parent."""

        test_class = scintillometry.backend.derivations.DeriveScintillometer()
        assert test_class
        assert test_class.constants
        assert isinstance(
            test_class.constants, scintillometry.backend.constants.AtmosConstants
        )

    @pytest.mark.dependency(name="TestBackendDerivations::test_derive_ct2")
    def test_derive_ct2(self):
        """Derive CT2 from Cn2, temperature, pressure."""

        test_ct2 = self.test_frame[["Cn2", "temperature_2m", "pressure"]].copy()
        compare_ct2 = self.test_derive_scintillometer.derive_ct2(dataframe=test_ct2)
        assert isinstance(compare_ct2, pd.DataFrame)
        assert "CT2" in compare_ct2.columns
        for key in test_ct2.columns:
            assert key in compare_ct2.columns

        assert all(
            ptypes.is_numeric_dtype(compare_ct2[key]) for key in compare_ct2.columns
        )
        assert not compare_ct2["CT2"].isnull().values.any()
        assert compare_ct2["CT2"].gt(compare_ct2["Cn2"]).values.any()

    @pytest.mark.dependency(name="TestBackendDerivations::test_kinematic_shf")
    def test_kinematic_shf(self):
        """Compute kinematic SHF."""

        test_kshf = self.test_frame[["CT2", "temperature_2m"]].copy()
        compare_kshf = self.test_derive_scintillometer.kinematic_shf(
            dataframe=test_kshf, z_eff=self.test_z_eff
        )
        assert isinstance(compare_kshf, pd.DataFrame)
        assert "Q_0" in compare_kshf.columns
        for key in test_kshf.columns:
            assert key in compare_kshf.columns

        assert all(
            ptypes.is_numeric_dtype(compare_kshf[key]) for key in compare_kshf.columns
        )
        assert not compare_kshf["Q_0"].isnull().values.any()

    @pytest.mark.dependency(name="TestBackendDerivations::test_free_convection_shf")
    def test_free_convection_shf(self):
        """Compute surface SHF with free convection."""

        test_sshf = self.test_frame[["CT2", "temperature_2m", "pressure", "Q_0"]].copy()
        compare_sshf = self.test_derive_scintillometer.free_convection_shf(
            dataframe=test_sshf
        )
        compare_keys = ["rho_air", "H_free"]
        assert isinstance(compare_sshf, pd.DataFrame)
        for key in compare_keys:
            assert key in compare_sshf.columns
        for key in test_sshf.columns:
            assert key in compare_sshf.columns

        assert all(
            ptypes.is_numeric_dtype(compare_sshf[key]) for key in compare_sshf.columns
        )
        assert not compare_sshf[["rho_air", "H_free"]].isnull().values.any()

    @pytest.mark.dependency(
        name="TestBackendDerivations::test_compute_fluxes",
        depends=[
            "TestBackendDerivations::test_derive_ct2",
            "TestBackendDerivations::test_kinematic_shf",
            "TestBackendDerivations::test_free_convection_shf",
        ],
        scope="class",
    )
    @pytest.mark.parametrize("arg_beam_params", [None, (850, 20)])
    def test_compute_fluxes(self, arg_beam_params):
        """Compute kinematic and surface sensible heat fluxes."""

        test_fluxes = self.test_frame[["Cn2", "temperature_2m", "pressure"]].copy()
        compare_fluxes = self.test_derive_scintillometer.compute_fluxes(
            input_data=test_fluxes,
            effective_height=self.test_z_eff,
            beam_params=arg_beam_params,
        )

        test_keys = ["CT2", "rho_air", "H_free", "Q_0"]
        compare_keys = compare_fluxes.columns
        assert isinstance(compare_fluxes, pd.DataFrame)

        for key in test_keys:
            assert key in compare_keys
        for key in test_fluxes.columns:
            assert key in compare_keys

        assert all(ptypes.is_numeric_dtype(compare_fluxes[key]) for key in compare_keys)
        assert not compare_fluxes[test_keys].isnull().values.any()
