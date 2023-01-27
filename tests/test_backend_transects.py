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
import pytest

import scintillometry.backend.transects


class TestTransects:
    """Test class for path weighting functions."""

    @pytest.mark.dependency(
        name="TestTransects::test_bessel_second",
        scope="class",
    )
    @pytest.mark.parametrize("arg_x", [0.5, 0.0, 0.8, 1])
    def test_bessel_second(self, arg_x):
        """Calculate Bessel function for path position."""

        test_y = scintillometry.backend.transects.bessel_second(arg_x)
        if arg_x == 0.5:  # bessel variable is zero
            assert isinstance(test_y, int)
            assert test_y == 1
        else:
            assert isinstance(test_y, float)

    @pytest.mark.dependency(
        name="TestTransects::test_path_weighting",
        depends=["TestTransects::test_bessel_second"],
        scope="class",
    )
    def test_path_weighting(self):
        """Calculate path weights."""

        path_transect = pd.Series(np.linspace(0, 1, num=10), dtype=float)
        test_weights = scintillometry.backend.transects.path_weighting(
            path_coordinates=path_transect
        )
        assert isinstance(test_weights, list)
        assert all(isinstance(weight, float) for weight in test_weights)
        assert len(test_weights) == 10

    @pytest.mark.dependency(
        name="TestTransects::test_define_stability_not_implemented",
        scope="class",
    )
    def test_define_stability_not_implemented(self):
        """Raise error for invalid stability conditions."""

        test_stability = "raise_error"
        error_message = f"{test_stability} is not an implemented stability condition."
        with pytest.raises(NotImplementedError, match=error_message):
            scintillometry.backend.transects.define_stability(
                stability_name="raise_error"
            )

    @pytest.mark.dependency(
        name="TestTransects::test_define_stability",
        depends=["TestTransects::test_define_stability_not_implemented"],
        scope="class",
    )
    @pytest.mark.parametrize("arg_stability", ["stable", "unstable", None])
    def test_define_stability(self, arg_stability):
        """Matches b-constant to stability condition."""

        test_b = scintillometry.backend.transects.define_stability(
            stability_name=arg_stability
        )
        if arg_stability == "stable":
            assert test_b == pytest.approx(-2 / 3)
        elif arg_stability == "unstable":
            assert test_b == pytest.approx(-4 / 3)
        else:
            assert test_b == 1

    @pytest.mark.dependency(
        name="TestTransects::test_compute_effective_z",
        depends=[
            "TestTransects::test_path_weighting",
            "TestTransects::test_define_stability",
        ],
        scope="class",
    )
    @pytest.mark.parametrize("arg_stability", ["stable", "unstable", None])
    def test_compute_effective_z(self, arg_stability):
        """Compute effective path height."""

        test_heights = pd.Series(np.linspace(5, 10, num=10), dtype=float)
        test_positions = pd.Series(np.linspace(0, 1, num=10), dtype=float)

        test_z_eff = scintillometry.backend.transects.compute_effective_z(
            path_heights=test_heights,
            path_positions=test_positions,
            stability=arg_stability,
        )

        assert isinstance(test_z_eff, float)

    @pytest.mark.dependency(
        name="TestTransects::test_get_z_parameters",
        depends=[
            "TestTransects::test_compute_effective_z",
        ],
        scope="class",
    )
    def test_get_z_parameters(self):
        """Calculate effective and mean path heights."""

        test_data = {
            "path_height": np.linspace(5, 10, num=10),
            "norm_position": np.linspace(0, 1, num=10),
        }
        test_transect = pd.DataFrame(test_data)
        test_effective, test_mean = scintillometry.backend.transects.get_z_parameters(
            transect_data=test_transect, stability_condition="unstable"
        )

        for param in [test_effective, test_mean]:
            assert isinstance(param, float)

        assert test_effective != pytest.approx(test_mean)
        assert test_mean == pytest.approx(np.mean(test_transect["path_height"]))
