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

Tests path weighting and topographical functions.
"""

import numpy as np
import pandas as pd
import pytest

import scintillometry.backend.transects


class TestBackendTransectTransform:
    """Test class for mathematical transformations."""

    test_transect_transform = scintillometry.backend.transects.TransectTransform()

    @pytest.mark.dependency(name="TestBackendTransectTransform::test_transform_init")
    def test_transform_init(self):
        """Inherit methods from parent."""

        test_class = scintillometry.backend.transects.TransectTransform()
        assert test_class

    @pytest.mark.dependency(name="TestBackendTransectTransform::test_bessel_second")
    @pytest.mark.parametrize("arg_x", [0.5, 0.0, 0.8, 1])
    def test_bessel_second(self, arg_x):
        """Calculate Bessel function for path position."""

        test_y = self.test_transect_transform.bessel_second(arg_x)
        if arg_x == 0.5:  # bessel variable is zero
            assert isinstance(test_y, int)
            assert test_y == 1
        else:
            assert isinstance(test_y, float)

    @pytest.mark.dependency(
        name="TestBackendTransectTransform::test_path_weighting",
        depends=["TestBackendTransectTransform::test_bessel_second"],
        scope="class",
    )
    def test_path_weighting(self):
        """Calculate path weights."""

        path_transect = pd.Series(np.linspace(0, 1, num=10), dtype=float)
        test_weights = self.test_transect_transform.path_weighting(
            path_coordinates=path_transect
        )
        assert isinstance(test_weights, list)
        assert all(isinstance(weight, float) for weight in test_weights)
        assert len(test_weights) == 10


class TestBackendTransectParameters:
    """Tests path height computations."""

    test_transect_parameters = scintillometry.backend.transects.TransectParameters()

    @pytest.mark.dependency(name="TestBackendTransectParameters::test_parameters_init")
    def test_parameters_init(self):
        """Inherit methods from parent."""

        test_class = scintillometry.backend.transects.TransectParameters()
        assert test_class
        assert test_class.transform
        assert isinstance(
            test_class.transform, scintillometry.backend.transects.TransectTransform
        )

    @pytest.mark.dependency(
        name="TestBackendTransectParameters::test_get_b_value_not_implemented"
    )
    def test_get_b_value_not_implemented(self):
        """Raise error for invalid stability conditions."""

        test_stability = "raise_error"
        error_message = f"{test_stability} is not an implemented stability condition."
        with pytest.raises(NotImplementedError, match=error_message):
            self.test_transect_parameters.get_b_value(stability_name="raise_error")

    @pytest.mark.dependency(
        name="TestBackendTransectParameters::test_get_b_value",
        depends=["TestBackendTransectParameters::test_get_b_value_not_implemented"],
    )
    @pytest.mark.parametrize("arg_stability", ["stable", "unstable", None])
    def test_get_b_value(self, arg_stability):
        """Matches b-constant to stability condition."""

        test_b = self.test_transect_parameters.get_b_value(stability_name=arg_stability)
        if arg_stability == "stable":
            assert test_b == pytest.approx(-2 / 3)
        elif arg_stability == "unstable":
            assert test_b == pytest.approx(-4 / 3)
        else:
            assert test_b == 1

    @pytest.mark.dependency(
        name="TestBackendTransectParameters::test_get_effective_path_height",
        depends=[
            "TestBackendTransectTransform::test_path_weighting",
            "TestBackendTransectParameters::test_parameters_init",
            "TestBackendTransectParameters::test_get_b_value",
        ],
        scope="module",
    )
    @pytest.mark.parametrize("arg_stability", ["stable", "unstable", None])
    def test_compute_effective_z(self, arg_stability):
        """Compute effective path height."""

        test_heights = pd.Series(np.linspace(5, 10, num=10), dtype=float)
        test_positions = pd.Series(np.linspace(0, 1, num=10), dtype=float)

        test_z_eff = self.test_transect_parameters.get_effective_path_height(
            path_heights=test_heights,
            path_positions=test_positions,
            stability=arg_stability,
        )

        assert isinstance(test_z_eff, float)

    @pytest.mark.dependency(
        name="TestBackendTransectParameters::test_get_path_heights",
        depends=["TestBackendTransectParameters::test_get_effective_path_height"],
    )
    def test_get_path_heights(self):
        """Calculate effective and mean path heights."""

        test_data = {
            "path_height": np.linspace(5, 10, num=10),
            "norm_position": np.linspace(0, 1, num=10),
        }
        test_transect = pd.DataFrame(test_data)
        test_effective, test_mean = self.test_transect_parameters.get_path_heights(
            transect_data=test_transect, stability_condition="unstable"
        )

        for param in [test_effective, test_mean]:
            assert isinstance(param, np.floating)

        assert test_effective != pytest.approx(test_mean)
        assert test_mean == pytest.approx(np.mean(test_transect["path_height"]))

    @pytest.mark.dependency(
        name="TestBackendTransectParameters::test_get_all_path_heights",
        depends=["TestBackendTransectParameters::test_get_path_heights"],
    )
    def test_get_all_path_heights(self):
        """Calculate effective & mean path heights in all conditions."""

        test_data = {
            "path_height": np.linspace(5, 10, num=10),
            "norm_position": np.linspace(0, 1, num=10),
        }
        test_transect = pd.DataFrame(test_data)
        test_keys = ["stable", "unstable", "None"]

        compare_heights = self.test_transect_parameters.get_all_path_heights(
            path_transect=test_transect
        )

        assert None not in compare_heights
        assert all(key in compare_heights for key in test_keys)
        for key, value in compare_heights.items():
            assert key in ["stable", "unstable", "None"]
            assert isinstance(value, tuple)
            assert isinstance(value[0], np.floating)
            assert isinstance(value[1], np.floating)
            assert value[1] > value[0]  # for this test case

    @pytest.mark.dependency(
        name="TestBackendTransectParameters::test_print_path_heights"
    )
    @pytest.mark.parametrize("arg_stability", ["stable", None])
    def test_print_path_heights(self, capsys, arg_stability):
        """Print effective and mean path height."""

        test_eff = 34
        test_mean = 31.245
        if arg_stability:
            test_suffix = f"{arg_stability} conditions"
        else:
            test_suffix = "no height dependency"

        test_print = (
            f"Selected {test_suffix}:\n",
            f"Effective path height:\t{test_eff:>0.2f} m.\n",
            f"Mean path height:\t{test_mean:>0.2f} m.\n",
        )
        print("".join(test_print))
        test_capture = capsys.readouterr()

        self.test_transect_parameters.print_path_heights(
            z_eff=np.float64(34), z_mean=np.float64(31.245), stability=arg_stability
        )
        compare_capture = capsys.readouterr()

        assert compare_capture == test_capture
