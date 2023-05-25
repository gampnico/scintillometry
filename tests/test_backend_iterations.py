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

Tests iterative scheme.
"""

import warnings

import mpmath
import numpy as np
import pandas as pd
import pytest

import scintillometry.backend.constants
import scintillometry.backend.iterations


class TestBackendIterationMost:
    """Test class for MOST iteration.

    Attributes:
        test_class (IterationMost): An initialised IterationMost object.
        test_coeffs (list[tuple, tuple]): MOST coefficients from Andreas
            (1988), formatted as
            [(unstable, unstable), (stable, stable)].
    """

    test_class = scintillometry.backend.iterations.IterationMost()
    test_coeffs = [(4.9, 6.1), (4.9, 2.2)]  # Andreas (1988)

    @pytest.mark.dependency(
        name="TestBackendIterationMost::test_most_iteration_init",
        depends=["TestBackendConstants::test_constants_init"],
        scope="session",
    )
    def test_most_iteration_init(self):
        test_most_iteration = scintillometry.backend.iterations.IterationMost()
        assert test_most_iteration.constants
        assert isinstance(
            test_most_iteration.constants,
            scintillometry.backend.constants.AtmosConstants,
        )

    @pytest.mark.dependency(
        name="TestBackendIterationMost::test_momentum_stability_unstable"
    )
    def test_momentum_stability_unstable(self):
        """Compute ISF for momentum, unstable conditions."""

        test_momentum = self.test_class.momentum_stability_unstable(obukhov=-100, z=10)
        assert isinstance(test_momentum, mpmath.mpf)
        assert test_momentum > 0
        assert test_momentum == pytest.approx(0.2836137)

    @pytest.mark.dependency(
        name="TestBackendIterationMost::test_momentum_stability_stable"
    )
    def test_momentum_stability_stable(self):
        """Compute ISF for momentum, stable conditions."""

        test_momentum = self.test_class.momentum_stability_stable(obukhov=200, z=10)
        assert isinstance(test_momentum, mpmath.mpf)
        assert test_momentum < 0
        assert test_momentum == pytest.approx(-0.25)

    @pytest.mark.dependency(
        name="TestBackendIterationMost::test_momentum_stability",
        depends=[
            "TestBackendIterationMost::test_momentum_stability_unstable",
            "TestBackendIterationMost::test_momentum_stability_stable",
        ],
        scope="class",
    )
    @pytest.mark.parametrize("arg_obukhov", [-100, 1e-10, 200])
    def test_momentum_stability(self, arg_obukhov):
        """Compute ISF for momentum."""

        test_momentum = self.test_class.momentum_stability(obukhov=arg_obukhov, z=10)

        if arg_obukhov < 0:
            assert isinstance(test_momentum, mpmath.mpf)
            assert test_momentum > 0
            assert test_momentum == pytest.approx(0.2836137)
        else:  # obukhov lengths of zero are adjusted in pre-processing.
            assert isinstance(test_momentum, mpmath.mpf)
            assert test_momentum < 0

    @pytest.mark.dependency(
        name="TestBackendIterationMost::test_get_most_coefficients_error"
    )
    def test_get_most_coefficients_error(self):
        """Raise errors for non-implemented MOST coefficients."""

        with pytest.raises(
            NotImplementedError,
            match="MOST coefficients are not implemented for Test ID.",
        ):
            self.test_class.get_most_coefficients(most_id="Test ID", most_type="ct2")
        with pytest.raises(
            NotImplementedError,
            match="MOST coefficients are not implemented for functions of Test Param.",
        ):
            self.test_class.get_most_coefficients(
                most_id="an1988", most_type="Test Param"
            )

    @pytest.mark.dependency(
        name="TestBackendIterationMost::test_get_most_coefficients",
        depends=["TestBackendIterationMost::test_get_most_coefficients_error"],
        scope="class",
    )
    def test_get_most_coefficients(self):
        """Fetch MOST coefficients from AtmosConstants class."""

        compare_coeffs = self.test_class.get_most_coefficients(
            most_id="an1988", most_type="ct2"
        )
        assert isinstance(compare_coeffs, list)
        assert all(isinstance(coeffs, tuple) for coeffs in compare_coeffs)
        for coeffs in compare_coeffs:
            assert all(isinstance(coeff, float) for coeff in coeffs)
        assert compare_coeffs == self.test_coeffs

    @pytest.mark.dependency(name="TestBackendIterationMost::test_similarity_function")
    @pytest.mark.parametrize("arg_obukhov", [(-100, False), (0, True), (100, True)])
    def test_similarity_function(self, arg_obukhov: tuple):
        """Compute similarity function."""

        test_f_ct2 = self.test_class.similarity_function(
            obukhov=arg_obukhov[0], z=10, coeffs=self.test_coeffs, stable=arg_obukhov[1]
        )

        assert isinstance(test_f_ct2, float)
        assert test_f_ct2 > 0

    @pytest.mark.dependency(name="TestBackendIterationMost::test_calc_theta_star")
    @pytest.mark.parametrize("arg_params", [(1.9e-04, 5.6, True), (2e-03, 3.6, False)])
    def test_calc_theta_star(self, arg_params: tuple):
        """Calculate temperature scale."""

        test_theta = self.test_class.calc_theta_star(
            ct2=arg_params[0], f_ct2=arg_params[1], z=10, stable=arg_params[2]
        )

        assert isinstance(test_theta, mpmath.mpf)
        if not arg_params[2]:
            assert test_theta < 0
        else:
            assert test_theta > 0

    @pytest.mark.dependency(
        name="TestBackendIterationMost::test_calc_u_star",
        depends=["TestBackendIterationMost::test_momentum_stability"],
        scope="class",
    )
    @pytest.mark.parametrize("arg_obukhov", [-100, 100])
    def test_calc_u_star(self, arg_obukhov):
        """Calculate friction velocity."""

        test_velocity = self.test_class.calc_u_star(
            u=1.0, z_u=10, r_length=1, o_length=arg_obukhov
        )
        assert isinstance(test_velocity, mpmath.mpf)
        assert test_velocity > 0

    @pytest.mark.dependency(name="TestBackendIterationMost::test_calc_obukhov_length")
    @pytest.mark.parametrize("arg_theta", [0.05, -0.05])
    def test_calc_obukhov_length(self, arg_theta):
        """Calculate Obukhov length."""

        compare_lob = self.test_class.calc_obukhov_length(
            temp=np.float64(295.0), u_star=0.2, theta_star=mpmath.mpmathify(arg_theta)
        )
        assert isinstance(compare_lob, mpmath.mpf)
        assert (compare_lob < 0) == (arg_theta < 0)  # obukhov and theta have same sign

    @pytest.mark.dependency(name="TestBackendIterationMost::test_check_signs")
    @pytest.mark.filterwarnings("error")
    @pytest.mark.parametrize("arg_shf", [-150, 0, 150])
    @pytest.mark.parametrize("arg_obukhov", [-100, 0, 100])
    def test_check_signs(self, arg_shf, arg_obukhov):
        """Warn if sign of variable doesn't match expected sign."""

        test_data = [{"shf": arg_shf, "obukhov": arg_obukhov}]
        test_frame = pd.DataFrame.from_records(test_data)

        if arg_shf > 0:
            with pytest.raises(UserWarning):
                self.test_class.check_signs(stable_flag=True, dataframe=test_frame)
        if arg_obukhov < 0:
            with pytest.raises(UserWarning):
                self.test_class.check_signs(stable_flag=True, dataframe=test_frame)

        if arg_shf == 0 or arg_obukhov == 0:  # never compare zeros.
            with warnings.catch_warnings():
                warnings.simplefilter("error")
        elif (arg_shf > 0) == (arg_obukhov > 0):
            with pytest.raises(UserWarning):
                self.test_class.check_signs(stable_flag=False, dataframe=test_frame)

    @pytest.mark.dependency(
        name="TestBackendIterationMost::test_most_iteration",
        depends=[
            "TestBackendIterationMost::test_most_iteration_init",
            "TestBackendIterationMost::test_similarity_function",
            "TestBackendIterationMost::test_calc_u_star",
            "TestBackendIterationMost::test_calc_theta_star",
            "TestBackendIterationMost::test_calc_obukhov_length",
            "TestBackendIterationMost::test_check_signs",
        ],
        scope="class",
    )
    @pytest.mark.parametrize("arg_stable", [(200, True), (-100, False)])
    def test_most_iteration(self, conftest_mock_merged_dataframe, arg_stable: tuple):
        """Iterate single row of dataframe using MOST."""

        test_data = conftest_mock_merged_dataframe.iloc[0].copy(deep=True)
        test_data["obukhov"] = arg_stable[0]  # initial Obukhov Length

        compare_most = self.test_class.most_iteration(
            dataframe=test_data,
            zm_bls=30,
            stable_flag=arg_stable[1],
            most_coeffs=self.test_coeffs,
        )

        compare_keys = ["u_star", "theta_star", "f_ct2", "shf"]
        for key in compare_keys:
            assert key in compare_most.keys()
            assert isinstance(compare_most[key], mpmath.mpf)

        assert compare_most["obukhov"] != arg_stable[0]
        assert not (compare_most.isnull()).any()

        # signs match stability
        assert (compare_most["obukhov"] > 0) == (arg_stable[0] > 0)
        assert (compare_most["shf"] < 0) == (arg_stable[0] > 0)
        assert (compare_most["obukhov"] > 0) == (compare_most["shf"] < 0)

    @pytest.mark.dependency(
        name="TestBackendIterationMost::test_most_iteration_nan",
        depends=["TestBackendIterationMost::test_most_iteration"],
        scope="class",
    )
    @pytest.mark.parametrize("arg_stable", [(200, True), (-100, False)])
    def test_most_iteration_nan(self, conftest_mock_merged_dataframe, arg_stable):
        """Iterate single row of dataframe using MOST with NaNs."""

        test_data = conftest_mock_merged_dataframe.iloc[0].copy(deep=True)
        test_data["CT2"] = np.nan
        test_data["obukhov"] = arg_stable[0]  # initial Obukhov Length

        compare_most = self.test_class.most_iteration(
            dataframe=test_data,
            zm_bls=30,
            stable_flag=arg_stable[1],
            most_coeffs=self.test_coeffs,
        )

        compare_keys = ["u_star", "theta_star", "f_ct2", "shf"]
        for key in compare_keys:
            assert key in compare_most.keys()
            assert isinstance(compare_most[key], (mpmath.mpf, float))

        assert compare_most["obukhov"] != arg_stable[0]
        assert mpmath.isnan(compare_most["obukhov"])
        assert mpmath.isnan(compare_most["shf"])

    @pytest.mark.dependency(
        name="TestBackendIterationMost::test_most_method",
        depends=[
            "TestBackendIterationMost::test_most_iteration",
            "TestBackendIterationMost::test_check_signs",
        ],
        scope="class",
    )
    @pytest.mark.parametrize("arg_stable", [["stable", True], ["unstable", False]])
    def test_most_method(self, capsys, conftest_mock_merged_dataframe, arg_stable):
        """Calculate Obukhov length and sensible heat fluxes."""

        test_data = conftest_mock_merged_dataframe

        compare_most = self.test_class.most_method(
            dataframe=test_data, eff_h=30, stability=arg_stable[0], coeff_id="an1988"
        )
        test_print = capsys.readouterr()
        assert f"Started iteration ({arg_stable[0]})..." in test_print.out
        assert f"Completed iteration ({arg_stable[0]}) in" in test_print.out
        compare_keys = [
            "u_star",
            "theta_star",
            "f_ct2",
            "shf",
            "obukhov",
        ]
        assert isinstance(compare_most, pd.DataFrame)
        for key in compare_keys:
            assert not (compare_most[key].isnull()).any()
            assert key in compare_most.keys()
            assert all(isinstance(x, mpmath.mpf) for x in compare_most[key])

        # signs match stability
        assert (compare_most["obukhov"] > 0).all() == arg_stable[1]
        assert (compare_most["shf"] < 0).all() == arg_stable[1]
        assert (compare_most["obukhov"] < 0).all() == (compare_most["shf"] > 0).all()
