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

Tests iterative scheme.
"""

import mpmath
import pytest

import scintillometry.backend.constants
import scintillometry.backend.iterations


class TestBackendIterationMost:
    """Test class for MOST iteration.

    Attributes:
        test_class (scintillometry.backend.iterations.IterationMost(): An
            initialised IterationMost object.
        test_coeffs (list[tuple, tuple]): MOST coefficients from Andreas
        (1988), formatted as [(unstable, unstable), (stable, stable)].
    """

    test_class = scintillometry.backend.iterations.IterationMost()
    test_coeffs = [(4.9, 6.1), (4.9, 2.2)]  # Andreas (1988)

    @pytest.mark.dependency(
        name="TestBackendIterationMost::test_most_iteration_init",
        depends=["TestBackendConstants::test_init"],
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
        name="TestBackendIterationMost::test_momentum_stability_unstable",
        scope="class",
    )
    def test_momentum_stability_unstable(self):
        """Integrated stability function, momentum, unstable conditions."""

        test_momentum = self.test_class.momentum_stability_unstable(obukhov=-100, z=10)
        assert isinstance(test_momentum, mpmath.mpf)
        assert test_momentum > 0
        assert test_momentum == pytest.approx(0.2836137)

    @pytest.mark.dependency(
        name="TestBackendIterationMost::test_momentum_stability_stable",
        scope="class",
    )
    def test_momentum_stability_stable(self):
        """Integrated stability function, momentum, stable conditions."""

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
        """Integrated stability function for momentum."""

        test_momentum = self.test_class.momentum_stability(obukhov=arg_obukhov, z=10)

        if arg_obukhov < 0:
            assert isinstance(test_momentum, mpmath.mpf)
            assert test_momentum > 0
            assert test_momentum == pytest.approx(0.2836137)
        else:  # obukhov lengths of zero are adjusted in pre-processing.
            assert isinstance(test_momentum, mpmath.mpf)
            assert test_momentum < 0

    @pytest.mark.dependency(
        name="TestBackendIterationMost::test_get_most_coefficients_error",
        scope="class",
    )
    def test_get_most_coefficients_error(self):
        """Raises errors for non-implemented MOST coefficients."""

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
        """Fetches MOST coefficients from AtmosConstants class."""

        compare_coeffs = self.test_class.get_most_coefficients(
            most_id="an1988", most_type="ct2"
        )
        assert isinstance(compare_coeffs, list)
        assert all(isinstance(coeffs, tuple) for coeffs in compare_coeffs)
        for coeffs in compare_coeffs:
            assert all(isinstance(coeff, float) for coeff in coeffs)
        assert compare_coeffs == self.test_coeffs

    @pytest.mark.dependency(
        name="TestBackendIterationMost::test_similarity_function",
        scope="class",
    )
    @pytest.mark.parametrize("arg_obukhov", [(-100, False), (0, True), (100, True)])
    def test_similarity_function(self, arg_obukhov):
        """Computes similarity function."""

        test_f_ct2 = self.test_class.similarity_function(
            obukhov=arg_obukhov[0], z=10, coeffs=self.test_coeffs, stable=arg_obukhov[1]
        )

        assert isinstance(test_f_ct2, float)
        assert test_f_ct2 > 0
