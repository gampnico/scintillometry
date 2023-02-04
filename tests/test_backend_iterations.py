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
    """Test class for MOST iteration."""

    test_class = scintillometry.backend.iterations.IterationMost()

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

    def test_momentum_stability_unstable(self):
        """Integrated stability function, momentum, unstable conditions."""

        test_momentum = self.test_class.momentum_stability_unstable(obukhov=-100, z=10)
        assert isinstance(test_momentum, mpmath.mpf)
        assert test_momentum > 0
        assert test_momentum == pytest.approx(0.2836137)

    def test_momentum_stability_stable(self):
        """Integrated stability function, momentum, stable conditions."""

        test_momentum = self.test_class.momentum_stability_stable(obukhov=200, z=10)
        assert isinstance(test_momentum, mpmath.mpf)
        assert test_momentum < 0
        assert test_momentum == pytest.approx(-0.25)

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
