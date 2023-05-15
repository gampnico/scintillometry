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

Tests constants.
"""

import numpy as np
import pandas as pd
import pytest
from numpy.random import default_rng

import scintillometry.backend.constants


class TestBackendConstants:
    """Test class for AtmosConstants class.

    Attributes:
        test_rng (np.random.Generator): Random number generator.
    """

    test_rng = default_rng()

    @pytest.mark.dependency(name="TestBackendConstants::test_constants_init")
    def test_constants_init(self):
        test_class = scintillometry.backend.constants.AtmosConstants()
        attributes_dict = test_class.__dict__
        for constant in attributes_dict.values():
            assert isinstance(constant, (float, int, dict))
            if isinstance(constant, dict):
                assert all(
                    isinstance(value, (float, list)) for value in constant.values()
                )

    @pytest.mark.dependency(
        name="TestBackendConstants::test_get_error",
        depends=["TestBackendConstants::test_constants_init"],
        scope="class",
    )
    def test_get_error(self):
        """Raise AttributeError for missing attribute."""

        test_name = "raise_error"
        error_msg = f"'AtmosConstants' object has no attribute '{test_name}'"

        with pytest.raises(AttributeError, match=error_msg):
            test_class = scintillometry.backend.constants.AtmosConstants()
            test_class.get(test_name)

    @pytest.mark.dependency(
        name="TestBackendConstants::test_get",
        depends=[
            "TestBackendConstants::test_constants_init",
            "TestBackendConstants::test_get_error",
        ],
        scope="class",
    )
    def test_get(self):
        """Get attribute value."""

        test_attribute = ("k", 0.4)  # von Kármán's constant, k=0.4
        test_class = scintillometry.backend.constants.AtmosConstants()
        compare_attribute = test_class.get(test_attribute[0])

        assert isinstance(compare_attribute, float)
        assert compare_attribute == pytest.approx(test_attribute[1])

    @pytest.mark.dependency(
        name="TestBackendConstants::test_overwrite",
        depends=["TestBackendConstants::test_constants_init"],
        scope="class",
    )
    @pytest.mark.parametrize("arg_value", [-0.4, 0.5, 1])
    def test_overwrite(self, arg_value):
        """Overwrite attribute value."""

        test_class = scintillometry.backend.constants.AtmosConstants()
        test_class.overwrite("k", arg_value)

        assert isinstance(test_class.k, (float, int))
        assert test_class.k == pytest.approx(arg_value)

    @pytest.mark.dependency(
        name="TestBackendConstants::test_convert_pressure",
        depends=["TestBackendConstants::test_constants_init"],
        scope="class",
    )
    @pytest.mark.parametrize("arg_type", ["series", "frame"])
    @pytest.mark.parametrize("arg_units", [1, 1000, 100000])
    @pytest.mark.parametrize("arg_base", [True, False])
    def test_convert_pressure(self, arg_type, arg_units, arg_base):
        """Convert pressure to Pa or hPa."""

        test_class = scintillometry.backend.constants.AtmosConstants()
        test_ref = pd.Series(
            self.test_rng.uniform(low=0.8, high=1.2, size=100), name="pressure"
        )
        assert isinstance(test_ref, pd.Series)

        if arg_type == "series":
            test_pressure = test_ref.multiply(arg_units)
            compare_pressure = test_class.convert_pressure(
                pressure=test_pressure, base=arg_base
            )
            assert isinstance(compare_pressure, pd.Series)
        else:
            test_data = {0: test_ref, 10: test_ref}
            test_ref = pd.DataFrame(data=test_data, columns=[0, 10])
            assert isinstance(test_ref, pd.DataFrame)

            test_pressure = test_ref.multiply(arg_units)
            compare_pressure = test_class.convert_pressure(
                pressure=test_pressure, base=arg_base
            )
            assert isinstance(compare_pressure, pd.DataFrame)

        if not arg_base:
            assert np.allclose(compare_pressure, test_ref * 1000)
        else:
            assert np.allclose(compare_pressure, test_ref * 100000)

    @pytest.mark.dependency(
        name="TestBackendConstants::test_convert_temperature",
        depends=["TestBackendConstants::test_constants_init"],
        scope="class",
    )
    @pytest.mark.parametrize("arg_type", ["series", "frame"])
    @pytest.mark.parametrize("arg_units", [0, 1])
    @pytest.mark.parametrize("arg_base", [True, False])
    def test_convert_temperature(self, arg_type, arg_units, arg_base):
        """Convert temperature to kelvins or Celsius."""

        test_class = scintillometry.backend.constants.AtmosConstants()
        test_ref = pd.Series(
            self.test_rng.uniform(low=200.0, high=400.0, size=100), name="temperature"
        )
        assert isinstance(test_ref, pd.Series)

        if arg_type == "series":
            test_temperature = test_ref.subtract(test_class.kelvin * arg_units)
            compare_temperature = test_class.convert_temperature(
                temperature=test_temperature, base=arg_base
            )
            assert isinstance(compare_temperature, pd.Series)
        else:
            test_data = {0: test_ref, 10: test_ref}
            test_ref = pd.DataFrame(data=test_data, columns=[0, 10])
            assert isinstance(test_ref, pd.DataFrame)

            test_temperature = test_ref.subtract(test_class.kelvin * arg_units)
            compare_temperature = test_class.convert_temperature(
                temperature=test_temperature, base=arg_base
            )
            assert isinstance(compare_temperature, pd.DataFrame)

        if not arg_base:  # output is celsius
            assert np.allclose(
                compare_temperature, test_ref.subtract(test_class.kelvin)
            )
            assert compare_temperature.lt(130).all().all()
        else:
            assert np.allclose(compare_temperature, test_ref)
            assert compare_temperature.gt(130).all().all()
