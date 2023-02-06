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

import pytest

import scintillometry.backend.constants


class TestBackendConstants:
    """Test class for AtmosConstants class."""

    @pytest.mark.dependency(name="TestBackendConstants::test_init", scope="session")
    def test_constants_init(self):
        test_class = scintillometry.backend.constants.AtmosConstants()
        attributes_dict = test_class.__dict__
        for constant in attributes_dict.values():
            assert isinstance(constant, (float, dict))
            if isinstance(constant, dict):
                assert all(
                    isinstance(value, (float, list)) for value in constant.values()
                )

    @pytest.mark.dependency(name="TestBackendConstants::test_get_error")
    def test_get_error(self):
        """Raise AttributeError for missing attribute."""

        test_name = "raise_error"
        error_msg = f"'AtmosConstants' object has no attribute '{test_name}'"

        with pytest.raises(AttributeError, match=error_msg):
            test_class = scintillometry.backend.constants.AtmosConstants()
            test_class.get(test_name)

    @pytest.mark.dependency(
        name="TestBackendConstants::test_get",
        depends=["TestBackendConstants::test_get_error"],
    )
    def test_get(self):
        """Get attribute value."""

        test_attribute = ("k", 0.4)  # von Kármán's constant, k=0.4
        test_class = scintillometry.backend.constants.AtmosConstants()
        compare_attribute = test_class.get(test_attribute[0])

        assert isinstance(compare_attribute, float)
        assert compare_attribute == pytest.approx(test_attribute[1])

    @pytest.mark.parametrize("arg_value", [-0.4, 0.5, 1])
    def test_overwrite(self, arg_value):
        """Overwrite attribute value."""

        test_class = scintillometry.backend.constants.AtmosConstants()
        test_class.overwrite("k", arg_value)

        assert isinstance(test_class.k, (float, int))
        assert test_class.k == pytest.approx(arg_value)
