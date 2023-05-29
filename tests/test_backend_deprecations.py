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

This module is used to test deprecation decorators, not deprecating
functions.

Use the `conftest_boilerplate` fixture to avoid duplicating tests.
"""

import inspect
from typing import Callable

import pytest

from scintillometry.backend.deprecations import Decorators, DeprecationHandler


class TestBackendDeprecationsMock:
    """Test mock class targeted by decorators."""

    class MockDeprecationsClass:
        """Mock class to call when testing decorators."""

        def add_one(self, a: int):
            """Adds 1 to input integer."""

            a += 1

            return a

    @pytest.mark.dependency(
        name="TestBackendDeprecationsMock::test_mock_deprecations_class"
    )
    def test_mock_deprecations_class(self):
        """Create mock class for testing deprecation."""
        assert inspect.isclass(self.MockDeprecationsClass)
        mock_class = self.MockDeprecationsClass
        assert mock_class.__name__ == "MockDeprecationsClass"
        mock_class_instance = self.MockDeprecationsClass()
        assert mock_class_instance

    @pytest.mark.dependency(
        name="TestBackendDeprecationsMock::test_add_one",
        depends=["TestBackendDeprecationsMock::test_mock_deprecations_class"],
    )
    def test_add_one(self):
        """Add 1 to integer."""

        mock_class_instance = self.MockDeprecationsClass()
        test_integer = 3
        compare_integer = mock_class_instance.add_one(a=test_integer)
        assert isinstance(compare_integer, int)
        assert compare_integer == test_integer + 1


class TestBackendDeprecationsHandler:
    """Handles deprecation methods.

    Attributes:
        test_stages (dict[str, tuple[str, Exception]]): Names of
            deprecation labels and respective text and warning category.
        test_details (dict[str, str]): Mock details for decorator.
        test_mock (type): Mock class called when testing decorators.
        test_mock_instance (MockDeprecationsClass): An instantiated mock
            class called when testing methods.
        test_handler (DeprecationHandler): Class called by decorator
            marking deprecation.
    """

    test_stages = {
        "pending": ("pending deprecation.", PendingDeprecationWarning),
        "deprecated": ("deprecated.", DeprecationWarning),
        "eol": ("deprecated.", FutureWarning),
        "defunct": ("defunct.", RuntimeError),
    }
    test_details = {"version": "1.1.2", "reason": "Some reason."}
    test_mock = TestBackendDeprecationsMock.MockDeprecationsClass
    test_mock_instance = TestBackendDeprecationsMock.MockDeprecationsClass()
    test_handler = DeprecationHandler()

    def setup_warning(
        self,
        obj: Callable,
        stage: str = "deprecated",
        reason: bool = False,
        version: bool = False,
    ):
        """Creates regex string for warning message."""

        test_details = {}
        suffix = ""
        prefix = ""
        if reason:
            test_details["reason"] = self.test_details["reason"]
            suffix = test_details["reason"]
        if version:
            test_details["version"] = self.test_details["version"]
            prefix = f"Ver. {test_details['version']}: "

        suffix_string = " ".join((self.test_stages[stage][0], suffix))
        if inspect.isclass(obj):
            assert obj.__name__ == "MockDeprecationsClass"
            object_string = f"class {obj.__name__}"
        else:
            object_string = f"function {obj.__name__}"

        regex = f"{prefix}The {object_string} is {suffix_string}".strip()

        return regex, prefix, suffix, test_details

    @pytest.mark.parametrize("arg_stage", ["pending", "deprecated"])
    @pytest.mark.parametrize("arg_class", [True, False])
    @pytest.mark.parametrize("arg_reason", [True, False])
    @pytest.mark.parametrize("arg_version", [True, False])
    @pytest.mark.dependency(
        name="TestBackendDeprecationsHandler::test_setup_warning",
        depends=["TestBackendDeprecationsMock::test_add_one"],
    )
    def test_setup_warning(self, arg_class, arg_stage, arg_reason, arg_version):
        """Create regex string to test warning message."""

        if arg_class:
            test_object = self.test_mock
            assert isinstance(test_object, type)
            assert inspect.isclass(test_object)
        else:
            test_object = self.test_mock_instance.add_one
            assert not isinstance(test_object, type)
            assert not inspect.isclass(test_object)
            assert isinstance(test_object, Callable)

        warn_params = self.setup_warning(
            obj=test_object,
            stage=arg_stage,
            reason=arg_reason,
            version=arg_version,
        )
        assert isinstance(warn_params[0], str)
        if arg_version:
            assert warn_params[1] == f"Ver. {self.test_details['version']}: "
            assert "version" in warn_params[3]
        else:
            assert warn_params[1] == ""
            assert "version" not in warn_params[3]
        if arg_reason:
            assert warn_params[2] == self.test_details["reason"]
            assert "reason" in warn_params[3]
        else:
            assert warn_params[2] == ""
            assert "reason" not in warn_params[3]

    def get_mock_object(self, is_class: bool = False):
        """Gets mock class or function."""

        if is_class:
            obj = self.test_mock
        else:
            obj = self.test_mock_instance.add_one

        return obj

    @pytest.mark.dependency(name="TestBackendDeprecationsHandler::test_get_mock_object")
    @pytest.mark.parametrize("arg_class", [True, False])
    def test_get_mock_object(self, arg_class):
        """Get mock class or function."""

        compare_object = self.get_mock_object(is_class=arg_class)
        if arg_class:
            assert inspect.isclass(compare_object)
        else:
            assert not inspect.isclass(compare_object)

    @pytest.mark.dependency(name="TestBackendDeprecationsHandler::test_get_stage_error")
    def test_get_stage_error(self):
        """Raise error for incorrect or missing stage argument."""

        test_stage = "incorrect stage"
        with pytest.raises(
            ValueError, match=f"{test_stage} is not a valid deprecation stage."
        ):
            self.test_handler.get_stage(name=test_stage)
        with pytest.raises(
            TypeError, match=f"{int} is an invalid type. Use {str} instead."
        ):
            self.test_handler.get_stage(name=1)

    @pytest.mark.dependency(
        name="TestBackendDeprecationsHandler::test_get_stage",
        depends=["TestBackendDeprecationsHandler::test_get_stage_error"],
    )
    def test_get_stage(self):
        """Get deprecation stage and warning category."""

        for key, value in self.test_stages.items():
            compare_stage, compare_warning = self.test_handler.get_stage(name=key)
            assert compare_stage == value[0]
            assert compare_warning == value[1]

    @pytest.mark.dependency(
        name="TestBackendDeprecationsHandler::test_get_reason_invalid"
    )
    def test_get_reason_invalid(self):
        """Return empty string for missing value or invalid type."""

        test_kwargs = {"stage": "pending"}
        compare_reason = self.test_handler.get_reason(**test_kwargs)
        assert compare_reason == ""
        compare_reason = self.test_handler.get_reason(reason=1)
        assert compare_reason == ""

    @pytest.mark.dependency(
        name="TestBackendDeprecationsHandler::test_get_reason",
        depends=["TestBackendDeprecationsHandler::test_get_reason_invalid"],
    )
    def test_get_reason(self):
        """Get reason for deprecation."""

        test_kwargs = {"stage": "pending", "reason": "Some reason."}

        compare_reason = self.test_handler.get_reason(**test_kwargs)
        assert compare_reason == test_kwargs["reason"]
        compare_reason = self.test_handler.get_reason(reason=test_kwargs["reason"])
        assert compare_reason == test_kwargs["reason"]

    @pytest.mark.dependency(
        name="TestBackendDeprecationsHandler::test_get_version_invalid"
    )
    def test_get_version_invalid(self):
        """Return empty string for missing version or invalid type."""

        test_kwargs = {"stage": "pending"}
        compare_reason = self.test_handler.get_version(**test_kwargs)
        assert compare_reason == ""
        compare_reason = self.test_handler.get_version(reason=1)
        assert compare_reason == ""

    @pytest.mark.dependency(
        name="TestBackendDeprecationsHandler::test_get_version",
        depends=["TestBackendDeprecationsHandler::test_get_version_invalid"],
    )
    def test_get_version(self):
        """Get release number of last deprecation update."""

        test_kwargs = {"stage": "pending", "version": "1.1.0"}

        compare_version = self.test_handler.get_version(**test_kwargs)
        assert compare_version == f"Ver. {test_kwargs['version']}: "
        compare_version = self.test_handler.get_version(version=test_kwargs["version"])
        assert compare_version == f"Ver. {test_kwargs['version']}: "

    @pytest.mark.dependency(
        name="TestBackendDeprecationsHandler::test_raise_warning",
        depends=["TestBackendDeprecationsHandler::test_setup_warning"],
    )
    @pytest.mark.parametrize("arg_stage", ["pending", "deprecated", "eol"])
    @pytest.mark.parametrize("arg_class", [True, False])
    @pytest.mark.parametrize("arg_reason", [True, False])
    @pytest.mark.parametrize("arg_version", [True, False])
    def test_raise_warning(self, arg_class, arg_stage, arg_reason, arg_version):
        """Raise warning for deprecated object."""

        assert arg_stage in self.test_stages
        test_object = self.get_mock_object(is_class=arg_class)
        test_warning = self.setup_warning(
            obj=test_object, stage=arg_stage, reason=arg_reason, version=arg_version
        )

        with pytest.warns(self.test_stages[arg_stage][1], match=test_warning[0]):
            self.test_handler.raise_warning(
                obj=test_object, stage=arg_stage, details=test_warning[3]
            )

    @pytest.mark.dependency(
        name="TestBackendDeprecationsHandler::test_raise_warning_error",
        depends=["TestBackendDeprecationsHandler::test_setup_warning"],
    )
    @pytest.mark.parametrize("arg_class", [True, False])
    @pytest.mark.parametrize("arg_reason", [True, False])
    @pytest.mark.parametrize("arg_version", [True, False])
    def test_raise_warning_error(self, arg_reason, arg_version, arg_class):
        """Raise RuntimeError when stage="defunct."""

        assert "defunct" in self.test_stages
        test_object = self.get_mock_object(is_class=arg_class)
        test_warning = self.setup_warning(
            obj=test_object, stage="defunct", reason=arg_reason, version=arg_version
        )

        with pytest.raises(self.test_stages["defunct"][1], match=test_warning[0]):
            self.test_handler.raise_warning(
                obj=test_object, stage="defunct", details=test_warning[3]
            )

    @pytest.mark.dependency(
        name="TestBackendDeprecationsHandler::test_rename_arguments",
        depends=["TestBackendDeprecationsHandler::test_setup_warning"],
    )
    @pytest.mark.parametrize("arg_stage", ["pending", "deprecated", "eol"])
    @pytest.mark.parametrize("arg_reason", [True, False])
    @pytest.mark.parametrize("arg_version", [True, False])
    def test_rename_arguments(self, arg_stage, arg_reason, arg_version):
        """Raise warning and redirect deprecated argument."""

        assert arg_stage in self.test_stages
        test_object = self.get_mock_object(is_class=False)
        test_warning = self.setup_warning(
            obj=test_object, stage=arg_stage, reason=arg_reason, version=arg_version
        )
        test_alias_old = "a"
        test_alias_new = "b"
        test_regex = (
            f"{test_warning[1]}The argument {test_alias_old} in {test_object.__name__}",
            f"is {self.test_stages[arg_stage][0]}",
            f"Use {test_alias_new} instead.",
            f"{test_warning[3].get('reason','')}",
        )
        test_kwargs = {test_alias_old: 2, "extra_arg": 3}
        test_alias = {test_alias_old: test_alias_new}

        with pytest.warns(
            self.test_stages[arg_stage][1], match=" ".join(test_regex).strip()
        ):
            self.test_handler.rename_arguments(
                obj=test_object,
                stage=arg_stage,
                kwargs=test_kwargs,
                alias=test_alias,
                version=test_warning[3].get("version", None),
                reason=test_warning[3].get("reason", None),
            )
        assert test_alias_old not in test_kwargs  # argument is replaced
        assert test_alias_new in test_kwargs
        assert test_kwargs[test_alias_new] == 2
        assert "extra_arg" in test_kwargs

    @pytest.mark.dependency(
        name="TestBackendDeprecationsHandler::test_rename_arguments_error",
        depends=["TestBackendDeprecationsHandler::test_setup_warning"],
    )
    @pytest.mark.parametrize("arg_stage", ["pending", "deprecated", "eol"])
    @pytest.mark.parametrize("arg_reason", [True, False])
    @pytest.mark.parametrize("arg_version", [True, False])
    def test_rename_arguments_error(self, arg_stage, arg_reason, arg_version):
        """Raise warning and redirect deprecated argument."""

        assert arg_stage in self.test_stages
        test_object = self.get_mock_object(is_class=False)
        test_warning = self.setup_warning(
            obj=test_object, stage=arg_stage, reason=arg_reason, version=arg_version
        )
        test_alias_old = "a"
        test_alias_new = "b"
        test_regex = (
            f"{test_warning[1]}{test_object.__name__}",
            f"received both {test_alias_old} and {test_alias_new} as arguments.",
            f"{test_alias_old} is {self.test_stages[arg_stage][0]} "
            f"Use {test_alias_new} instead.",
            f"{test_warning[3].get('reason','')}",
        )
        test_kwargs = {test_alias_old: 1, test_alias_new: 3, "extra_arg": 2}
        test_alias = {test_alias_old: test_alias_new}

        with pytest.raises(TypeError, match=" ".join(test_regex).strip()):
            self.test_handler.rename_arguments(
                obj=test_object,
                stage=arg_stage,
                kwargs=test_kwargs,
                alias=test_alias,
                version=test_warning[3].get("version", None),
                reason=test_warning[3].get("reason", None),
            )
        assert test_alias_old in test_kwargs
        assert test_alias_new in test_kwargs
        assert test_kwargs[test_alias_new] == 3  # argument is not replaced
        assert "extra_arg" in test_kwargs

    @pytest.mark.dependency(
        name="TestBackendDeprecationsHandler::test_rename_arguments_missing",
        depends=["TestBackendDeprecationsHandler::test_setup_warning"],
    )
    @pytest.mark.parametrize("arg_stage", ["pending", "deprecated", "eol"])
    @pytest.mark.parametrize("arg_reason", [True, False])
    @pytest.mark.parametrize("arg_version", [True, False])
    def test_rename_arguments_missing(self, arg_stage, arg_reason, arg_version):
        """Raise warning and redirect deprecated argument."""

        assert arg_stage in self.test_stages
        test_object = self.get_mock_object(is_class=False)
        test_warning = self.setup_warning(
            obj=test_object, stage=arg_stage, reason=arg_reason, version=arg_version
        )
        test_alias_old = "a"
        test_alias_new = "b"
        test_kwargs = {test_alias_new: 3, "extra_arg": 2}
        test_alias = {test_alias_old: test_alias_new}

        self.test_handler.rename_arguments(
            obj=test_object,
            stage=arg_stage,
            kwargs=test_kwargs,
            alias=test_alias,
            version=test_warning[3].get("version", None),
            reason=test_warning[3].get("reason", None),
        )
        assert test_alias_old not in test_kwargs
        assert test_alias_new in test_kwargs
        assert test_kwargs[test_alias_new] == 3  # argument is not replaced
        assert "extra_arg" in test_kwargs


class TestBackendDeprecationsDecorator(TestBackendDeprecationsHandler):
    """Tests decorators marking deprecated objects."""

    @pytest.mark.dependency(
        name="TestBackendDeprecationsDecorator::test_decorators_init"
    )
    def test_decorators_init(self):
        test_instance = Decorators()
        assert test_instance
        assert inspect.isclass(Decorators)
        assert inspect.isfunction(Decorators.deprecated)

    @pytest.mark.dependency(
        name="TestBackendDeprecationsDecorator::test_deprecated_decorator",
        depends=["TestBackendDeprecationsMock::test_add_one"],
    )
    @pytest.mark.parametrize("arg_stage", ["pending", "deprecated", "eol"])
    @pytest.mark.parametrize("arg_reason", [True, False])
    @pytest.mark.parametrize("arg_version", [True, False])
    def test_deprecated_decorator(self, arg_stage, arg_reason, arg_version):
        """Use decorator to raise warning."""

        assert arg_stage in self.test_stages
        test_details = {}
        if arg_version:
            test_details["version"] = self.test_details["version"]
        if arg_reason:
            test_details["reason"] = self.test_details["reason"]

        @Decorators.deprecated(stage=arg_stage, **test_details)
        def deprecated_one(x):  # pragma: no cover
            b = self.test_mock_instance.add_one(a=x)
            return b

        test_warning = self.setup_warning(
            obj=deprecated_one, stage=arg_stage, reason=arg_reason, version=arg_version
        )

        with pytest.warns(self.test_stages[arg_stage][1], match=test_warning[0]):
            y = deprecated_one(x=1)
            assert y == 2

    @pytest.mark.dependency(
        name="TestBackendDeprecationsDecorator::test_deprecated_error",
        depends=[
            "TestBackendDeprecationsMock::test_add_one",
            "TestBackendDeprecationsHandler::test_raise_warning",
            "TestBackendDeprecationsHandler::test_raise_warning_error",
        ],
    )
    @pytest.mark.parametrize("arg_reason", [True, False])
    @pytest.mark.parametrize("arg_version", [True, False])
    def test_deprecated_error(self, arg_reason, arg_version):
        """Raise RuntimeError when stage="defunct."""

        assert "defunct" in self.test_stages
        test_details = {}
        if arg_version:
            test_details["version"] = self.test_details["version"]
        if arg_reason:
            test_details["reason"] = self.test_details["reason"]

        @Decorators.deprecated(stage="defunct", **test_details)
        def deprecated_one(x):  # pragma: no cover
            b = self.test_mock_instance.add_one(a=x)
            return b

        test_warning = self.setup_warning(
            obj=deprecated_one, stage="defunct", reason=arg_reason, version=arg_version
        )

        with pytest.raises(
            self.test_stages["defunct"][1],
            match=test_warning[0],
        ):
            deprecated_one(x=1)

    @pytest.mark.dependency(
        name="TestBackendDeprecationsDecorator::test_deprecated_argument_decorator",
        depends=["TestBackendDeprecationsMock::test_add_one"],
    )
    @pytest.mark.parametrize("arg_stage", ["pending", "deprecated", "eol"])
    @pytest.mark.parametrize("arg_reason", [True, False])
    @pytest.mark.parametrize("arg_version", [True, False])
    def test_deprecated_argument_decorator(self, arg_stage, arg_reason, arg_version):
        """Use decorator to raise warning."""

        assert arg_stage in self.test_stages
        test_details = {}
        if arg_version:
            test_details["version"] = self.test_details["version"]
        if arg_reason:
            test_details["reason"] = self.test_details["reason"]
        test_alias_new = "a"

        @Decorators.deprecated_argument(
            stage=arg_stage,
            reason=test_details.get("reason", ""),
            version=test_details.get("version", ""),
            x=test_alias_new,
        )
        def deprecated_one(a):  # pragma: no cover
            b = self.test_mock_instance.add_one(a=a)
            return b

        test_warning = self.setup_warning(
            obj=deprecated_one, stage=arg_stage, reason=arg_reason, version=arg_version
        )
        test_regex = (
            f"{test_warning[1]}The argument x in {deprecated_one.__name__}",
            f"is {self.test_stages[arg_stage][0]}",
            f"Use {test_alias_new} instead.",
            f"{test_warning[3].get('reason','')}",
        )

        with pytest.warns(
            self.test_stages[arg_stage][1], match=" ".join(test_regex).strip()
        ):
            y = deprecated_one(x=1)  # pylint:disable=no-value-for-parameter
            assert y == 2
