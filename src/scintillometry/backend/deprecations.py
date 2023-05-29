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

Handles deprecating, deprecated, and defunct code. The deprecation
process takes one patch release cycle, the removal process takes two
major release cycles.

EOL Cycle
---------

Import the decorator
:func:`~scintillometry.backend.deprecations.deprecated` from this
module. Mark deprecating functions like so:

.. code-block:: python

    @deprecated(stage="deprecated", reason="Some reason", version="1.1.1")
    def foobar(...)
        ...

Where ``stage`` is the stage in the function's deprecation cycle,
``reason`` is an optional message, and ``version`` is the release number
when ``stage`` was last updated.

Update ``stage`` by following this schedule:

1. A function pending deprecation is marked as **pending** and issues a
PendingDeprecationWarning during patch development.

2. The function is marked as **deprecated** and issues a
DeprecationWarning from the next patch release. It must still work as
intended.

3. The function is marked as **eol** and issues a FutureWarning during
the next minor release cycle. It must still work as intended.

4. The function is marked as **defunct** and throws an error during the
next major release cycle, but is not removed.

5. The defunct function is entirely removed from release candidates in
preparation for the subsequent major release.
"""


import functools
import inspect
import warnings
from typing import Callable


class DeprecationHandler:
    """Methods for marking and handling deprecation cycles.

    Attributes:
        string_types (tuple): Supported string types (str and
            byte literals).
    """

    def __init__(self):
        super().__init__()
        self.string_types = (type(b""), type(""))  # support byte literals

    def get_stage(self, name):
        """Gets stage in deprecation cycle.

        Args:
            name (str): Stage of deprecation cycle. Valid values
                are:

                    - pending
                    - deprecated
                    - eol
                    - defunct

        Returns:
            tuple[str, Exception]: The description of the deprecation stage,
            and the Exception subclass matching this stage.

        Raises:
            TypeError: <type(name)> is an invalid type. Use str instead.
            ValueError: <name> is not a valid deprecation stage.
        """

        if isinstance(name, self.string_types):
            stage_lower = name.lower()
            if stage_lower == "deprecated":
                stage_string = "deprecated."
                category = DeprecationWarning
            elif stage_lower == "pending":
                stage_string = "pending deprecation."
                category = PendingDeprecationWarning
            elif stage_lower == "eol":
                stage_string = "deprecated."
                category = FutureWarning
            elif stage_lower == "defunct":
                stage_string = "defunct."
                category = RuntimeError
            else:
                raise ValueError(f"{stage_lower} is not a valid deprecation stage.")
        else:
            raise TypeError(f"{type(name)} is an invalid type. Use {str} instead.")

        return stage_string, category

    def get_reason(self, **kwargs):
        """Gets reason for deprecation.

        Keyword Args:
            reason (str): Reason for deprecation.

        Returns:
            str: The reason for deprecation if available, otherwise returns
            an empty string.
        """

        reason = kwargs.get("reason", None)
        if not isinstance(reason, self.string_types):
            reason = ""

        return reason

    def get_version(self, **kwargs):
        """Gets release number of latest stage in deprecation.

        Keyword Args:
            version (str): Release number of latest stage in deprecation.

        Returns:
            str: Formatted version number of latest stage in deprecation if
            provided, otherwise returns an empty string.
        """

        version = kwargs.get("version", None)
        if not isinstance(version, self.string_types):
            version = ""
        else:
            version = f"Ver. {version}: "

        return version

    def raise_warning(self, obj, stage, details):
        """Raises warning or error with informative message.

        Raises a warning or error stating the function or class' stage in
        its deprecation cycle. Optionally, lists a release number and
        reason.

        Args:
            obj (Callable): The function or class being deprecated.
            stage (str): The current stage in the deprecation cycle.
            details (dict): A dictionary map optionally containing the
                keys:

                    - **reason**: the reason for deprecation.
                    - **version**: the release number of the latest
                      change in the deprecation cycle.

        Raises:
            RuntimeError: The function <obj.__name__> is <stage>.
        """

        stage_string, warn_class = self.get_stage(name=stage)
        reason = self.get_reason(**details)
        version = self.get_version(**details)
        suffix = " ".join((stage_string, reason))

        if inspect.isclass(obj):
            warn_string = f"{version}The class {obj.__name__} is {suffix}"
        else:
            warn_string = f"{version}The function {obj.__name__} is {suffix}"

        if stage.lower() != "defunct":
            warnings.warn(
                message=warn_string.strip(),
                category=warn_class,
                stacklevel=2,
            )
        else:
            raise RuntimeError(warn_string.strip())

    def rename_arguments(self, obj, stage, kwargs, alias, reason=None, version=None):
        """Marks argument as deprecated and redirects to alias.

        The wrapped function's arguments are wrapped into `kwargs` and
        are safe from being overwritten by arguments in alias.

        Args:
            obj (Callable): The function or class being deprecated.
            stage (str): The current stage in the deprecation cycle.
            kwargs (dict): Keyword arguments.
            alias (dict): A dictionary map optionally containing the
                keys:

            reason (str): The reason for deprecation. Default None.
            version (str): The release number of the latest change in
                the deprecation cycle. Default None.
        """

        stage_string, warn_class = self.get_stage(name=stage)
        reason = self.get_reason(**{"reason": reason})
        version = self.get_version(**{"version": version})

        for old, new in alias.items():
            if old in kwargs:
                if new in kwargs:
                    warn_string = (
                        f"{version}{obj.__name__}",
                        f"received both {old} and {new} as arguments. {old}",
                        f"is {stage_string}",
                        f"Use {new} instead.",
                        f"{reason}",
                    )
                    raise TypeError(" ".join(warn_string).strip())
                else:
                    warn_string = (
                        f"{version}The argument {old} in {obj.__name__}",
                        f"is {stage_string}",
                        f"Use {new} instead.",
                        f"{reason}",
                    )
                    warnings.warn(
                        message=" ".join(warn_string).strip(),
                        category=warn_class,
                        stacklevel=2,
                    )
                kwargs[new] = kwargs.pop(old)


class Decorators:
    def __init__(self):
        super().__init__()

    @staticmethod
    def deprecated(stage="deprecated", **details):
        """Decorator for deprecated function and method arguments.

        Example:

        .. code-block:: python

            @deprecated(stage="pending", reason="Foobar", version="1.3.2")
            def some_function(foo):
                ...


        Args:
            stage(str): Stage of deprecation cycle. Valid values
                are:

                    - pending
                    - deprecated
                    - eol
                    - defunct

                Default "deprecated".
        Keyword Args:
            reason (str): Reason for deprecation.
            version (str): Release number of latest stage in deprecation.

        Returns:
            Callable: Decorator for deprecated argument.
        """

        internals = DeprecationHandler()

        def decorator(f: Callable):
            @functools.wraps(f)
            def wrapper(*args, **kwargs):
                internals.raise_warning(f, stage, details)
                return f(*args, **kwargs)

            return wrapper

        return decorator

    @staticmethod
    def deprecated_argument(stage="deprecated", reason=None, version=None, **aliases):
        """Decorator for deprecated function and method arguments.

        Use as follows:

        .. code-block:: python

            @deprecated_argument(old_argument="new_argument")
            def myfunc(new_arg):
                ...

        Args:
            stage(str): Stage of deprecation cycle. Valid values
                are:

                    - pending
                    - deprecated
                    - eol
                    - defunct

                Default "deprecated".
            reason (str): Reason for deprecation. Default None.
            version (str): Release number of latest stage in
                deprecation. Default None.
            aliases (dict[str, str]): Deprecated argument and its
                alternative.

        Returns:
            Callable: Decorator for deprecated argument.

        """

        internals = DeprecationHandler()

        def decorator(f: Callable):
            @functools.wraps(f)
            def wrapper(*args, **kwargs):
                internals.rename_arguments(
                    obj=f,
                    stage=stage,
                    reason=reason,
                    version=version,
                    kwargs=kwargs,
                    alias=aliases,
                )
                return f(*args, **kwargs)

            return wrapper

        return decorator
