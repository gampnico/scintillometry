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

Any test that creates a plot should be explicitly appended with
`plt.close()` if the test scope is outside the function, otherwise the
plots remain open in memory.
"""

import datetime

import matplotlib.pyplot as plt
import pytest
import pytz

import scintillometry.visuals.plotting


class TestVisualsFormatting:
    """Tests figure and axis formatting."""

    @pytest.mark.dependency(name="TestVisualsFormatting::test_label_selector")
    @pytest.mark.parametrize(
        "arg_label",
        [
            ["shf", ("Sensible Heat Flux", r"$Q_{H}$", r"[W$\cdot$m$^{-2}$]")],
            ["SHF", ("Sensible Heat Flux", r"$Q_{H}$", r"[W$\cdot$m$^{-2}$]")],
            ["missing key", ("Missing Key", r"$missing key$", "")],
        ],
    )
    def test_label_selector(self, arg_label):
        """Construct axis label and title from dependent variable name."""

        test_label = scintillometry.visuals.plotting.label_selector(
            dependent=arg_label[0]
        )

        assert isinstance(test_label, tuple)
        assert all(isinstance(label, str) for label in test_label)
        assert len(test_label) == 3
        for i in range(0, 3):
            assert test_label[i] == arg_label[1][i]

    @pytest.mark.dependency(name="TestVisualsFormatting::test_get_date_and_timezone")
    def test_get_date_and_timezone(self, conftest_mock_bls_dataframe_tz):
        """Get start date and timezone from dataframe."""

        test_data = conftest_mock_bls_dataframe_tz
        compare_times = scintillometry.visuals.plotting.get_date_and_timezone(
            data=test_data
        )
        assert isinstance(compare_times, dict)
        assert all(key in compare_times for key in ("date", "tzone"))

        assert isinstance(compare_times["date"], str)
        assert compare_times["date"] == "03 June 2020"

        assert isinstance(compare_times["tzone"], datetime.tzinfo)
        assert compare_times["tzone"].zone == "CET"

    @pytest.mark.dependency(name="TestVisualsFormatting::test_title_plot")
    @pytest.mark.parametrize("arg_location", ["", "Test Location", None])
    def test_title_plot(self, arg_location):
        """Construct title and legend."""

        test_fig = plt.figure(figsize=(26, 6))
        test_title = r"Test Title $X_{sub}$"
        test_date = "03 June 2020"

        compare_title = scintillometry.visuals.plotting.title_plot(
            title=test_title, timestamp=test_date, location=arg_location
        )

        assert isinstance(compare_title, str)
        assert compare_title[:20] == test_title
        assert compare_title[-12:] == test_date
        if arg_location:
            location_idx = -14 - len(arg_location)
            assert compare_title[location_idx:-14] == arg_location
        else:
            assert not arg_location
        assert test_fig.legend
        plt.close()

    @pytest.mark.dependency(
        name="TestVisualsFormatting::test_set_xy_labels",
        depends=["TestVisualsFormatting::test_label_selector"],
        scope="class",
    )
    @pytest.mark.parametrize("arg_name", ["shf", "test variable"])
    def test_set_xy_labels(self, arg_name):
        """Construct title and legend."""

        plt.figure(figsize=(26, 6))
        test_timezone = pytz.timezone(zone="CET")
        test_axis = plt.gca()
        compare_axis = scintillometry.visuals.plotting.set_xy_labels(
            ax=test_axis, timezone=test_timezone, name=arg_name
        )

        assert isinstance(compare_axis, plt.Axes)
        assert compare_axis.xaxis.get_label().get_text() == "Time, CET"

        compare_name = compare_axis.yaxis.get_label().get_text()
        if arg_name != "shf":
            assert compare_name == arg_name.title()
        else:
            assert compare_name == r"Sensible Heat Flux, [W$\cdot$m$^{-2}$]"
        plt.close()
