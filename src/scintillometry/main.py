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

Analyse data & 2D flux footprints from Scintec's BLS scintillometers.

Usage: ``src/main.py [-h] [-i <input_data>] [-d] [...] [-v]``

Options and arguments (and corresponding environment variables):

Required arguments::
    ``-i, --input <path>: Path to raw BLS450 data.``
    ``-p, --path <path>: Path to topographical path transect.``

Optional switches::
    ``-h, --help: Show this help message and exit.``
    ``-z, --dry-run: Dry run of model.``
    ``-v, --verbose: Verbose mode.``

Optional arguments::
    ``-t, --timezone <str>: Convert to local timezone. Default "CET".``
    ``-c, --calibrate <float> <float>: Recalibrate path lengths.``
    ``-s, --stability <str>: Set stability condition.``
"""

import argparse

import scintillometry.wrangler.data_parser


def user_argumentation():
    """Parses user arguments when run as main.

    Required arguments::

        -i, --input <path>: Set path to input data file.
        -p, --path <path>: Path to topographical path transect.

    Optional switches::

        -h, --help: Show this help message and exit.
        -z, --dry-run: Dry run of model.
        -v, --verbose: Verbose mode.

    Optional arguments::

        -t, --timezone <str>: Convert to local timezone. Default "CET".
        -c, --calibrate <float> <float>: Recalibrate path lengths.
        -s, --stability <str>: Set stability condition.

    Returns:
        argparse.Namespace: Namespace of user arguments.
    """

    tagline = "Analyse data & 2D flux footprints from Scintec's BLS scintillometers."
    parser = argparse.ArgumentParser(description=tagline)
    # Required
    parser.add_argument(
        "-i",
        "--input",
        default=None,
        dest="input",
        type=str,
        metavar="<path>",
        required=True,
        help="path to raw BLS450 data",
    )
    parser.add_argument(
        "-p",
        "--path",
        default=None,
        dest="path_topography",
        type=str,
        metavar="<path>",
        required=True,
        help="path to topographical path transect",
    )
    # Switches
    parser.add_argument(
        "-z",
        "--dry-run",
        action="store_true",
        default=None,
        dest="dry_run",
        help="dry run",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        default=False,
        dest="verbosity",
        help="verbose mode",
    )
    # Parameters
    parser.add_argument(
        "-t",
        "--timezone",
        dest="timezone",
        metavar="<str>",
        type=str,
        required=False,
        default="CET",
        help="convert to local timezone. Default 'CET'",
    )
    parser.add_argument(  # -c <wrong_length> <correct_length>
        "-c",
        "--calibrate",
        nargs=2,
        dest="calibrate",
        required=False,
        metavar="<float>",
        default=None,
        help="recalibrate path lengths",
    )
    parser.add_argument(
        "-s",
        "--stability",
        dest="stability",
        metavar="<str>",
        type=str,
        required=False,
        default=None,
        choices=["stable", "unstable", None],
        help="set stability condition",
    )

    arguments = parser.parse_args()

    return arguments


def main():
    args = user_argumentation()
    user_input_file = args.input

    # Import raw BLS450 data
    input_data = scintillometry.wrangler.data_parser.parse_scintillometer(
        file_path=user_input_file, timezone=args.timezone, calibration=args.calibrate
    )
    print(input_data.head())


if __name__ == "__main__":
    main()
