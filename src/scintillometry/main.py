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

Analyse data & 2D flux footprints from Scintec's BLS scintillometers.

Usage: ``src/main.py [-h] [-i <input_data>] [-d] [...] [-v]``

Options and arguments (and corresponding environment variables):

Required arguments:
    -i, --input <path>      Path to raw BLS450 data.
    -t, --transect <path>       Path to topographical path transect.

Optional switches:
    -h, --help              Show this help message and exit.
    -z, --dry-run           Dry run of model.
    -v, --verbose           Verbose mode.

Optional arguments:
    -e, --eddy <str>            Path to eddy covariance data (innFLUX).
    -p, --profile <str>         Path prefix to vertical temperature and
                                    humidity measurements (HATPRO).
    -l, --local-timezone <str>      Convert to local timezone.
                                        Default "CET".
    -c, --calibrate <float float>       Recalibrate path lengths.
    -r, --regime <str>          Set default stability condition.
    -m, --most-name <str>       ID of MOST coefficients.
                                    Default "an1988".
    -s, --switch-time <str>     Override local time of switch between
                                    stability regimes.
    -a, --algorithm <str>       Algorithm used to calculate switch time.
                                    Default "sun".
    -k, --station-id <str>      ZAMG station ID (Klima-ID).
                                    Default 11803.
    --location <str>            Location of experiment. Overrides any
                                    other location metadata.
    --beam-wavelength <int>     Transmitter beam wavelength, nm.
                                    Default 850 nm.
    --beam-error <int>          Transmitter beam wavelength error, nm.
                                    Default 20 nm.
"""

import argparse

import scintillometry.metrics.calculations as calculations
import scintillometry.wrangler.data_parser as data_parser


def user_argumentation():
    """Parses user arguments when run as main.

    Required arguments:
        -i, --input <path>      Path to raw BLS450 data.
        -t, --transect <path>   Path to topographical path transect.

    Optional switches:
        -h, --help          Show this help message and exit.
        -z, --dry-run       Dry run of model.
        -v, --verbose       Verbose mode.

    Optional arguments:
        -e, --eddy <str>        Path to eddy covariance data (InnFLUX).
        -p, --profile <str>     Path prefix to vertical temperature and
                                    humidity measurements (HATPRO).
        -l, --local-timezone <str>      Convert to local timezone.
                                            Default "CET".
        -c, --calibrate <float float>       Recalibrate path lengths.
        -r, --regime <str>          Set default stability condition.
        -m, --most-name <str>       ID of MOST coefficients.
                                        Default "an1988".
        -s, --switch-time <str>     Override local time of switch
                                        between stability regimes.
        -a, --algorithm <str>       Algorithm used to calculate switch time.
                                    Default "sun".
        -k, --station-id <str>      ZAMG station ID (Klima-ID).
                                        Default 11803.
        --location <str>            Location of experiment. Overrides
                                        any other location metadata.
        --beam-wavelength <int>     Transmitter beam wavelength, nm.
                                        Default 850 nm.
        --beam-error <int>          Transmitter beam wavelength error,
                                        nm. Default 20 nm.

    Returns:
        argparse.Namespace: Namespace of user arguments.
    """

    tagline = "Analyse data & 2D flux footprints from Scintec's BLS scintillometers."
    parser = argparse.ArgumentParser(prog="scintillometry-tools", description=tagline)
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
        "-t",
        "--transect",
        default=None,
        dest="transect_path",
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
    group = parser.add_mutually_exclusive_group()
    # --switch-time and --algorithm are incompatible
    group.add_argument(
        "-s",
        "--switch-time",
        dest="switch_time",
        metavar="<str>",
        type=str,
        required=False,
        default=None,
        help="override local time of switch between stability regimes",
    )
    group.add_argument(
        "-a",
        "--algorithm",
        dest="algorithm",
        metavar="<str>",
        type=str,
        required=False,
        default="sun",
        choices=["sun", "bulk", "lapse", "static"],
        help="algorithm used to calculate switch time. Default 'sun'",
    )
    parser.add_argument(
        "-p",
        "--profile",
        default=None,
        dest="profile_prefix",
        type=str,
        metavar="<path>",
        required=False,
        help="path prefix to vertical temperature and humidity measurements (HATPRO)",
    )
    parser.add_argument(
        "-e",
        "--eddy",
        default=None,
        dest="eddy_path",
        type=str,
        metavar="<path>",
        required=False,
        help="path to eddy covariance data (innFLUX)",
    )
    parser.add_argument(
        "-l",
        "--local-timezone",
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
        dest="calibration",
        required=False,
        metavar="<float>",
        default=None,
        help="recalibrate path lengths",
    )
    parser.add_argument(
        "-r",
        "--regime",
        dest="regime",
        metavar="<str>",
        type=str,
        required=False,
        default=None,
        choices=["stable", "unstable", None],
        help="set default stability condition",
    )
    parser.add_argument(
        "-m",
        "--most-name",
        dest="most_name",
        metavar="<str>",
        type=str,
        required=False,
        default="an1988",
        choices=["an1988", "li2012", "wy1971", "wy1973", "ma2014", "br2014"],
        help="ID of MOST coefficients. Default 'an1988'",
    )

    parser.add_argument(
        "-k",
        "--station-id",
        dest="station_id",
        metavar="<str>",
        type=str,
        required=False,
        default=11803,
        help="ZAMG station ID (Klima-ID). Default 11803",
    )
    parser.add_argument(
        "--location",
        dest="location",
        metavar="<str>",
        type=str,
        required=False,
        default="",
        help="Location of experiment. Overrides any other location metadata",
    )
    parser.add_argument(
        "--beam-wavelength",
        dest="beam_wavelength",
        metavar="<int>",
        type=int,
        required=False,
        default=880,
        help="Transmitter beam wavelength, nm. Default 880 nm",
    )
    parser.add_argument(
        "--beam-error",
        dest="beam_error",
        metavar="<int>",
        type=int,
        required=False,
        default=20,
        help="Transmitter beam error, nm. Default 20 nm",
    )

    arguments = parser.parse_args()

    return arguments


def perform_data_parsing(**kwargs):
    """Parses data from command line arguments.

    Keyword Arguments:
        bls_path (str): Path to a raw .mnd data file using FORMAT-1.
        transect_path (str): Path to processed transect. The data must
            be formatted as <path_height>, <normalised_path_position>.
            The normalised path position maps to:
            [0: receiver location, 1: transmitter location].
        calibration (list): Contains the incorrect and correct path
            lengths. Format as [incorrect, correct].
        station_id (str): ZAMG weather station ID (Klima-ID).
            Default 11803.
        timezone (str): Local timezone during the scintillometer's
            operation. Default "CET".
        profile_prefix (str): Path to vertical measurements. For HATPRO
            Retrieval data there should be two HATPRO files ending with
            "humidity" and "temp". The path should be identical for both
            files, e.g.::

                ./path/to/file_humidity.csv
                ./path/to/file_temp.csv

            would require `file_path = "./path/to/file_"`. Default None.

    Returns:
        dict: Parsed and labelled datasets for scintillometry
        measurements, weather observations, and topography.
    """

    parser = data_parser.WranglerParsing()

    # Parse BLS, weather, and topographical data
    datasets = parser.wrangle_data(
        bls_path=kwargs["input"],
        transect_path=kwargs["transect_path"],
        calibrate=kwargs["calibration"],
        station_id=kwargs["station_id"],
        tzone=kwargs["timezone"],
        weather_source="zamg",
    )

    # Parse vertical measurements
    if kwargs["profile_prefix"]:
        datasets["vertical"] = parser.vertical.parse_vertical(
            file_path=kwargs["profile_prefix"],
            source="hatpro",
            levels=None,
            tzone=kwargs["timezone"],
        )

    return datasets


def perform_analysis(datasets, **kwargs):
    """Analyses flux data.

    Calculates and plots parsed data, and optionally compares it to
    third-party data.

    Defaults for keyword arguments only apply if the kwargs are passed
    via command line arguments.

    Arguments:
        datasets (dict): Parsed and labelled datasets for scintillometry
            measurements, weather observations, topography, and
            optionally vertical measurements.

    Keyword Args:
        eddy_path (str): Path to eddy covariance measurements.
            Default None.
        regime (str): Target stability condition. Default None.
        timezone (str): Local timezone of the measurement period.
            Default "CET".
        most_name (str): MOST coefficients for unstable and stable
            conditions. Default "an1988".
        method (str): Method to calculate switch time. Default "sun".
        switch_time (Union[str, pd.Timestamp]): Local time of switch
            between stability conditions. Overrides <method>.
            Default None.
        location (str): Location of data collection. Default empty
            string.
        beam_wavelength (int): Transmitter beam wavelength, nm.
            Default 880 nm.
        beam_error (int): Transmitter beam error, nm. Default 20 nm.

    Returns:
        dict: Passes input datasets. If a path to eddy covariance data
        is provided, adds the key "eddy" containing the parsed eddy
        covariance data.
    """

    metrics_class = calculations.MetricsWorkflow()
    parser = data_parser.WranglerParsing()
    metrics_data = metrics_class.calculate_standard_metrics(data=datasets, **kwargs)
    if kwargs["eddy_path"]:
        eddy_frame = parser.eddy.parse_eddy_covariance(
            file_path=kwargs["eddy_path"], tzone=kwargs["timezone"], source="innflux"
        )
        metrics_data["eddy"] = eddy_frame
        metrics_class.compare_eddy(
            own_data=metrics_data["iteration"],
            ext_data=eddy_frame,
            source="innflux",
            location=kwargs["location"],
        )

    return metrics_data


def main():
    """Parses command line arguments and executes analysis.

    Converts command line arguments into kwargs. Imports and parses
    scintillometer, weather, and transect data. If the appropriate
    arguments are specified:

        - Parses vertical measurements
        - Calculates sensible heat fluxes
        - Compares calculated fluxes to external data.

    The majority of kwarg expansions should occur in this module. Do not
    rely on kwargs for passing arguments between backend modules.
    """

    arguments = user_argumentation()
    kwarg_args = vars(arguments)

    parsed_datasets = perform_data_parsing(**kwarg_args)
    if not arguments.dry_run:
        perform_analysis(datasets=parsed_datasets, **kwarg_args)
    else:
        print("Dry run - no analysis available.")


if __name__ == "__main__":
    main()
