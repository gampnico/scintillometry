"""Copyright 2023 Scintillometry Tools Contributors.

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
    -q, --specific-humidity     Derive fluxes from specific humidity.
    -z, --dry-run           Dry run of model.
    -v, --verbose           Verbose mode.

Optional arguments:
    -e, --eddy <str>            Path to eddy covariance data (InnFLUX).
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

import scintillometry.metrics.calculations as MetricsCalculations
import scintillometry.wrangler.data_parser as DataParser


def user_argumentation():
    """Parses user arguments when run as main.

    Required arguments:
        -i, --input <path>      Path to raw BLS450 data.
        -t, --transect <path>   Path to topographical path transect.

    Optional switches:
        -h, --help          Show this help message and exit.
        -q, --specific-humidity    Derive fluxes from specific humidity.
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
        "-q",
        "--specific-humidity",
        action="store_true",
        default=None,
        dest="specific_humidity",
        help="derive fluxes from specific humidity",
    )
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
        choices=["sun", "static", "bulk", "eddy"],
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
        help="path to eddy covariance data (InnFLUX)",
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


def main():
    args = user_argumentation()
    print(args)
    if args.specific_humidity:
        raise NotImplementedError(
            "Deriving fluxes from specific humidity is not yet implemented."
        )

    # Import and parse BLS450, ZAMG, transect data
    parsed_datasets = DataParser.wrangle_data(
        bls_path=args.input,
        transect_path=args.transect_path,
        calibrate=args.calibration,
        tzone=args.timezone,
        station_id=args.station_id,
    )

    # Parse vertical measurements
    if args.profile_prefix:
        parsed_datasets["vertical"] = DataParser.parse_vertical(
            file_path=args.profile_prefix,
            device="hatpro",
            levels=None,
            tzone=args.timezone,
        )

    metrics_class = MetricsCalculations.MetricsWorkflow()
    metrics_data = metrics_class.calculate_standard_metrics(
        arguments=args, data=parsed_datasets
    )
    if args.eddy_path:
        innflux_frame = DataParser.parse_innflux(
            file_name=args.eddy_path,
            tzone=args.timezone,
            headers=None,
        )
        metrics_class.compare_innflux(
            arguments=args,
            innflux_data=innflux_frame,
            comparison_data=metrics_data["iteration"],
        )


if __name__ == "__main__":
    main()
