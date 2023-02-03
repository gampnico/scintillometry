"""Copyright 2019-2023 Nicolas Gampierakis, Josef Zink.

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

Calculates path weighting and effective height from a path transect.
"""

import math

import numpy as np
import scipy


def bessel_second(x):
    """Calculates the Bessel function for a specific path position.

    Args:
        x (float): Normalised path position.

    Returns:
        float: Bessel function of path position.
    """

    bessel_variable = 2.283 * math.pi * (x - 0.5)
    if bessel_variable == 0:
        y = 1
    else:
        y = 2 * (scipy.special.jv(1, bessel_variable)) / bessel_variable

    return y


def path_weighting(path_coordinates):
    """Path weighting function for computing effective path height.

    Args:
        path_coordinates (pd.Series): Normalised path position along
            path transect.

    Returns:
        list: Contains weights for each coordinate along the path.
    """

    weights = []
    for i in path_coordinates:
        weights.append(2.163 * bessel_second(i))

    return weights


def define_stability(stability_name):
    """Checks implementation and gets b-value for stability condition.

    Args:
        stability_name (str): Name of stability condition.

    Returns:
        float: Constant "b" accounting for height dependence of |Cn2|.
            Values of "b" are from Hartogenesis et al. (2003), and
            Kleissl et al. (2008).

    Raises:
        NotImplementedError: <stability> is not an implemented stability
            condition.
    """

    # Hartogenesis et al. (2003), Kleissl et al. (2008).
    stability_dict = {"stable": -2 / 3, "unstable": -4 / 3}

    if not stability_name:
        b_constant = 1
        print("No height dependency selected.")
    elif stability_name in stability_dict.keys():
        b_constant = stability_dict[stability_name]
    else:
        error_msg = f"{stability_name} is not an implemented stability condition."
        raise NotImplementedError(error_msg)

    return b_constant


def compute_effective_z(path_heights, path_positions, stability):
    """Computes effective path height for a path transect.

    Calculates the effective path height across the entire
    scintillometer beam path, using the actual path heights and
    normalised positions.

    Args:
        path_heights (pd.Series): Actual path heights, in metres.
        path_positions (pd.Series): Normalised positions along transect.
        stability (str): Stability conditions. Can be stable, unstable,
            or None for no height dependency.

    Returns:
        np.floating: Effective path height, |z_eff| [m].
    """

    b_value = define_stability(stability_name=stability)

    weighted_heights = np.multiply(
        path_heights**b_value, path_weighting(path_positions)
    )

    # For path-building intersections
    weighted_heights[np.isnan(weighted_heights)] = 0
    z_eff = (
        (scipy.integrate.trapz(weighted_heights))
        / (scipy.integrate.trapz(path_weighting(path_positions)))
    ) ** (1 / b_value)

    return z_eff


def get_z_parameters(transect_data, stability_condition):
    """Calculates effective and mean path height for a path transect.

    Args:
        transect_data (pd.DataFrame): Parsed path transect data.
        stability_condition (str): Stability conditions.

    Returns:
        tuple[np.floating, np.floating]: Effective and mean path height
            of transect.
    """

    effective_path_height = compute_effective_z(
        path_heights=transect_data["path_height"],
        path_positions=transect_data["norm_position"],
        stability=stability_condition,
    )
    mean_path_height = np.mean(transect_data["path_height"])

    return effective_path_height, mean_path_height


def get_all_z_parameters(path_transect):
    """Calculates effective & mean path heights in all conditions.

    Args:
        path_transect (pd.DataFrame): Parsed path transect data.

    Returns:
        dict[tuple[np.floating, np.floating]]: Effective and mean path
            heights of transect, with each stability condition as key.
    """

    path_heights_dict = {}
    for stability in ["stable", "unstable", None]:
        path_heights = get_z_parameters(
            transect_data=path_transect, stability_condition=stability
        )
        path_heights_dict[str(stability)] = path_heights  # str(None) == "None"

    return path_heights_dict


def print_z_parameters(z_eff, z_mean, stability):
    """Prints effective and mean path height for specific conditions.

    Args:
        z_eff (np.floating): Effective path height of transect,
            |z_eff| [m].
        z_mean (np.floating): Mean effective path height of transect,
            |z_mean| [m].
        stability (str): Stability conditions. Can be stable, unstable,
            or None for no height dependency.
    """

    if not stability:
        stability_suffix = "no height dependency"
    else:
        stability_suffix = f"{stability} conditions"

    z_eff_print = (
        f"Selected {stability_suffix}:\n",
        f"Effective path height:\t{z_eff:>0.2f} m.\n",
        f"Mean path height:\t{z_mean:>0.2f} m.\n",
    )
    print("".join(z_eff_print))
