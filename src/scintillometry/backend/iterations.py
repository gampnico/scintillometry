"""Copyright 2019-2023 Scintillometry Contributors.

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

Iterative methods to calculate heat fluxes.

IterationMost is descended from an old Python port of a method initially
written in R by Dr. Helen Ward (2019), but has been completely
rewritten - the implementations and features available are fundamentally
different so the results from one should not be assumed identical to the
other.

The current method used by IterationMost is based on mathematical theory
available in:

    - Scintec AG (2022). Scintec Scintillometers Theory Manual,
      Version 1.05. Scintec AG, Rottenburg, Germany. [#scintec2022]_

    - Scintec AG (2008). Scintec Boundary Layer Scintillometer User
      Manual, Version 1.49. Scintec AG, Rottenburg, Germany.
      [#scintec2008]_

This method has been updated to resolve some inconsistencies between
the manuals, and has additional modifications to enhance performance and
broaden its use with different scintillometer models. Results may
therefore differ slightly between this and other iterative methods.

TODO:
    - ST-45: Implement MM5 method (Zhang and Anthes, 1982).
    - ST-46: Implement Li method (Li et al., 2014; 2015).
"""

import time
import warnings

import mpmath
import numpy as np
import tqdm

from scintillometry.backend.constants import AtmosConstants


class IterationMost:
    """Classic MOST Iteration.

    A detailed description of the iterative scheme is available in the
    Scintec Scintillometers Theory Manual, Version 1.05. [#scintec2022]_

    This iterative method resolves some inconsistencies between
    different scintillometer manuals. Results may therefore differ
    slightly with other third-party implementations.

    Attributes:
        constants (AtmosConstants): Inherited atmospheric constants.
    """

    def __init__(self):
        super().__init__()
        mpmath.dps = 30
        mpmath.pretty = True
        self.constants = AtmosConstants()

    def momentum_stability_unstable(self, obukhov, z):
        """Integrated stability function for momentum, unstable.

        Uses formula from Paulson (1970) [#paulson1970]_. Apply when
        |LOb| < 0.

        Args:
            obukhov (float): Obukhov length, |LOb| [m].
            z (float): Height, |z| [m].

        Returns:
            mpmath.mpf: Integrated stability function for momentum,
            |Psi_m| [0-Dim].
        """

        x = (1 - 16 * (z / obukhov)) ** (1 / 4)
        psi_momentum = (
            2 * mpmath.ln((1 + x) / 2)
            + mpmath.ln((1 + x**2) / 2)
            - 2 * mpmath.atan(x)
            + mpmath.pi / 2
        )

        return psi_momentum

    def momentum_stability_stable(self, obukhov, z):
        """Integrated stability function for momentum, stable.

        Apply when |LOb| > 0.

        Args:
            obukhov (float): Obukhov length, |LOb| [m].
            z (float): Height, |z| [m].

        Returns:
            mpmath.mpf: Integrated stability function for momentum,
            |Psi_m| [0-Dim].
        """

        psi_momentum = (-5) * z / obukhov

        return mpmath.mpmathify(psi_momentum)

    def momentum_stability(self, obukhov, z):
        """Integrated stability function for momentum.

        Calculates integrated stability function for momentum under
        stable or unstable conditions. No implementation for neutral
        stability.

        Args:
            obukhov (float): Obukhov length, |LOb| [m].
            z (float): Height, |z| [m].

        Returns:
            mpmath.mpf: Integrated stability function for momentum,
            |Psi_m| [0-Dim].
        """

        if obukhov > 0:
            psi_m = self.momentum_stability_stable(obukhov, z)
        else:
            psi_m = self.momentum_stability_unstable(obukhov, z)

        return psi_m

    def get_most_coefficients(self, most_id="an1988", most_type="ct2"):
        """Returns MOST coefficients for similarity functions.

        Implemented MOST coefficients:

        - **an1988**: E.L. Andreas (1988) [#andreas1988]_
        - **li2012**: D. Li et al. (2012) [#li2012]_
        - **wy1971**: Wyngaard et al. (1971) [#wyngaard1971]_
        - **wy1973**: Wyngaard et al. (1973) in Kooijmans and
          Hartogensis (2016) [#kooijmans2016]_
        - **ma2014**: Maronga et al. (2014) [#maronga2014]_
        - **br2014**: Braam et al. (2014) [#braam2014]_

        Braam et al. (2014) and Maronga et al. (2014) do not provide
        coefficients for stable conditions, so gradient functions will
        evaluate to zero for stable conditions.

        Args:
            most_id (str): MOST coefficients for unstable and stable
                conditions. Default "an1988".
            most_type (str): MOST function. Default "ct2".

        Returns:
            list: MOST coefficients formatted as
            ``[(unstable, unstable), (stable, stable)]``

        Raises:
            NotImplementedError: MOST coefficients are not implemented
                for <most_id>.
            NotImplementedError: MOST coefficients are not implemented
                for functions of <most_type>.
        """

        if most_type.lower() != "ct2":
            raise NotImplementedError(
                f"MOST coefficients are not implemented for functions of {most_type}."
            )
        elif most_id.lower() not in self.constants.most_coeffs_ft:
            raise NotImplementedError(
                f"MOST coefficients are not implemented for {most_id}."
            )
        else:
            coeffs = self.constants.most_coeffs_ft[most_id.lower()]

        return coeffs

    def similarity_function(self, obukhov, z, coeffs, stable):
        """Similarity function of |CT2|.

        Adapted from Wyngaard et al. (1971) [#wyngaard1971]_; Thiermann
        and Grassl (1992) [#thiermann1992]_; Kooijmans and Hartogensis (2016)
        [#kooijmans2016]_.

        Args:
            obukhov (float): Obukhov length, |LOb| [m].
            z (float): Effective height, |z| [m].
            stable (bool): True for stable conditions, otherwise False.
            coeffs (list): MOST coefficients for unstable and stable
                conditions.

        Returns:
            float: Similarity function of |CT2|, |f_CT2|.
        """

        if obukhov == 0:  # otherwise zero division
            obukhov += 1e-10

        # Standard shape from Kooijmans and Hartogensis (2016)
        if not stable:
            f_ct2 = coeffs[0][0] * (1 - coeffs[0][1] * (z / obukhov)) ** (-2 / 3)
        else:
            f_ct2 = coeffs[1][0] * (1 + coeffs[1][1] * ((z / obukhov) ** (2 / 3)))

        return f_ct2

    def calc_theta_star(self, ct2, f_ct2, z, stable):
        """Calculate temperature scale.

        Args:
            ct2 (float): Structure parameter of temperature,
                |CT2| [|K^2m^-2/3|].
            f_ct2 (float): MOST function of |CT2|, |f_CT2|.
            z (float): Effective height, |z| [m].
            stable (bool): True for stable conditions, otherwise False.

        Returns:
            mpmath.mpf: Temperature scale, |theta*|.
        """

        theta_star = mpmath.sqrt(ct2 * (z ** (2 / 3)) / f_ct2)
        if not stable:
            theta_star = -theta_star

        return theta_star

    def calc_u_star(self, u, z_u, r_length, o_length):
        """Calculates friction velocity.

        Args:
            u (float): Wind speed, |u| [|ms^-1|].
            z_u (float): Height of wind speed measurement including
                displacement (z-d), |z_u| [m].
            r_length (float): Roughness length, |z_0| [m].
            o_length (float): Obukhov length |LOb| [m].

        Returns:
            mpmath.mpf: Friction velocity |u*|.
        """

        friction_velocity = (
            self.constants.k
            * u
            / (
                mpmath.ln(z_u / r_length)
                - self.momentum_stability(o_length, z=z_u)
                + self.momentum_stability(o_length, z=r_length)
            )
        )

        return friction_velocity

    def calc_obukhov_length(self, temp, u_star, theta_star):
        """Calculate Obukhov length.

        Args:
            temp (np.floating): Air temperature in Kelvin, |T| [K].
            u_star (mpmath.mpf): Friction velocity |u*|.
            theta_star (mpmath.mpf): Temperature scale, |theta*|.

        Returns:
            mpmath.mpf: Obukhov Length, |LOb| [m].
        """

        obukhov_length = (temp * (u_star**2)) / (
            self.constants.g * self.constants.k * theta_star
        )

        return obukhov_length

    def check_signs(self, stable_flag, dataframe):
        """Warns if sign of variable doesn't match expected sign.

        Args:
            stable_flag (bool): True for stable conditions, otherwise
                False.
            dataframe (pd.DataFrame): Dataframe with sensible heat flux
                and Obukhov lengths.

        Warns:
            UserWarning: Sign of Obukhov lengths should be opposite of
                SHFs.
            UserWarning: SHFs should never be positive for stable
                conditions.
            UserWarning: Obukhov lengths should never be negative for
                stable conditions.
        """

        if stable_flag and (dataframe["shf"] > 0).any():
            msg = "SHFs should never be positive for stable conditions."
            warnings.warn(UserWarning(msg))
        if stable_flag and (dataframe["obukhov"] < 0).any():
            msg = "Obukhov lengths should never be negative for stable conditions."
            warnings.warn(UserWarning(msg))

        # use "not" with <= to avoid warnings when comparing zeros.
        if not ((dataframe["shf"] <= 0).any()) == ((dataframe["obukhov"] >= 0).any()):
            msg = "Sign of Obukhov lengths should be opposite of SHFs."
            warnings.warn(UserWarning(msg))

    def most_iteration(self, dataframe, zm_bls, stable_flag, most_coeffs):
        """Iterative MOST method.

        A detailed description of the iterative method is available in:

            - Scintec AG (2022). *Scintec Scintillometers Theory
              Manual*, Version 1.05. Scintec AG, Rottenburg, Germany.
              [#scintec2022]_


        Iteration until convergence is slower than vectorisation, but
        more accurate. If a value never converges, the iteration stops
        after ten runs.

        Args:
            dataframe (pd.DataFrame): Parsed, localised dataframe row
                containing at least |CT2|, wind speed, air density,
                and temperature.
            zm_bls (float): Effective height of scintillometer.
            stable_flag (bool): Stability conditions. If true, assumes
                stable conditions, otherwise assumes unstable
                conditions.
            most_coeffs (list): MOST coefficients for unstable and
                stable conditions.

        Returns:
            pd.DataFrame: Dataframe with additional columns for Obukhov
            lengths, sensible heat fluxes, frictional velocity, and
            temperature scale.
        """

        z0 = zm_bls * 0.1  # estimated roughness length
        iter_step = 0  # enforce limit if no convergence
        obukhov_diff = 1

        # converges around 5 iterations, but 10 is safer
        while not (np.abs(obukhov_diff) < 0.1 or iter_step >= 10):
            dataframe["f_ct2"] = self.similarity_function(
                obukhov=dataframe["obukhov"],
                z=zm_bls,
                stable=stable_flag,
                coeffs=most_coeffs,
            )
            dataframe["u_star"] = self.calc_u_star(
                u=dataframe["wind_speed"],
                z_u=zm_bls,
                r_length=z0,
                o_length=dataframe["obukhov"],
            )
            dataframe["theta_star"] = self.calc_theta_star(
                ct2=dataframe["CT2"],
                f_ct2=dataframe["f_ct2"],
                z=zm_bls,
                stable=stable_flag,
            )
            obukhov_tmp = self.calc_obukhov_length(
                temp=dataframe["temperature_2m"],
                u_star=dataframe["u_star"],
                theta_star=dataframe["theta_star"],
            )

            # Use difference to determine convergence
            obukhov_diff = np.abs(obukhov_tmp - dataframe["obukhov"])
            dataframe["obukhov"] = obukhov_tmp

            if mpmath.isnan(dataframe["obukhov"]):  # iteration unnecessary if NaN
                iter_step = 10
            iter_step += 1

        # Calculate SHF after iteration
        dataframe["shf"] = (
            (-1)
            * dataframe["rho_air"]
            * self.constants.cp
            * dataframe["u_star"]
            * dataframe["theta_star"]
        )
        if stable_flag:
            dataframe["shf"] = -mpmath.fabs(dataframe["shf"])

        return dataframe

    def most_method(self, dataframe, eff_h, stability, coeff_id="an1988"):
        """Calculate Obukhov length and sensible heat fluxes with MOST.

        Iteration is more accurate but slower than vectorisation.

        Implemented MOST coefficients:

        - **an1988**: E.L. Andreas (1988) [#andreas1988]_
        - **li2012**: D. Li et al. (2012) [#li2012]_
        - **wy1971**: Wyngaard et al. (1971) [#wyngaard1971]_
        - **wy1973**: Wyngaard et al. (1973) in Kooijmans and
          Hartogensis (2016) [#kooijmans2016]_
        - **ma2014**: Maronga et al. (2014) [#maronga2014]_
        - **br2014**: Braam et al. (2014) [#braam2014]_

        Braam et al. (2014) and Maronga et al. (2014) do not provide
        coefficients for stable conditions, so gradient functions will
        evaluate to zero for stable conditions.

        Args:
            dataframe (pd.DataFrame): Parsed, localised dataframe
                containing at least |CT2|, wind speed, air density,
                and temperature.
            eff_h (float): Effective path height.
            stability (str): Stability conditions. Either "stable" or
                "unstable".
            coeff_id (str): ID of MOST coefficients for unstable and
                stable conditions. Default "an1988".

        Returns:
            pd.DataFrame: Dataframe with additional columns for Obukhov
            lengths, sensible heat fluxes, frictional velocity, and
            temperature scale.
        """

        coeffs = self.get_most_coefficients(most_id=coeff_id, most_type="ct2")
        # Prepare dataframe
        iteration = dataframe.copy(deep=True)
        if stability == "unstable":
            iteration["obukhov"] = -100  # unstable conditions
            stable = False
        else:
            iteration["obukhov"] = 200  # stable conditions
            stable = True

        print(f"Started iteration ({stability})...")
        start = time.time()
        tqdm.tqdm.pandas()  # register pd.Series.map_apply with tqdm
        iteration = iteration.progress_apply(
            lambda x: self.most_iteration(
                x, zm_bls=eff_h, stable_flag=stable, most_coeffs=coeffs
            ),
            axis=1,
        )

        end = time.time()
        self.check_signs(stable_flag=stable, dataframe=iteration)
        print(f"Completed iteration ({stability}) in {end - start:>0.2f}s.\n")

        return iteration
