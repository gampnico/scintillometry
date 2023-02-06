"""Copyright 2019-2023 Helen Ward, Nicolas Gampierakis, Josef Zink.

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

For IterationMost, the original code was a Python port of a method
initially written in R by Dr. Helen Ward, but has since been modified.

The mathematical theory behind this iterative method is available in:

    - Scintec AG (2022). Scintec Scintillometers Theory Manual,
      Version 1.05. Scintec AG, Rottenburg, Germany.

    - Scintec AG (2008). Scintec Boundary Layer Scintillometer User
      Manual, Version 1.49. Scintec AG, Rottenburg, Germany.

This method has been updated to resolve some inconsistencies between
the manuals, and has additional modifications to enhance performance and
broaden its use with different scintillometer models. Results may
therefore differ slightly between this and previous iterative methods.

TODO:
    - ST-45: Implement MM5 method (Zhang and Anthes, 1982).
    - ST-46: Implement Li method (Li et al., 2014; 2015).
"""

import mpmath

from scintillometry.backend.constants import AtmosConstants


class IterationMost(AtmosConstants):
    """Classic MOST Iteration.

    A detailed description of the iterative scheme is available in:

    Scintec AG (2022). Scintec Scintillometers Theory Manual. Version
    1.05. Scintec AG, Rottenburg, Germany.

    This iterative method resolves some inconsistencies between
    different scintillometer manuals. Results may therefore differ
    slightly with other third-party implementations.

    Attributes:
        constants (AtmosConstants): Class containing atmospheric
            constants.
    """

    def __init__(self):
        super().__init__()
        mpmath.dps = 30
        mpmath.pretty = True
        self.constants = AtmosConstants()

    def momentum_stability_unstable(self, obukhov, z):
        """Integrated stability function for momentum, unstable.

        Uses Paulson (1970). Apply when |LOb| < 0.

        Args:
            obukhov (float): Obukhov length, |LOb| [m].
            z (float): Height, z [m].

        Returns:
            mpmath.mpf: Integrated stability function for momentum,
            |Ψm| [0-Dim].

        .. |LOb| replace:: L :sub:`Ob`
        .. |Ψm| replace:: Ψ :sub:`m`
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
            z (float): Height, z [m].

        Returns:
            mpmath.mpf: Integrated stability function for momentum,
            |Ψm| [0-Dim].
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
            z (float): Height, z [m].

        Returns:
            mpmath.mpf: Integrated stability function for momentum,
            |Ψm| [0-Dim].
        """

        if obukhov > 0:
            psi_m = self.momentum_stability_stable(obukhov, z)
        else:
            psi_m = self.momentum_stability_unstable(obukhov, z)

        return psi_m

    def get_most_coefficients(self, most_id="an1988", most_type="ct2"):
        """Returns MOST coefficients for similarity functions.

        Implemented MOST coefficients:

        * **an1988**: E.L. Andreas (1988), DOI: 10.1364/JOSAA.5.000481
        * **li2012**: D. Li et al. (2012),
          DOI: 10.1007/s10546-011-9660-y
        * **wy1971**: Wyngaard et al. (1971),
          DOI: 10.1364/JOSA.61.001646
        * **wy1973**: Wyngaard et al. (1973) in Kooijmans and
          Hartogensis (2016), DOI: 10.1007/s10546-016-0152-y
        * **ma2014**: Maronga et al. (2014),
          DOI: 10.1007/s10546-014-9955-x
        * **br2014**: Braam et al. (2014),
          DOI: 10.1007/s10546-014-9938-y

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

        Adapted from Wyngaard et al. (1971); Thiermann and Grassl, 1992;
        Kooijmans and Hartogensis, 2016.

        Args:
            obukhov (float): Obukhov length, |LOb| [m].
            z (float): Effective height, z [m].
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
